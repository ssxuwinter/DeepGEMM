#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../heuristics/sm90.hpp"

#include "epilogue.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM90FP8Gemm1D2DRuntime final: public LaunchRuntime<SM90FP8Gemm1D2DRuntime> {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        LaunchArgs launch_args;
        // TODO: move this into `gemm_desc`
        const std::optional<std::string>& epilogue_type;

        cute::UMMA::Major major_sfb;
        void *sfb, *grouped_layout;
        // Raw pointers for cp.async load path
        void *gmem_a;
        uint32_t stride_a;    // row stride of A in fp8 elements
        void *gmem_sfa;
        uint32_t stride_sfa;  // row stride of sfa in float elements (= sf_k for K-major row-major)
        void *gather_index;   // optional int32 row remap for A/sfa; nullptr keeps logical rows
        // Per-rank ready-flag overlap (optional): set all three together. See
        // docs/sm90_fp8_gemm_1d2d_gather_index_rank_overlap.md for the contract.
        void *rank_flags;     // (num_ranks,) int64 on device; nullptr disables overlap
        void *tile_rank;      // (ceil(m/BLOCK_M),) int32 on device; nullptr disables overlap
        uint32_t num_ranks;   // <= 8 (kNumRanksMax in kernel); 0 when overlap is off
        bool sfa_is_kmajor;   // true = K-major [m, sf_k] (overlap path); false = MN-major [sf_k, m] (coalescing)
        // TMA descriptors kept for reference (A/sfa currently unused in kernel)
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_d;
        CUtensorMap tensor_map_sfa;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_gemm_1d2d_impl<
        {},
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {},
        {}, {},
        {}, {},
        {}
    >);
}};
)",
        // TODO: add CD dtype
        to_string(args.major_sfb),
        get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.k, 'k', args.gemm_desc.compiled_dims),
        args.gemm_desc.num_groups,
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n, args.gemm_config.layout.block_k,
        args.gemm_config.storage_config.swizzle_a_mode, args.gemm_config.storage_config.swizzle_b_mode, args.gemm_config.storage_config.swizzle_cd_mode,
        args.gemm_config.pipeline_config.num_stages,
        args.gemm_config.launch_config.num_tma_threads, args.gemm_config.launch_config.num_math_threads,
        args.gemm_config.layout.get_cluster_size(), args.gemm_config.layout.cluster_n > 1,
        args.gemm_config.launch_config.num_sms, to_string(args.gemm_desc.gemm_type),
        get_default_epilogue_type(args.epilogue_type));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // TODO: optimize `args` copy
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.sfb, args.grouped_layout,
            args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k,
            args.gmem_a, args.stride_a,
            args.gmem_sfa, args.stride_sfa,
            args.gather_index,
            args.rank_flags, args.tile_rank, args.num_ranks,
            args.sfa_is_kmajor,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_d, args.tensor_map_sfa));
    }
};

static void sm90_fp8_gemm_1d2d(const torch::Tensor& a, const torch::Tensor& sfa,
                               const torch::Tensor& b, const torch::Tensor& sfb,
                               const std::optional<torch::Tensor>& c,
                               const torch::Tensor& d,
                               const int& m, const int& n, const int& k,
                               const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b, const cute::UMMA::Major& major_sfb,
                               const std::string& compiled_dims,
                               const std::optional<std::string>& epilogue_type = std::nullopt,
                               const std::optional<torch::Tensor>& gather_index = std::nullopt,
                               // Per-rank ready-flag overlap, see
                               // docs/sm90_fp8_gemm_1d2d_gather_index_rank_overlap.md.
                               // All three must be set together (else all unset).
                               const std::optional<torch::Tensor>& rank_flags = std::nullopt,
                               const std::optional<torch::Tensor>& tile_rank = std::nullopt,
                               const std::optional<int>& num_ranks = std::nullopt) {
    DG_HOST_ASSERT(not c.has_value() and d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    if (gather_index.has_value()) {
        DG_HOST_ASSERT(gather_index->is_cuda() and gather_index->is_contiguous());
        DG_HOST_ASSERT(gather_index->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(gather_index->numel() >= m);
        // When gather_index is set, A may carry m_pool >= m rows from which the
        // kernel will gather m logical output rows. SFA must already be
        // transformed for the same m_pool (the caller passes the layout-
        // transformed sfa whose stride(-1) reflects this).
        DG_HOST_ASSERT(static_cast<int>(a.size(0)) >= m);
    }
    // Per-rank overlap: the three optional inputs must be coherent. Either all
    // three are set (overlap enabled) or all unset (overlap disabled).
    const bool has_overlap = rank_flags.has_value();
    if (has_overlap) {
        DG_HOST_ASSERT(tile_rank.has_value() and num_ranks.has_value());
        DG_HOST_ASSERT(gather_index.has_value() and "rank overlap requires gather_index");
        DG_HOST_ASSERT(rank_flags->is_cuda() and rank_flags->is_contiguous());
        DG_HOST_ASSERT(rank_flags->scalar_type() == torch::kLong);
        DG_HOST_ASSERT(tile_rank->is_cuda() and tile_rank->is_contiguous());
        DG_HOST_ASSERT(tile_rank->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(num_ranks.value() > 0 and num_ranks.value() <= 8);
        DG_HOST_ASSERT(rank_flags->numel() >= num_ranks.value());
        // tile_rank size check is deferred until BLOCK_M is known (after
        // get_best_config below).
    } else {
        DG_HOST_ASSERT(not tile_rank.has_value() and not num_ranks.has_value());
    }

    const auto desc = GemmDesc {
        .gemm_type = GemmType::Normal,
        .kernel_type = KernelType::Kernel1D2D,
        .m = m, .n = n, .k = k, .num_groups = 1,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims
    };
    const auto config = get_best_config<SM90ArchSpec>(desc);

    // Requires no TMA splits
    DG_HOST_ASSERT(config.storage_config.swizzle_a_mode == config.layout.block_k);
    DG_HOST_ASSERT(config.storage_config.swizzle_b_mode == config.layout.block_k);
    // Now that `BLOCK_M` is known, finish validating tile_rank for the rank-overlap path.
    if (has_overlap) {
        const int num_m_tiles = (m + config.layout.block_m - 1) / config.layout.block_m;
        DG_HOST_ASSERT(tile_rank->numel() >= num_m_tiles);
    }
    // cp.async path: SFA layout depends on has_overlap (= sfa_is_kmajor).
    // has_overlap=true  → K-major [m, sf_k]:   stride(-1)==1,  stride_sfa = sf_k.
    // has_overlap=false → MN-major [sf_k, m]:  stride(-2)==1,  stride_sfa = tma_aligned_m.
    uint32_t stride_sfa_elems;
    if (has_overlap) {
        DG_HOST_ASSERT(sfa.stride(-1) == 1);
        DG_HOST_ASSERT(sfa.stride(-2) == sfa.size(-1));
        stride_sfa_elems = static_cast<uint32_t>(sfa.stride(-2));
    } else {
        DG_HOST_ASSERT(sfa.stride(-2) == 1);
        DG_HOST_ASSERT(sfa.stride(-1) == sfa.size(-2));
        stride_sfa_elems = static_cast<uint32_t>(sfa.stride(-1));
    }
    const uint32_t stride_a_elems = static_cast<uint32_t>(a.stride(0));
    // TMA descriptors (A/sfa kept for reference, currently unused in kernel)
    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), 1,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_d = make_tma_cd_desc(d, m, static_cast<int>(d.size(-1)),
                                               config.storage_config.store_block_m,
                                               config.storage_config.store_block_n,
                                               static_cast<int>(d.stride(-2)), 1,
                                               config.storage_config.swizzle_cd_mode);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, config.layout.block_k, 1, 0);

    // Launch
    const SM90FP8Gemm1D2DRuntime::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .epilogue_type = epilogue_type,
        .major_sfb = major_sfb,
        .sfb = sfb.data_ptr(),
        .grouped_layout = nullptr,
        .gmem_a = a.data_ptr(),
        .stride_a = stride_a_elems,
        .gmem_sfa = sfa.data_ptr(),
        .stride_sfa = stride_sfa_elems,
        .gather_index = gather_index.has_value() ? gather_index->data_ptr() : nullptr,
        .rank_flags = has_overlap ? rank_flags->data_ptr() : nullptr,
        .tile_rank = has_overlap ? tile_rank->data_ptr() : nullptr,
        .num_ranks = has_overlap ? static_cast<uint32_t>(num_ranks.value()) : 0u,
        .sfa_is_kmajor = has_overlap,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d,
        .tensor_map_sfa = tensor_map_sfa,
    };
    const auto code = SM90FP8Gemm1D2DRuntime::generate(args);
    const auto runtime = compiler->build("sm90_fp8_gemm_1d2d", code);
    SM90FP8Gemm1D2DRuntime::launch(runtime, args);
}

static void sm90_m_grouped_fp8_gemm_contiguous_1d2d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                    const torch::Tensor& b, const torch::Tensor& sfb,
                                                    const torch::Tensor& d,
                                                    const torch::Tensor& m_indices,
                                                    const int& num_groups, const int& m, const int& n, const int& k,
                                                    const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b, const cute::UMMA::Major& major_sfb,
                                                    const std::string& compiled_dims,
                                                    const bool& use_psum_layout,
                                                    const std::optional<int>& expected_m_for_psum_layout,
                                                    const std::optional<torch::Tensor>& gather_index = std::nullopt,
                                                    // Per-rank ready-flag overlap (optional). See
                                                    // docs/sm90_fp8_gemm_1d2d_gather_index_rank_overlap.md.
                                                    const std::optional<torch::Tensor>& rank_flags = std::nullopt,
                                                    const std::optional<torch::Tensor>& tile_rank = std::nullopt,
                                                    const std::optional<int>& num_ranks = std::nullopt) {
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    if (gather_index.has_value()) {
        DG_HOST_ASSERT(gather_index->is_cuda() and gather_index->is_contiguous());
        DG_HOST_ASSERT(gather_index->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(gather_index->numel() >= m);
        // For m-grouped contiguous + gather, A is the ORIGINAL (un-permuted) token
        // pool shaped (m_pool, k); m_pool may differ from m (the grouped output
        // row count). Caller is responsible for ensuring gather_index values
        // stay within [0, m_pool).
    }
    // Per-rank overlap is only valid alongside gather_index. Same triple-validity
    // contract as in `sm90_fp8_gemm_1d2d`.
    const bool has_overlap = rank_flags.has_value();
    if (has_overlap) {
        DG_HOST_ASSERT(tile_rank.has_value() and num_ranks.has_value());
        DG_HOST_ASSERT(gather_index.has_value() and "rank overlap requires gather_index");
        DG_HOST_ASSERT(rank_flags->is_cuda() and rank_flags->is_contiguous());
        DG_HOST_ASSERT(rank_flags->scalar_type() == torch::kLong);
        DG_HOST_ASSERT(tile_rank->is_cuda() and tile_rank->is_contiguous());
        DG_HOST_ASSERT(tile_rank->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(num_ranks.value() > 0 and num_ranks.value() <= 8);
        DG_HOST_ASSERT(rank_flags->numel() >= num_ranks.value());
    } else {
        DG_HOST_ASSERT(not tile_rank.has_value() and not num_ranks.has_value());
    }

    const auto gemm_type = use_psum_layout ?
        GemmType::MGroupedContiguousWithPsumLayout : GemmType::MGroupedContiguous;

    // Only psum layout can use expected m
    if (expected_m_for_psum_layout)
        DG_HOST_ASSERT(use_psum_layout);

    const auto desc = GemmDesc {
        .gemm_type = gemm_type,
        .kernel_type = KernelType::Kernel1D2D,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims,
        .expected_m = expected_m_for_psum_layout.value_or(m),
        .expected_n = n, .expected_k = k,
        .expected_num_groups = expected_m_for_psum_layout.has_value() ? num_groups : 1
    };
    const auto config = get_best_config<SM90ArchSpec>(desc);

    // Requires no TMA splits
    DG_HOST_ASSERT(config.storage_config.swizzle_a_mode == config.layout.block_k);
    DG_HOST_ASSERT(config.storage_config.swizzle_b_mode == config.layout.block_k);
    if (has_overlap) {
        const int num_m_tiles = (m + config.layout.block_m - 1) / config.layout.block_m;
        DG_HOST_ASSERT(tile_rank->numel() >= num_m_tiles);
    }
    // cp.async path: SFA layout depends on has_overlap (= sfa_is_kmajor).
    // has_overlap=true  → K-major [m, sf_k]:  stride(-1)==1,  stride_sfa = sf_k.
    // has_overlap=false → MN-major [sf_k, m]: stride(-2)==1,  stride_sfa = tma_aligned_m.
    uint32_t stride_sfa_elems;
    if (has_overlap) {
        DG_HOST_ASSERT(sfa.stride(-1) == 1);
        DG_HOST_ASSERT(sfa.stride(-2) == sfa.size(-1));
        stride_sfa_elems = static_cast<uint32_t>(sfa.stride(-2));
    } else {
        DG_HOST_ASSERT(sfa.stride(-2) == 1);
        DG_HOST_ASSERT(sfa.stride(-1) == sfa.size(-2));
        stride_sfa_elems = static_cast<uint32_t>(sfa.stride(-1));
    }
    const uint32_t stride_a_elems = static_cast<uint32_t>(a.stride(0));
    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_d = make_tma_cd_desc(d, m, n,
                                               config.storage_config.store_block_m,
                                               config.storage_config.store_block_n,
                                               static_cast<int>(d.stride(-2)), 1,
                                               config.storage_config.swizzle_cd_mode);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, config.layout.block_k, 1, 0);

    // Launch
    const SM90FP8Gemm1D2DRuntime::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .epilogue_type = std::nullopt,
        .major_sfb = major_sfb,
        .sfb = sfb.data_ptr(),
        .grouped_layout = m_indices.data_ptr(),
        .gmem_a = a.data_ptr(),
        .stride_a = stride_a_elems,
        .gmem_sfa = sfa.data_ptr(),
        .stride_sfa = stride_sfa_elems,
        .gather_index = gather_index.has_value() ? gather_index->data_ptr() : nullptr,
        .rank_flags = has_overlap ? rank_flags->data_ptr() : nullptr,
        .tile_rank = has_overlap ? tile_rank->data_ptr() : nullptr,
        .num_ranks = has_overlap ? static_cast<uint32_t>(num_ranks.value()) : 0u,
        .sfa_is_kmajor = has_overlap,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d,
        .tensor_map_sfa = tensor_map_sfa,
    };
    const auto code = SM90FP8Gemm1D2DRuntime::generate(args);
    const auto runtime = compiler->build("sm90_m_grouped_fp8_gemm_contiguous_1d2d", code);
    SM90FP8Gemm1D2DRuntime::launch(runtime, args);
}

static void sm90_m_grouped_fp8_gemm_masked_1d2d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                const torch::Tensor& b, const torch::Tensor& sfb,
                                                const torch::Tensor& d,
                                                const torch::Tensor& masked_m,
                                                const int& num_groups, const int& m, const int& n, const int& k,
                                                const int& expected_m,
                                                const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b, const cute::UMMA::Major& major_sfb,
                                                const std::string& compiled_dims) {
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    const auto desc = GemmDesc {
        .gemm_type = GemmType::MGroupedMasked,
        .kernel_type = KernelType::Kernel1D2D,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims,
        .expected_m = expected_m, .expected_n = n, .expected_k = k, .expected_num_groups = num_groups
    };
    const auto config = get_best_config<SM90ArchSpec>(desc);

    // Requires no TMA splits
    DG_HOST_ASSERT(config.storage_config.swizzle_a_mode == config.layout.block_k);
    DG_HOST_ASSERT(config.storage_config.swizzle_b_mode == config.layout.block_k);
    DG_HOST_ASSERT(sfa.stride(-1) == 1);
    DG_HOST_ASSERT(sfa.stride(-2) == sfa.size(-1));
    const uint32_t stride_a_elems = static_cast<uint32_t>(a.stride(get_non_contiguous_dim(major_a)));
    const uint32_t stride_sfa_elems = static_cast<uint32_t>(sfa.stride(-2));
    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), num_groups,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_d = make_tma_cd_desc(d, m, n,
                                               config.storage_config.store_block_m,
                                               config.storage_config.store_block_n,
                                               static_cast<int>(d.stride(-2)), num_groups,
                                               config.storage_config.swizzle_cd_mode);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, config.layout.block_k, num_groups, 0);

    // Launch
    const SM90FP8Gemm1D2DRuntime::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .epilogue_type = std::nullopt,
        .major_sfb = major_sfb,
        .sfb = sfb.data_ptr(),
        .grouped_layout = masked_m.data_ptr(),
        .gmem_a = a.data_ptr(),
        .stride_a = stride_a_elems,
        .gmem_sfa = sfa.data_ptr(),
        .stride_sfa = stride_sfa_elems,
        .gather_index = nullptr,
        .rank_flags = nullptr,
        .tile_rank = nullptr,
        .num_ranks = 0u,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d,
        .tensor_map_sfa = tensor_map_sfa,
    };
    const auto code = SM90FP8Gemm1D2DRuntime::generate(args);
    const auto runtime = compiler->build("sm90_fp8_m_grouped_gemm_masked_1d2d", code);
    SM90FP8Gemm1D2DRuntime::launch(runtime, args);
}

static void sm90_fp8_bmm(const torch::Tensor& a, const torch::Tensor& sfa,
                         const torch::Tensor& b, const torch::Tensor& sfb,
                         const std::optional<torch::Tensor>& c,
                         const torch::Tensor& d,
                         const int& batch_size, const int& m, const int& n, const int& k,
                         const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b, const cute::UMMA::Major& major_sfb,
                         const std::string& compiled_dims) {
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    const auto desc = GemmDesc {
        .gemm_type = GemmType::Batched,
        .kernel_type = KernelType::Kernel1D2D,
        .m = m, .n = n, .k = k, .num_groups = batch_size,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims
    };
    const auto config = get_best_config<SM90ArchSpec>(desc);

    // Requires no TMA splits
    DG_HOST_ASSERT(config.storage_config.swizzle_a_mode == config.layout.block_k);
    DG_HOST_ASSERT(config.storage_config.swizzle_b_mode == config.layout.block_k);
    // For BMM: a shape is [batch_size, m, k], stride along m-dim = k (fp8 elements)
    DG_HOST_ASSERT(sfa.stride(-1) == 1);
    DG_HOST_ASSERT(sfa.stride(-2) == sfa.size(-1));
    const uint32_t stride_a_elems = static_cast<uint32_t>(a.stride(1));
    const uint32_t stride_sfa_elems = static_cast<uint32_t>(sfa.stride(-2));

    const int load_block_m = config.storage_config.load_block_m;
    const auto tensor_map_a = make_tma_3d_desc(a, k, m, batch_size,
                                               config.layout.block_k, load_block_m, 1,
                                               a.stride(1),
                                               a.stride(0),
                                               config.storage_config.swizzle_a_mode);

    const int load_block_n = config.storage_config.load_block_n;
    const auto tensor_map_b = make_tma_3d_desc(b, k, n, batch_size,
                                               config.layout.block_k, load_block_n, 1,
                                               b.stride(1),
                                               b.stride(0),
                                               config.storage_config.swizzle_b_mode);

    const int store_block_m = config.storage_config.store_block_m;
    const int store_block_n = config.storage_config.store_block_n;
    const auto tensor_map_d = make_tma_3d_desc(d, n, m, batch_size,
                                               store_block_n, store_block_m, 1,
                                               d.stride(1), d.stride(0),
                                               config.storage_config.swizzle_cd_mode);

    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, config.layout.block_k, batch_size, 0);

    // Launch
    const SM90FP8Gemm1D2DRuntime::Args& args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .epilogue_type = std::nullopt,
        .major_sfb = major_sfb,
        .sfb = sfb.data_ptr(),
        .grouped_layout = nullptr,
        .gmem_a = a.data_ptr(),
        .stride_a = stride_a_elems,
        .gmem_sfa = sfa.data_ptr(),
        .stride_sfa = stride_sfa_elems,
        .gather_index = nullptr,
        .rank_flags = nullptr,
        .tile_rank = nullptr,
        .num_ranks = 0u,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d,
        .tensor_map_sfa = tensor_map_sfa,
    };
    const auto code = SM90FP8Gemm1D2DRuntime::generate(args);
    const auto runtime = compiler->build("sm90_fp8_gemm_1d2d", code);
    SM90FP8Gemm1D2DRuntime::launch(runtime, args);
}

} // namespace deep_gemm
