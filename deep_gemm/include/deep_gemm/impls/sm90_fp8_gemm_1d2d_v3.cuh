#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/scheduler/gemm.cuh>
template <typename T, typename U>
__device__ __forceinline__ void cp_async4(T* smem_ptr, const U* glob_ptr) {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{ cp.async.cg.shared.global [%0], [%1], %2; }\n"
        :: "r"(smem), "l"((const void*)glob_ptr), "n"(BYTES));
#else
    *smem_ptr = *reinterpret_cast<const T*>(glob_ptr);
#endif
}
namespace deep_gemm {

template <uint32_t kNumFormerIters, uint32_t kGap, uint32_t kEnd, typename func_t>
CUTLASS_DEVICE void dispatch_num_former_iters(uint32_t num_former_iters, const func_t& func) {
    if (num_former_iters == kNumFormerIters) {
        func(cute::Int<kNumFormerIters>{});
        return;
    }

    if constexpr (kNumFormerIters + kGap <= kEnd)
        dispatch_num_former_iters<kNumFormerIters + kGap, kGap, kEnd>(num_former_iters, func);
}

// ---------------------------------------------------------------------------
// Barrier debug helpers — throttled edition
//
// Controls (all have compile-time defaults, override via -D flag if needed):
//   DG_BARRIER_DEBUG    : 0=off, 1=on  (default 1)
//   DG_DBG_BLOCK_ID     : only this blockIdx.x prints  (default 0)
//   DG_DBG_MAX_STAGES   : only stage < this value prints  (default 2)
//   DG_DBG_PRINT_TID    : which threadIdx.x inside each WG is the "reporter"
//                         TMA WG reporter  = kNumMathThreads + DG_DBG_PRINT_TID  (default 0)
//                         CPA WG reporter  = kNumMathThreads + 128 + DG_DBG_PRINT_TID
//                         Math WG reporter = DG_DBG_PRINT_TID
//
// Net effect: at most (DG_DBG_MAX_STAGES * <num_print_sites>) lines per run —
// easily fits in the 1 MB CUDA printf buffer.
// ---------------------------------------------------------------------------
#ifndef DG_BARRIER_DEBUG
#define DG_BARRIER_DEBUG 0
#endif
#ifndef DG_DBG_BLOCK_ID
#define DG_DBG_BLOCK_ID 0
#endif
#ifndef DG_DBG_MAX_STAGES
#define DG_DBG_MAX_STAGES 4u
#endif
#ifndef DG_DBG_PRINT_TID
#define DG_DBG_PRINT_TID 0
#endif

#if DG_BARRIER_DEBUG
// Print barrier pointer info.
// DG_BARRIER_DEBUG=1: only print generic pointer address (safe, no shared mem access).
// DG_BARRIER_DEBUG=2: also read and decode the mbarrier word (requires valid smem layout).
__device__ __forceinline__ void dg_print_barrier(
        const char* tag,
        uint32_t blockX, uint32_t tidX,
        uint32_t stage,
        const cutlass::arch::ClusterTransactionBarrier* ptr,
        uint32_t wait_phase) {
    // Use generic address (uintptr_t) to avoid any cvta instruction that could fault.
    const uintptr_t generic_addr = reinterpret_cast<uintptr_t>(ptr);
#if DG_BARRIER_DEBUG >= 2
    // PTX mbarrier layout (SM90):
    //   bit 0     : phase/parity bit (current phase)
    //   bits 1-11 : pending arrive count  (how many more arrives needed)
    //   bits 12-31: pending tx bytes      (how many more bytes the HW must deliver)
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t raw = 0;
    asm volatile("ld.shared.b64 %0, [%1];" : "=l"(raw) : "r"(smem_addr));
    uint32_t parity       = (uint32_t)(raw & 1u);
    uint32_t arrive_rem   = (uint32_t)((raw >> 1) & 0x7FFu);
    uint32_t tx_rem       = (uint32_t)((raw >> 12) & 0xFFFFFu);
    printf("[DBG-BARRIER] blk=%u tid=%u  %-22s  stage=%u  ptr=0x%llx"
           "  raw=0x%012llx  parity=%u(wait=%u)  arrive_rem=%u  tx_rem=%u\n",
           blockX, tidX, tag, stage, (unsigned long long)generic_addr,
           (unsigned long long)raw,
           parity, wait_phase, arrive_rem, tx_rem);
#else
    printf("[DBG-BARRIER] blk=%u tid=%u  %-22s  stage=%u  ptr=0x%llx  wait_phase=%u\n",
           blockX, tidX, tag, stage, (unsigned long long)generic_addr, wait_phase);
#endif
}
// Guard: only block DG_DBG_BLOCK_ID, only stage < DG_DBG_MAX_STAGES,
//        only thread threadIdx.x == reporter_tid (passed by caller).
// reporter_tid is different per WG so the macro takes it explicitly.
#define DG_DBG_BARRIER(tag, stage, ptr, wphase, reporter_tid)          \
    do {                                                                \
        if (blockIdx.x == (uint32_t)DG_DBG_BLOCK_ID &&                 \
            (stage) < DG_DBG_MAX_STAGES &&                             \
            threadIdx.x == (uint32_t)(reporter_tid)) {                 \
            dg_print_barrier((tag), blockIdx.x, threadIdx.x,           \
                             (stage), (ptr), (wphase));                \
        }                                                               \
    } while (0)
#else
#define DG_DBG_BARRIER(tag, stage, ptr, wphase, reporter_tid) ((void)0)
#endif

template <cute::UMMA::Major kMajorSFB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleDMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs, GemmType kGemmType,
          typename epilogue_type_t>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_fp8_gemm_1d2d_impl(float* sfb, int* grouped_layout,
                        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                        const __nv_fp8_e4m3* __restrict__ gmem_a,
                        uint32_t stride_a,          // row stride of A in fp8 elements
                        const float* __restrict__ gmem_sfa,
                        uint32_t stride_sfa,        // row stride of sfa in float elements (= M for MN-major)
                        const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_d,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_sfa) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Scaling checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(
        math::constexpr_ceil_div(BLOCK_N, BLOCK_K) == 1 or
        (math::constexpr_gcd(BLOCK_N, BLOCK_K) == BLOCK_N - BLOCK_K), "Too much B scales in a single block");

    // Types
    using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0 or BLOCK_M < WGMMA::M, "Invalid block size");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Shared memory
    static constexpr bool kMustUseUniformedScaleB = (BLOCK_K % BLOCK_N == 0);
    static constexpr uint32_t SMEM_D_SIZE = math::constexpr_align(BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(__nv_bfloat16)), 1024u);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_SFA_SIZE_PER_STAGE = math::constexpr_align(SMEM_SFA_SIZE_PER_STAGE, 128u);
    const uint32_t shape_k_scales = math::ceil_div(shape_k, BLOCK_K);
    const uint32_t shape_n_sfb = math::ceil_div(shape_n, BLOCK_K);
    const uint32_t smem_sfb_size = math::align<uint32_t>(shape_k_scales * (kMustUseUniformedScaleB ? 1 : 2) * sizeof(float), sizeof(Barrier));

    // NOTES: Make sure we have enough shared memory for WGMMA padding
    static constexpr uint32_t WGMMA_A_SIZE_PER_STAGE = WGMMA::M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    DG_STATIC_ASSERT(WGMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory Out of bound for WGMMA");

    // Configs
    const uint32_t num_total_k_blocks = math::ceil_div(shape_k, BLOCK_K);
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // Data on shared memory
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    constexpr uint32_t SMEM_SF_OFFSET = SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + i * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + kNumStages * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);

    // Fill barriers
    // Layout: [full_barriers_b(0..kNumStages-1)] [full_barriers_a(kNumStages..2*kNumStages-1)] [empty_barriers(2*kNumStages..3*kNumStages-1)]
    // full_barriers_b: produced by TMA WG (B + sfa)
    // full_barriers_a: produced by cp.async WG (A)
    // empty_barriers:  signaled by Math WG after consuming
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_sfb) + smem_sfb_size);
    auto full_barriers_b   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + i; });
    auto full_barriers_a   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });
    auto empty_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + 2 * kNumStages + i; });

#if DG_BARRIER_DEBUG
    // Diagnose SMEM layout: print offsets and total usage (block 0, thread 0 only).
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        uint32_t smem_base      = static_cast<uint32_t>(__cvta_generic_to_shared(smem_buffer));
        uint32_t barrier_off    = static_cast<uint32_t>(
            reinterpret_cast<uint8_t*>(barrier_start_ptr) - smem_buffer);
        uint32_t barrier_end_off= barrier_off + 3u * kNumStages * 8u;
        printf("[DBG-SMEM-LAYOUT] blk=0 tid=0  smem_base=0x%x"
               "  SMEM_D=%u A_per=%u B_per=%u SFA_per=%u"
               "  sfb_size=%u  barrier_off=%u barrier_end=%u"
               "  kNumStages=%u\n",
               smem_base,
               (unsigned)SMEM_D_SIZE,
               (unsigned)SMEM_A_SIZE_PER_STAGE,
               (unsigned)SMEM_B_SIZE_PER_STAGE,
               (unsigned)ALIGNED_SMEM_SFA_SIZE_PER_STAGE,
               (unsigned)smem_sfb_size,
               barrier_off, barrier_end_off,
               (unsigned)kNumStages);
    }
#endif

    // Initialize barriers
    DG_STATIC_ASSERT(kNumTMAMulticast <= 32, "Too many TMA multicast");
    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        // NOTES: we always use `lane_idx` to arrive for the `lane_idx`-th CTA in the cluster,
        // even with TMA multicast disabled, we want to make the behavior aligned
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers_b[i]->init(1);  // TMA WG is sole producer for B + sfa
            full_barriers_a[i]->init(1);  // cp.async WG is sole producer for A
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters      = 24;   // TMA WG: lightweight, only issues TMA
    constexpr uint32_t kNumCpAsyncRegisters  = 40;   // cp.async WG: needs address arithmetic
    constexpr uint32_t kNumMathRegisters     = kNumMathThreads == 128 ? 248 : 216;

    static constexpr uint32_t kCpAsyncWidth   = 16;
    static constexpr uint32_t kCpAsyncThreads = 128;
    static constexpr uint32_t kAItersPerThread = SMEM_A_SIZE_PER_STAGE / kCpAsyncThreads / kCpAsyncWidth;
    static_assert(kAItersPerThread >= 1,
                  "Not enough cp.async threads for A load");
    static_assert(SMEM_A_SIZE_PER_STAGE % (kCpAsyncThreads * kCpAsyncWidth) == 0,
                  "A smem size must be divisible by (128 * 16)");

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs>(shape_m, shape_n, shape_k, grouped_layout);

    // Pipeline phase tracker
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // -------------------------------------------------------------------------
    // TMA WG: threadIdx.x in [kNumMathThreads, kNumMathThreads + 127]
    //         warp_idx in [kNumMathThreads/32, kNumMathThreads/32 + 3]
    // Responsible for: loading B and sfa via TMA, signaling full_barriers_b
    // -------------------------------------------------------------------------
    if (warp_idx >= kNumMathThreads / 32 and warp_idx < kNumMathThreads / 32 + 4) {
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            constexpr bool kWithGroupOffsetA = kGemmType == GemmType::MGroupedMasked;
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");

                // Wait for Math WG to finish consuming this stage
                empty_barriers[stage_idx]->wait(phase ^ 1);

                constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                const uint32_t batch_idx    = (kIsBatchedMM ? scheduler.current_group_idx : 0);
                const uint32_t k_idx        = k_block_idx * BLOCK_K;
                auto& full_b = *full_barriers_b[stage_idx];

                // Only one elected thread issues TMA; the TMA descriptor will trigger
                // full_b to arrive_and_expect_tx when data lands in smem
                if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
                    tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode, __nv_fp8_e4m3, kIsBatchedMM>(&tensor_map_b, &full_b,
                             smem_b[stage_idx],
                             k_idx,
                             scheduler.get_global_idx<true>(shape_n, BLOCK_N, n_block_idx, m_block_idx),
                             num_tma_multicast_b, batch_idx);
                    full_b.arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE);
                    // NOTE: No printf here — TMA WG only has 24 registers (warpgroup_reg_dealloc),
                    //       not enough for printf/vfprintf_internal which needs ~30-50 registers.
                }
            }
            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1) {
                for (uint32_t i = 0; i < kNumStages; advance_pipeline(i))
                    empty_barriers[stage_idx]->wait(phase ^ 1);
            }
        }

    // -------------------------------------------------------------------------
    // cp.async WG: threadIdx.x in [kNumMathThreads + 128, kNumMathThreads + 255]
    //              warp_idx in [kNumMathThreads/32 + 4, kNumMathThreads/32 + 7]
    // Responsible for: loading A via cp.async with swizzle-128B, signaling full_barriers_a
    // -------------------------------------------------------------------------
    } else if (warp_idx >= kNumMathThreads / 32 + 4) {
        cutlass::arch::warpgroup_reg_dealloc<kNumCpAsyncRegisters>();

        const uint32_t tid_in_cpasync_wg = threadIdx.x - kNumMathThreads - 128;

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            constexpr bool kWithGroupOffsetA = kGemmType == GemmType::MGroupedMasked;
            const uint32_t m_global_base = scheduler.get_global_idx<kWithGroupOffsetA>(shape_m, BLOCK_M, m_block_idx);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait for Math WG to finish consuming this stage
                empty_barriers[stage_idx]->wait(phase ^ 1);

                const uint32_t k_idx = k_block_idx * BLOCK_K;
                __nv_fp8_e4m3* dst_a = smem_a[stage_idx];

                // Load A tile with swizzle-128B layout
                // Each 16B cp.async covers one contiguous chunk; we swizzle col XOR (row%8)*16
                // to match the WGMMA layout_type=1 (B128 swizzle) descriptor
                #pragma unroll
                for (uint32_t i = 0; i < kAItersPerThread; ++i) {
                    const uint32_t linear = (tid_in_cpasync_wg * kAItersPerThread + i) * kCpAsyncWidth;
                    const uint32_t row = linear / BLOCK_K;
                    const uint32_t col = linear % BLOCK_K;
                    cp_async4(dst_a + row * BLOCK_K + (col ^ ((row % 8) * 16)),
                              gmem_a + (m_global_base + row) * stride_a + k_idx + col);
                }
                if (tid_in_cpasync_wg < BLOCK_M) {
                    const uint32_t sfa_k_idx = scheduler.template get_global_idx<kWithGroupOffsetA, sched::IndexType::SF_K>(shape_k_scales, 1, k_block_idx);
                    float* dst_sfa = smem_sfa[stage_idx] + tid_in_cpasync_wg;
                    const float* src_sfa = gmem_sfa + (m_global_base + tid_in_cpasync_wg) * stride_sfa + sfa_k_idx;
                    asm volatile(
                        "cp.async.ca.shared.global [%0], [%1], %2;\n"
                        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_sfa))),
                            "l"(reinterpret_cast<const void*>(src_sfa)),
                            "n"(4));
                }

                // Commit and wait for all cp.async in this group to complete
                asm volatile("cp.async.commit_group;\n" ::: "memory");
                asm volatile("cp.async.wait_group 0;\n"  ::: "memory");
                cutlass::arch::NamedBarrier::sync(128, 2);
                // One thread signals that A is ready for this stage
                if (tid_in_cpasync_wg == 0) {
                    full_barriers_a[stage_idx]->arrive();
                    // NOTE: No printf here — cp.async WG only has 40 registers (warpgroup_reg_dealloc),
                    //       not enough for printf/vfprintf_internal which needs ~30-50 registers.
                }
            }
            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1) {
                for (uint32_t i = 0; i < kNumStages; advance_pipeline(i))
                    empty_barriers[stage_idx]->wait(phase ^ 1);
            }
        }

    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        const auto r_0 = warp_idx * 16 + lane_idx / 4, r_1 = r_0 + 8;

        auto a_desc = mma::sm90::make_smem_desc(smem_a[0] + math_wg_idx * WGMMA::M * BLOCK_K, 1); 
        auto b_desc = mma::sm90::make_smem_desc(smem_b[0], 1);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Decide the number of scales B to load
            DG_TRAP_ONLY_DEVICE_ASSERT(shape_n % 8 == 0);
            uint32_t num_former_iters = BLOCK_N / 8, num_full_iters = num_former_iters;
            if constexpr (not kMustUseUniformedScaleB) {
                num_former_iters = min(BLOCK_N, BLOCK_K - n_block_idx * BLOCK_N % BLOCK_K) / 8;
                num_full_iters = min(shape_n - n_block_idx * BLOCK_N, BLOCK_N) / 8;
            }
            uint32_t num_sfb = shape_k_scales * (num_former_iters >= num_full_iters ? 1 : 2);

            // Load B scales with math warp-groups
            // NOTES: except the first warp, we want to overlap loading B scales with TMA stores between tasks
            if (threadIdx.x >= 32) {
                auto previous_group_offset = scheduler.template get_global_idx<true, sched::IndexType::SF_K>(shape_n_sfb * shape_k_scales, 0, 0, m_block_idx);
                const uint32_t stride_n_sfb = kMajorSFB == cute::UMMA::Major::MN ? 1 : shape_k_scales;
                const uint32_t stride_k_sfb = kMajorSFB == cute::UMMA::Major::MN ? shape_n_sfb : 1;
                auto local_sfb = sfb + previous_group_offset + ((n_block_idx * BLOCK_N) / BLOCK_K) * stride_n_sfb;

                #pragma unroll
                for (uint32_t i = threadIdx.x - 32; i < num_sfb; i += kNumMathThreads - 32)
                    ptx::st_shared(smem_sfb + i, i < shape_k_scales ? local_sfb[i * stride_k_sfb] : local_sfb[(i - shape_k_scales) * stride_k_sfb + stride_n_sfb]);
            }
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

            // Accumulation for WGMMA or CUDA promotion
            constexpr uint32_t WAVE_BLOCK_M = BLOCK_M <= WGMMA::M ? BLOCK_M : WGMMA::M * 2;
            DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block sizes");
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};
            
            // Pick threads whose WGMMA results are to be stored in shared memory
            DG_STATIC_ASSERT(BLOCK_M >= 64 or kNumMathThreads == 128, "Only one math warp group for `BLOCK_M < 64`");
            constexpr uint32_t kNumWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA::M);
            const bool do_wgmma_store = BLOCK_M >= WGMMA::M or warp_idx < kNumWGMMAStoreThreads / 32;

            // Empty barrier arrival
            auto empty_barrier_arrive = [&]() {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[stage_idx]->arrive() : void();
                } else {
                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    lane_idx < kNumTMAMulticast ? empty_barriers[stage_idx]->arrive(target_cta) : void();
                }
            };

            // Skip useless computations
            if (scheduler.is_computation_valid(m_block_idx, math_wg_idx * WGMMA::M)) {
                // The compiler must know the dynamic variable `num_former_iters`'s real value
                constexpr bool kShouldOptimize = BLOCK_K / math::constexpr_gcd(BLOCK_K, BLOCK_N) <= 4 and not kMustUseUniformedScaleB;
                constexpr uint32_t kGap = math::constexpr_gcd(BLOCK_K, BLOCK_N) / 8;
                constexpr uint32_t kEnd = kShouldOptimize ? BLOCK_K / 8 : 0;

                // Dispatch `num_former_iters` and launch MMAs
                dispatch_num_former_iters<0, kGap, kEnd>(kShouldOptimize ? num_former_iters : 0, [&](auto _) {
                    #pragma unroll 8
                    for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                        const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                        const auto b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);

                        // Read B scales
                        float scale_b_0 = ptx::ld_shared(smem_sfb + k_block_idx), scale_b_1;
                        // NOTES: even some blocks do not need to read the second row, but we still load one to align with other blocks
                        if constexpr (not kMustUseUniformedScaleB)
                            scale_b_1 = ptx::ld_shared(smem_sfb + k_block_idx + shape_k_scales);

                        // Wait B+sfa (TMA WG) and A (cp.async WG) ready
                        full_barriers_b[stage_idx]->wait(phase);
                        DG_DBG_BARRIER("Math:full_b.pass ", stage_idx, full_barriers_b[stage_idx], phase, 0);
                        full_barriers_a[stage_idx]->wait(phase);
                        DG_DBG_BARRIER("Math:full_a.pass ", stage_idx, full_barriers_a[stage_idx], phase, 0);

                        // TODO: remove some useless computation for unaligned Ms
                        #pragma unroll
                        for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                            auto m_offset = local_idx * WAVE_BLOCK_M;

                            // Read A scales
                            // NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled block polluting the results
                            auto scale_a_0 = do_wgmma_store ? ptx::ld_shared(smem_sfa[stage_idx] + r_0 + m_offset) : 0;
                            auto scale_a_1 = do_wgmma_store ? ptx::ld_shared(smem_sfa[stage_idx] + r_1 + m_offset) : 0;

                            // Commit WGMMA instructions
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                a_desc.reg32_[0] = a_desc_base_lo + (m_offset * BLOCK_K + k * WGMMA::K) / 16;
                                b_desc.reg32_[0] = b_desc_base_lo + k * WGMMA::K / 16;
                                WGMMA::wgmma(a_desc, b_desc, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            // Notify barrier arrival at the last warpgroup wave
                            if (local_idx == BLOCK_M / WAVE_BLOCK_M - 1)
                                empty_barrier_arrive();

                            // Skip promotion for the unfilled parts
                            if (not do_wgmma_store)
                                continue;

                            // Promote with scales
                            // NOTES: making it as predicates is very important for performance, comparing to two loops
                            float scale_0_0 = scale_a_0 * scale_b_0, scale_1_0 = scale_a_1 * scale_b_0;
                            float scale_0_1, scale_1_1;
                            if constexpr (not kMustUseUniformedScaleB)
                                scale_0_1 = scale_a_0 * scale_b_1, scale_1_1 = scale_a_1 * scale_b_1;

                            auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                // NOTES: for unrolled `num_former_iters` cases, we expect the compiler to automatically make it a constant
                                const bool predicate = kMustUseUniformedScaleB or i < num_former_iters;
                                shifted_accum[i * 4 + 0] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 0];
                                shifted_accum[i * 4 + 1] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 1];
                                shifted_accum[i * 4 + 2] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 2];
                                shifted_accum[i * 4 + 3] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 3];
                            }
                        }
                    }
                });
            } else {
                #pragma unroll
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    full_barriers_b[stage_idx]->wait(phase);
                    DG_DBG_BARRIER("Math(skip):full_b", stage_idx, full_barriers_b[stage_idx], phase, 0);
                    full_barriers_a[stage_idx]->wait(phase);
                    DG_DBG_BARRIER("Math(skip):full_a", stage_idx, full_barriers_a[stage_idx], phase, 0);
                    empty_barrier_arrive();
                }
            }

            // TMA checks
            constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16);
            constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode == 0 ? BLOCK_N : (kSwizzleDMode / kNumElemBytes);
            constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;
            DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
            DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N / TMA_D_BLOCK_N <= 32,
                            "Unaligned TMA store or too many TMA store instructions");
            DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");

            // Skip WGMMA store for the unfilled parts
            if (not do_wgmma_store)
                continue;

            // Wait last TMA store to be finished
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

            // Write back to shared memory using STSM and issue TMA stores
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            #pragma unroll
            for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                auto m_offset = local_idx * WAVE_BLOCK_M;
                auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                #pragma unroll
                for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    // Swizzle or padding into the correct address
                    uint8_t* smem_ptr = nullptr;
                    if constexpr (kSwizzleDMode > 0) {
                        // Calculate the swizzling atom offset and in-atom offset
                        constexpr uint32_t kNumBankGroupBytes = 16;
                        auto atom_offset = i / (TMA_D_BLOCK_N / 8), in_atom_offset = i % (TMA_D_BLOCK_N / 8);

                        // Calculate the index of the bank group to be written in the atom
                        auto bank_group_index = in_atom_offset + lane_idx * (kSwizzleDMode / kNumBankGroupBytes);

                        // Reshape the atom in another view and swizzle
                        //  - original: `(BLOCK_M, kSwizzleDMode / kNumBankGroupBytes)`
                        //  - new: `(BLOCK_M * kSwizzleDMode / kNumBankGroupBytes / 8, 8)`
                        constexpr bool kHasShortcut = (kSwizzleDMode / kNumBankGroupBytes) == 8;
                        auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
                        auto col = kHasShortcut ? (in_atom_offset) : (bank_group_index % 8);
                        col ^= row % (kSwizzleDMode / 16);

                        // Add back into the base pointer
                        // NOTES: think twice before modifying this, as changes may affect the number of instructions
                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +                // Base pointer
                            warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +            // Warp offset
                            m_offset * kSwizzleDMode +                                 // Wave offset
                            atom_offset * BLOCK_M * kSwizzleDMode +                    // Swizzle atom offset (constants)
                            row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes; // In-atom offset
                    } else {
                        // No swizzling, just padding
                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx) * BLOCK_N + i * 8);
                    }

                    // NOTES: only 16 lanes' addresses are used
                    ptx::SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                        __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}),
                        __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
                        smem_ptr
                    );
                }
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

            // Use TMA store to write back to global memory
            // TODO: compatible with FP32 output
            constexpr bool kWithGroupOffsetD = kGemmType == GemmType::MGroupedMasked;
            DG_STATIC_ASSERT(kNumWGMMAStoreThreads >= BLOCK_N / TMA_D_BLOCK_N, "Too many TMA blocks");
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M;
                auto n_idx = epilogue_type_t::apply_index_n<TMA_D_BLOCK_N>(n_block_idx * BLOCK_N + in_block_n_offset);
                auto m_idx = scheduler.get_global_idx<kWithGroupOffsetD>(shape_m, BLOCK_M, m_block_idx);
                if constexpr (kGemmType == GemmType::Batched) {
                    cute::SM90_TMA_STORE_3D::copy(&tensor_map_d, smem_ptr,
                                                  n_idx, m_idx, scheduler.current_group_idx);
                } else {
                    cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_ptr, n_idx, m_idx);
                }
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
