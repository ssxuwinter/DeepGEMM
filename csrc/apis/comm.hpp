#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>

#include "../jit/handle.hpp"
#include "../utils/exception.hpp"

namespace deep_gemm::comm {

static void check_payloads(const std::vector<torch::Tensor>& inputs,
                           const std::vector<torch::Tensor>& outputs,
                           const int& local_rank,
                           const int& num_ranks) {
    DG_HOST_ASSERT(not inputs.empty());
    DG_HOST_ASSERT(inputs.size() == outputs.size());
    DG_HOST_ASSERT(num_ranks > 0);
    DG_HOST_ASSERT(local_rank >= 0 and local_rank < num_ranks);
    for (size_t i = 0; i < inputs.size(); ++i) {
        DG_HOST_ASSERT(inputs[i].is_cuda() and outputs[i].is_cuda());
        DG_HOST_ASSERT(inputs[i].is_contiguous() and outputs[i].is_contiguous());
        DG_HOST_ASSERT(inputs[i].device() == outputs[i].device());
        DG_HOST_ASSERT(outputs[i].nbytes() % static_cast<size_t>(num_ranks) == 0);
        DG_HOST_ASSERT(outputs[i].nbytes() / static_cast<size_t>(num_ranks) == inputs[i].nbytes());
    }
}

static void check_rank_flags(const torch::Tensor& rank_flags, const int& num_ranks) {
    DG_HOST_ASSERT(rank_flags.is_cuda() and rank_flags.is_contiguous());
    DG_HOST_ASSERT(rank_flags.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(rank_flags.numel() >= num_ranks);
}

static void stream_write_value64(const torch::Tensor& dst, const int64_t& index, const int64_t& value) {
    DG_HOST_ASSERT(dst.is_cuda() and dst.is_contiguous());
    DG_HOST_ASSERT(dst.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(index >= 0 and index < dst.numel());

    const c10::cuda::CUDAGuard guard(dst.device());
    const auto stream = at::cuda::getCurrentCUDAStream(dst.device().index());
    const auto ptr = dst.data_ptr<int64_t>() + index;
    const auto cu_stream = reinterpret_cast<CUstream>(static_cast<cudaStream_t>(stream));
    const auto cu_ptr = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr));
    // Load the v2 symbol explicitly. The unsuffixed legacy export may exist but
    // return CUDA_ERROR_NOT_SUPPORTED on systems where the v2 API works.
    DG_CUDA_DRIVER_CHECK(lazy_cuStreamWriteValue64_v2(cu_stream, cu_ptr, static_cast<cuuint64_t>(value), 0));
}

static void write_rank_flag_if_needed(const std::optional<torch::Tensor>& rank_flags,
                                      const int& rank,
                                      const int& num_ranks,
                                      const int& value) {
    if (rank_flags.has_value()) {
        check_rank_flags(rank_flags.value(), num_ranks);
        stream_write_value64(rank_flags.value(), rank, value);
    }
}

static void copy_bytes_async(void* dst, const void* src, const size_t& num_bytes, const cudaStream_t& stream) {
    if (num_bytes == 0 or dst == src)
        return;
    DG_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst, src, num_bytes, cudaMemcpyDeviceToDevice, stream));
}

static void single_node_allgather_copy_local(const std::vector<torch::Tensor>& inputs,
                                             const std::vector<torch::Tensor>& outputs,
                                             const int& local_rank,
                                             const std::optional<torch::Tensor>& rank_flags = std::nullopt,
                                             const int& flag_value = 1) {
    DG_HOST_ASSERT(not inputs.empty() and inputs.size() == outputs.size());
    DG_HOST_ASSERT(inputs[0].nbytes() > 0);
    DG_HOST_ASSERT(outputs[0].nbytes() % inputs[0].nbytes() == 0);
    const int num_ranks = static_cast<int>(outputs[0].nbytes() / inputs[0].nbytes());
    check_payloads(inputs, outputs, local_rank, num_ranks);
    if (rank_flags.has_value())
        check_rank_flags(rank_flags.value(), num_ranks);

    const c10::cuda::CUDAGuard guard(outputs[0].device());
    const auto stream = static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream(outputs[0].device().index()));
    for (size_t i = 0; i < inputs.size(); ++i) {
        const size_t slot_bytes = inputs[i].nbytes();
        auto* dst = static_cast<uint8_t*>(outputs[i].data_ptr()) + static_cast<size_t>(local_rank) * slot_bytes;
        copy_bytes_async(dst, inputs[i].data_ptr(), slot_bytes, stream);
    }
    write_rank_flag_if_needed(rank_flags, local_rank, num_ranks, flag_value);
}

static void single_node_allgather_pull(const std::vector<torch::Tensor>& outputs,
                                       const std::vector<std::vector<int64_t>>& output_buffer_ptrs,
                                       const int& local_rank,
                                       const int& num_ranks,
                                       const std::optional<torch::Tensor>& rank_flags = std::nullopt,
                                       const int& flag_value = 1) {
    DG_HOST_ASSERT(not outputs.empty());
    DG_HOST_ASSERT(output_buffer_ptrs.size() == outputs.size());
    DG_HOST_ASSERT(num_ranks > 0);
    DG_HOST_ASSERT(local_rank >= 0 and local_rank < num_ranks);
    if (rank_flags.has_value())
        check_rank_flags(rank_flags.value(), num_ranks);

    for (size_t i = 0; i < outputs.size(); ++i) {
        DG_HOST_ASSERT(outputs[i].is_cuda() and outputs[i].is_contiguous());
        DG_HOST_ASSERT(outputs[i].nbytes() % static_cast<size_t>(num_ranks) == 0);
        DG_HOST_ASSERT(static_cast<int>(output_buffer_ptrs[i].size()) >= num_ranks);
    }

    const c10::cuda::CUDAGuard guard(outputs[0].device());
    const auto stream = static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream(outputs[0].device().index()));
    for (int step = 1; step < num_ranks; ++step) {
        const int src_rank = (local_rank + step) % num_ranks;
        for (size_t i = 0; i < outputs.size(); ++i) {
            const size_t slot_bytes = outputs[i].nbytes() / static_cast<size_t>(num_ranks);
            auto* dst = static_cast<uint8_t*>(outputs[i].data_ptr()) + static_cast<size_t>(src_rank) * slot_bytes;
            auto* src_base = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(output_buffer_ptrs[i][src_rank]));
            auto* src = src_base + static_cast<size_t>(src_rank) * slot_bytes;
            copy_bytes_async(dst, src, slot_bytes, stream);
        }
        write_rank_flag_if_needed(rank_flags, src_rank, num_ranks, flag_value);
    }
}

static void single_node_allgather(const std::vector<torch::Tensor>& inputs,
                                  const std::vector<torch::Tensor>& outputs,
                                  const std::vector<std::vector<int64_t>>& output_buffer_ptrs,
                                  const int& local_rank,
                                  const int& num_ranks,
                                  const pybind11::object& symm_handle,
                                  const std::optional<torch::Tensor>& rank_flags = std::nullopt,
                                  const int& flag_value = 1) {
    single_node_allgather_copy_local(inputs, outputs, local_rank, rank_flags, flag_value);
    DG_HOST_ASSERT(not symm_handle.is_none());
    symm_handle.attr("barrier")();
    single_node_allgather_pull(outputs, output_buffer_ptrs, local_rank, num_ranks, rank_flags, flag_value);
}

static void register_apis(pybind11::module_& m) {
    m.def("stream_write_value64", &stream_write_value64,
          pybind11::arg("dst"), pybind11::arg("index"), pybind11::arg("value"));
    m.def("single_node_allgather_copy_local", &single_node_allgather_copy_local,
          pybind11::arg("inputs"), pybind11::arg("outputs"), pybind11::arg("local_rank"),
          pybind11::arg("rank_flags") = std::nullopt, pybind11::arg("flag_value") = 1);
    m.def("single_node_allgather_pull", &single_node_allgather_pull,
          pybind11::arg("outputs"), pybind11::arg("output_buffer_ptrs"),
          pybind11::arg("local_rank"), pybind11::arg("num_ranks"),
          pybind11::arg("rank_flags") = std::nullopt, pybind11::arg("flag_value") = 1);
    m.def("single_node_allgather", &single_node_allgather,
          pybind11::arg("inputs"), pybind11::arg("outputs"), pybind11::arg("output_buffer_ptrs"),
          pybind11::arg("local_rank"), pybind11::arg("num_ranks"), pybind11::arg("symm_handle"),
          pybind11::arg("rank_flags") = std::nullopt, pybind11::arg("flag_value") = 1);
}

} // namespace deep_gemm::comm
