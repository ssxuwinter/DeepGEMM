# SM90 FP8 1D2D GEMM 的 gather_index 支持

本文档说明在 `sm90_fp8_gemm_1d2d_impl`（以及对应的 host JIT/Python API）上引入
`gather_index` 入参的设计意图、关键实现细节，以及对原有调用路径的兼容性。

## 1. 需求与语义

调用方希望在 GEMM `D = A @ B` 中，**为每一行输出 token 单独指定它要从 `A` 的哪一行
gather**：

```text
logical_m  = 输出行索引 ∈ [0, M)
source_m   = gather_index[logical_m]   // gather_index: int32 张量，长度 ≥ M
D[logical_m, :] = A[source_m, :] @ B
```

典型场景是 MoE / token shuffle 等需要在矩阵乘前做行级路由的场合。我们希望避免显式的
`A_perm = A[gather_index]` 物化，直接在 kernel 里按 `gather_index` 间接寻址 A 行。

约束：

- 只针对 SM90 FP8 1D2D 路径（`fp8_fp4_gemm_nt` 中 `gran_n != 1` 的分支）。
- 仅 `kGemmType == GemmType::Normal` 的入口（`sm90_fp8_gemm_1d2d`）暴露此参数；
  contiguous/masked/bmm 三条 grouped 路径不支持。
- `gather_index` 必须是 `torch.int32`，CUDA contiguous，元素数 ≥ M。
- `gather_index = None` 时，行为必须与改造前**逐字节一致**。

## 2. 为什么改 cp.async 而不改 TMA descriptor

读旧版的 `sm90_fp8_gemm_1d2d_impl` 可以发现：

- `tensor_map_a` / `tensor_map_sfa` 在 producer 主循环里**没有被使用**；A 和 sfa 都是
  通过 `cp.async` 直接按 `gmem_a + (m_global_base + row) * stride_a + ...` 这种
  显式地址加载的（B 仍然走 TMA）。
- TMA descriptor 仅作为参数保留，并在 kernel 入口做 `prefetch_tma_descriptor` 以兼容
  原版本。

因此引入 `gather_index` 时，**只需要改写 cp.async 路径里 A 行号和 sfa 行号的计算**，
完全不用动 TMA descriptor、smem 布局、swizzle、barrier 计数等。这是这次改动可以
做得这么局部的根本原因。

## 3. kernel 端实现

### 3.1 函数签名扩展

`deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh`：

```c++
sm90_fp8_gemm_1d2d_impl(float* sfb, int* grouped_layout,
                        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                        const __nv_fp8_e4m3* __restrict__ gmem_a,
                        uint32_t stride_a,
                        const float* __restrict__ gmem_sfa,
                        uint32_t stride_sfa,
                        const int* __restrict__ gather_index,   // ← 新增
                        const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                        ...)
```

约定：`gather_index == nullptr` 时表示禁用，行为与改前等价。

### 3.2 source row 计算的循环外提（关键优化）

最朴素的做法是“每次 cp.async 一个 chunk 时都读一次 `gather_index`”。但要注意
producer 在每个 M tile 内会沿 K 推进 `num_total_k_blocks` 次：

```text
for k_block_idx in [0, num_total_k_blocks):
    for i in [0, kAItersPerThread):
        cp.async A[..., source_m, ...]   // 每次都读 gather_index?
```

`source_m` 只取决于线程负责的 `logical_m = m_global_base + row`，**与 `k_block_idx`
完全无关**。因此把 `gather_index` 的查表搬到 K 主循环外，每个 M tile 只读一次：

```c++
while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
    const uint32_t m_global_base = scheduler.get_global_idx<...>(...);

    // 每个 producer thread 负责的 A row chunks 的源行号
    uint32_t source_m_for_a[kAItersPerThread];
    #pragma unroll
    for (uint32_t i = 0; i < kAItersPerThread; ++i) {
        const uint32_t row = ((i * kCpAsyncThreads + tid_in_wg) * kCpAsyncWidth) / BLOCK_K;
        const uint32_t logical_m = m_global_base + row;
        source_m_for_a[i] = (has_gather_index and logical_m < shape_m)
            ? __ldg(gather_index + logical_m) : logical_m;
    }

    // 每个 producer thread 负责的 sfa rows 的源行号
    uint32_t source_m_for_sfa[kSFAItersPerThread];
    #pragma unroll
    for (uint32_t i = 0; i < kSFAItersPerThread; ++i) {
        const uint32_t row = i * kCpAsyncThreads + tid_in_wg;
        const uint32_t logical_m = m_global_base + row;
        source_m_for_sfa[i] = (has_gather_index and row < BLOCK_M and logical_m < shape_m)
            ? __ldg(gather_index + logical_m) : logical_m;
    }

    for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
        ...
        // A load: 直接用缓存好的 source_m_for_a[i]
        cp_async4(dst_a + ..., gmem_a + source_m_for_a[i] * stride_a + k_idx + col);
        // sfa load: 直接用缓存好的 source_m_for_sfa[i]
        ...
    }
}
```

效果：

- 把 `gather_index` 的 ldg 次数从 `num_total_k_blocks * (kAItersPerThread + 1)` 降到
  每 M tile 一次的 `kAItersPerThread + kSFAItersPerThread`。对于 K 较大的形状
  （如 K=7168，`num_total_k_blocks = 56`），是 50× 以上的削减。
- 索引值常驻寄存器，K 主循环里零额外指令。
- 对 `gather_index = nullptr` 路径，编译期 `has_gather_index = false`（运行期常量
  其实，但条件简单，编译器很容易选择无分支的 `logical_m`）；与原行为完全等价。

### 3.3 A 与 sfa 必须使用同一个 gather mapping

FP8 channel-wise scaling 下，`A[m, k]` 的真实数值是 `A_fp8[m, k] * sfa[k_block, m]`。
gather A 行号时，**必须同时 gather 它对应的 scale 行号**，否则数值就被错位的 scale
还原，结果完全错。因此 A 和 sfa 共享同一个 `gather_index` 查表。

### 3.4 sfa load 路径泛化（BLOCK_M > 128 兼容）

旧代码假定 `BLOCK_M ≤ 128`：

```c++
if (tid_in_wg < BLOCK_M) {
    if (m_global_base + tid_in_wg < shape_m) {
        cp.async(... + (m_global_base + tid_in_wg));
    }
}
```

但 SM90 heuristics（`csrc/jit_kernels/heuristics/sm90.hpp`）的 normal/batched/
k-grouped + BF16 输出场景里，`block_m` 候选包含 256：

```c++
if (desc.cd_dtype != torch::kFloat)
    block_m_candidates.push_back(256);
```

这种情况下 `tid_in_wg < BLOCK_M` 始终为真，但每个线程**只发了一次** sfa cp.async，
覆盖到 `[0, 128)` 行；后 128 行的 sfa 没人加载。这是一个旧代码就潜在存在的 bug
（在没引入 gather_index 之前就有），只是恰好被 producer 进入 K loop 后 stage idx 重叠
等行为掩盖。

我们顺手补齐：把 sfa 也改成和 A 一样、按线程多次迭代覆盖所有行。

```c++
static constexpr uint32_t kSFAItersPerThread = math::constexpr_ceil_div(BLOCK_M, kCpAsyncThreads);

#pragma unroll
for (uint32_t i = 0; i < kSFAItersPerThread; ++i) {
    const uint32_t row = i * kCpAsyncThreads + tid_in_wg;
    if (row < BLOCK_M and m_global_base + row < shape_m) {
        ...
        const float* src_sfa = gmem_sfa + sfa_k_idx * stride_sfa + source_m_for_sfa[i];
        cp.async.ca.shared.global ...
    }
}
```

- `BLOCK_M ≤ 128`：`kSFAItersPerThread = 1`，外层只循环一次，等价于旧行为（保留
  `row < BLOCK_M` 的 mask）。
- `BLOCK_M = 256`：`kSFAItersPerThread = 2`，正确覆盖全部 256 行 sfa。
- 越界行（`logical_m >= shape_m`）跳过 cp.async；HW 的 `cp.async.mbarrier.arrive.noinc`
  对“没有 outstanding cp.async”仍然能正确退到 0，barrier 行为不变。

### 3.5 gather_index 越界保护

`gather_index` 数组长度为 logical M。最后一个 M tile 可能含有 `logical_m >= shape_m`
的 padding 行。在原版代码里这些行的 cp.async 已经被 `< shape_m` mask 掉了，但读
`gather_index[logical_m]` 仍然可能越过 host 提供的 buffer。所以预计算阶段加上
`logical_m < shape_m` 这一道保护：

```c++
source_m_for_a[i] = (has_gather_index and logical_m < shape_m)
    ? __ldg(gather_index + logical_m) : logical_m;
```

`logical_m` 不命中合法范围时 fall back 到自身，反正这一行的实际 cp.async 也被屏蔽。

## 4. JIT host 侧改动

`csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp`：

1. `SM90FP8Gemm1D2DRuntime::Args` 增加：
   ```c++
   void *gather_index;   // optional int32 row remap; nullptr keeps logical rows
   ```
2. `launch_impl` 把 `args.gather_index` 透传给 kernel，与 kernel 签名顺序对齐：
   ```c++
   launch_kernel(kernel, config,
       args.sfb, args.grouped_layout,
       args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k,
       args.gmem_a, args.stride_a,
       args.gmem_sfa, args.stride_sfa,
       args.gather_index,                  // ← 新增
       args.tensor_map_a, args.tensor_map_b,
       args.tensor_map_d, args.tensor_map_sfa);
   ```
3. `sm90_fp8_gemm_1d2d(...)` 入口增加可选参数：
   ```c++
   const std::optional<torch::Tensor>& gather_index = std::nullopt
   ```
   校验：CUDA、contiguous、`torch::kInt`、`numel() >= m`、`a.size(0) >= m`
   （即允许 A 池子大于 GEMM 输出 m）。
4. **`stride_sfa_elems` 改为 `sfa.stride(-1)`**（不再硬编码为 `m`）。
   - `transform_sf_into_required_layout` 之后的 sfa 是 MN-major，
     `stride(-1)` 等于 `tma_aligned_size(mn, 4) = align(mn, 4)`。
   - 旧实现 `stride_sfa = m` 在 `m % 4 != 0` 时会读到 SFA 行间的 padding
     字节（这是一个**预先存在的 bug**，例如 `m=1` 时输出几乎全噪声）。
   - 新实现 `stride_sfa = sfa.stride(-1)` 同时解决：
     - `m % 4 != 0` 的对齐问题；
     - gather 场景下 A 池子 m_pool > m 时，sfa 必须按 m_pool 对齐而不是
       GEMM 的 m。
5. 其余三条 launch 路径（`sm90_m_grouped_fp8_gemm_contiguous_1d2d`、
   `sm90_m_grouped_fp8_gemm_masked_1d2d`、`sm90_fp8_bmm`）的 Args 显式
   `.gather_index = nullptr`，保持原有 grouped 调用语义。这次改动**不改**
   它们的 `stride_sfa`（多维 SFA 的 layout 假设不同，留给后续 PR）。

JIT 缓存 key 不会受这次改动影响：`generate_impl` 拼出来的模板实参列表（block_m、
swizzle、num_stages 等）没有改变；新增的运行期指针不参与 JIT key。

## 5. Python / pybind 侧

`csrc/apis/gemm.hpp`：

- `fp8_fp4_gemm_nt(...)` 入口加 `const std::optional<torch::Tensor>& gather_index`。
- **形状校验改成 `m_pool` 与 `m_d` 分离**：
  - `m_a = a.first.shape[0]`（A 池子）。
  - `m_d = d.shape[0]`（GEMM 输出 M）。
  - 不带 gather 时要求 `m_a == m_d`（与原有契约一致）。
  - 带 gather 时要求 `m_a >= m_d`，并断言 `gather_index.numel() >= m_d`。
- **SFA layout transform 用 `m_pool = m_a`**（而不是 `m_d`），保证 sfa 张量
  的形状/对齐与 A 池子匹配，下游 `stride_sfa = sfa.stride(-1)` 自洽。
- 在 dispatch 时：
  - SM90 + `gran_n == 1`（即 1d1d 路径）：断言 `not gather_index.has_value()`。
  - SM90 + 1d2d：透传给 `sm90_fp8_gemm_1d2d(...)`。
  - SM100 路径：断言 `not gather_index.has_value()`。
- pybind 绑定增加 `py::arg("gather_index") = std::nullopt`。
- 由于 `m.attr("fp8_gemm_nt") = m.attr("fp8_fp4_gemm_nt")`，Python 端
  `deep_gemm.fp8_gemm_nt(..., gather_index=...)` 同步可用。

未支持（保留为 future work）：

- `fp8_gemm_nn / fp8_gemm_tn / fp8_gemm_tt`：会先 `transpose`，gather 语义需要重新
  定义，未启用。
- `m_grouped_fp8_gemm_*`、`m_grouped_bf16_gemm_*`、k-grouped、attention 入口：
  目前不需要 gather，未暴露。
- SM100、bf16 GEMM、cublasLt：未涉及。

## 6. 兼容性矩阵

| 调用入口                                  | 是否暴露 gather_index | 默认行为 |
| ----------------------------------------- | --------------------- | -------- |
| `deep_gemm.fp8_gemm_nt`                   | 是（仅 SM90 1d2d）   | None：与改前一致 |
| `deep_gemm.fp8_gemm_{nn,tn,tt}`           | 否                    | 与改前一致 |
| `deep_gemm.m_grouped_fp8_gemm_*`          | 否                    | 与改前一致 |
| `deep_gemm.m_grouped_bf16_gemm_*`         | 否                    | 与改前一致 |
| `deep_gemm.fp8_gemm_nt_skip_head_mid`     | 否（attention 入口）  | 与改前一致 |
| 任何 SM100 / cublasLt / bmm 路径          | 否                    | 与改前一致 |

`gather_index = None` 路径下：

- kernel 编译产物相同（template instantiation 没变）。
- 运行期 `has_gather_index` 是 const-true/false，且数据路径和原版一致：A 行号
  退化为 `m_global_base + row`，sfa 退化为 `m_global_base + row`。
- 唯一与原代码字面不同的，是 sfa load 现在用了 `kSFAItersPerThread` 循环。当
  `BLOCK_M ≤ 128` 时该循环只跑一次，且条件与原 `if (tid_in_wg < BLOCK_M)` 等价；当
  `BLOCK_M = 256` 时是修正了原本就缺的 sfa 后半段加载。

## 7. 文件改动清单

| 文件 | 用途 |
| --- | --- |
| `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh` | kernel 签名 + cp.async 路径的 source row 缓存与 gather 寻址 |
| `csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp` | JIT Args/launch 增加 gather_index；4 条 host 入口同步 |
| `csrc/apis/gemm.hpp` | `fp8_fp4_gemm_nt` 入口增加 gather_index；dispatch 路径限定；pybind 绑定 |

## 8. 使用示例

```python
import torch
import deep_gemm

m, n, k = 4096, 4096, 7168
a_fp8, sfa = ...    # FP8 e4m3 + per-128-channel scale，按现有 quant 流程构造
b_fp8, sfb = ...
d = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")

# 输出第 i 行希望从 a 的 perm[i] 行 gather
perm = torch.randperm(a_fp8.size(0), device="cuda", dtype=torch.int32)
gather_index = perm[:m].contiguous()

deep_gemm.fp8_gemm_nt(
    (a_fp8, sfa),
    (b_fp8, sfb),
    d,
    recipe=(1, 128, 128),
    gather_index=gather_index,    # ← 新增
)
# 等价于：deep_gemm.fp8_gemm_nt((a_fp8[gather_index], sfa[:, gather_index]),
#                              (b_fp8, sfb), d, recipe=...)
```

## 9. 测试与回归建议

- **正确性**：构造一个普通 FP8 NT GEMM，先用 `gather_index = arange(m, dtype=int32)`
  跑一次，确认结果与不传 `gather_index` 完全一致；再用一个 `randperm` 跑一次，
  与显式 `a_perm = a[gather_index]` 路径对比。
- **形状覆盖**：`BLOCK_M ∈ {64, 128, 256}`、`shape_m % BLOCK_M != 0`（验证 padding
  行 `gather_index` 越界保护）、不同 K 长度、是否触发 cluster multicast。
- **回归**：`tests/test_fp8_fp4.py`、`tests/test_fp8.py` 等不传 `gather_index` 的现有
  case 都应该不变。

## 10. 已知限制与后续工作

- 仅 SM90 1d2d；SM100、1d1d、bf16 路径若有需要再各自接入。
- `gather_index` 当前是 int32；如果未来出现 M 超 2^31，需要扩展到 int64。
- m-grouped 三个变种（contiguous/masked/contiguous-with-psum）目前共享 kernel
  入口但传 `nullptr`；如要支持 grouped 内部 gather，要叠加 `current_group_idx`
  这一层 offset，留作下一步。
