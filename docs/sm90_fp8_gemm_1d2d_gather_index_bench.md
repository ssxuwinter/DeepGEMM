# Gather-A FP8 GEMM Bench 指南

本文档说明如何使用 `tests/test_gather_index.py` 对 SM90 FP8 1D2D GEMM 的
`gather_index` 特性进行**精度验证 + 性能基准测试**。

> 实现细节请参考 [`sm90_fp8_gemm_1d2d_gather_index.md`](./sm90_fp8_gemm_1d2d_gather_index.md)。

## 1. 测试目的

我们要回答两个问题：

1. **正确性**：kernel 端 gather 的输出是否等价于显式 `A_perm = A[gather_index]`
   后再做 GEMM？是否仍在 FP8 量化误差容忍范围内？
2. **性能收益**：避免显式物化 `A_perm` 后，GEMM 本身有多少额外开销？端到端
   是否相对“显式 perm + GEMM”有提速？

## 2. 测试方案概览

我们对每组 `(m, m_total, n, k)` 形状构造三条等价路径：

| 路径 | 描述 |
| --- | --- |
| (1) `baseline`        | 标准 `fp8_gemm_nt(A, B, D)`，A 形状 `(m, k)`。仅作参照 |
| (2) `kernel-gather`   | `fp8_gemm_nt(A_full, B, D, gather_index=g)`，A_full 形状 `(m_total, k)`，kernel 在内核里按 `g[i]` 间接寻址 |
| (3) `explicit-perm`   | host 上先 `A_perm = A_full[g]; sfa_perm = sfa_full[g]`，再走 baseline GEMM |

判定准则：

- **Path (2) ≈ Path (3)**：两条路径数学上完全等价，结果应当 bit-exact
  （考虑到 FP8 浮点结合性的次序变化，我们用 `1e-6` 的 cosine-style diff 阈值）。
- **Path (2) ≈ bf16 ref**：与不带 gather 的 FP8 GEMM 同等精度（用
  `QuantConfig.max_diff()` 的标准阈值，legacy 配置下是 `1e-3`）。

性能层面我们关心：

- Path (2) vs Path (1) 的 **kernel-only 开销**：理想情况下应当几乎相等
  （只多了 producer warp-group 进入 K loop 前的几次 `__ldg(gather_index)`）。
- Path (2) vs Path (3) 的 **端到端时间**：Path (3) 多一次 `index_select`
  kernel 与额外的 GMEM 读写，理论上 Path (2) 总是更快。
- 内存带宽：用 `bytes_min = sizeof(A_perm) + sizeof(B) + sizeof(D)` 作下限，
  推算 kernel-gather 的有效 GMEM 带宽。

## 3. 运行方式

> **注意**：本特性涉及 C++ 扩展（pybind 入口签名变化）和 CUDA kernel
> 签名变化。修改后必须重新构建 deep_gemm 扩展，否则 Python 端调用会沿用
> 旧符号，得到错误的输出。
>
> ```bash
> # 推荐：editable install，源码改动后只需 build（耗时短）
> pip install -e . --no-build-isolation
> # 或：完整 reinstall
> pip install . --force-reinstall --no-deps
> ```

```bash
cd MoE_DeepGEMM
python tests/test_gather_index.py
```

可选参数：

```bash
python tests/test_gather_index.py --no-accuracy   # 仅跑性能
python tests/test_gather_index.py --no-perf       # 仅跑精度
```

脚本启动时会清空 `~/.deep_gemm/cache`，强制 JIT 重新编译，避免老缓存影响测量。

### 环境要求

- SM90 GPU（H100/H800 等）。`gather_index` 当前仅在 SM90 FP8 1D2D 路径暴露。
- PyTorch + DeepGEMM 已编译为对应 arch（`-arch=sm_90a`）。
- 需要 `tests/generators.py` 在 `sys.path` 上（脚本里已自动注入）。

### 形状约束

- 测试只在 `m % 16 == 0` 的形状上验证。`m % 4 != 0` 触发 SFA layout 不对齐
  的另一类问题（与 gather 无关，留给独立 PR 处理）。
- 当传入 `gather_index` 时，要求 `a.shape[0] >= m`：A 是“可池化”的源行，
  GEMM 的输出 m 取自 D 的形状，`gather_index[i]` 选择第 i 个输出对应的源行。

## 4. 输出解读

> **常见误读**：不要拿本 bench 的数字直接对比 `tests/test_fp8_fp4.py:test_m_grouped_gemm_contiguous` 的 `1200~1300 TFLOPS`。
> 那是 **m-grouped contiguous 1D2D**，total `m≈30000~36000`，单次 kernel 计算量是这里 6~10×。
> 本 bench 用的是 **`GemmType::Normal`**（gather_index 当前唯一支持的路径），
> 单次 `m≤8192`，单次计算量约 100~500 GFLOPs。这种规模即使是 H100 也只能跑到
> 800~1100 TFLOPS（pipeline fill/drain 占比、L2 复用率、SM 满载情况都不一样）。
> 如果想做同源对比，把 `tests/test_fp8_fp4.py` 末尾的 `# test_gemm()` 取消注释跑一遍，
> 它的 `m=4096, n=4096, k=7168` 那一行 TFLOPS 应该和本 bench 的 `baseline` 列接近。

### 4.1 精度部分

```text
Testing gather-A FP8 GEMM accuracy:
 > m=    1, m_total=    1, n=2112, k=7168, mode=identity  | kernel-vs-perm diff=0.00e+00 (OK) | kernel-vs-bf16 diff=0.00057 (OK)
 > m=  128, m_total=  256, n=4096, k=7168, mode=randperm  | kernel-vs-perm diff=2.31e-08 (OK) | kernel-vs-bf16 diff=0.00041 (OK)
 > m=  128, m_total=  256, n=4096, k=7168, mode=random    | kernel-vs-perm diff=1.95e-08 (OK) | kernel-vs-bf16 diff=0.00043 (OK)
 ...
```

- `mode=identity`：`gather_index = arange(m, dtype=int32)`，结果应与不传
  `gather_index` 的 baseline 完全一致；`kernel-vs-perm diff` 通常 == 0。
- `mode=randperm`：每行从 `A_full[0..m_total)` 选一个**不重复**的源行；MoE 路由
  典型场景。
- `mode=random`：允许重复，覆盖 `gather_index[i] == gather_index[j]` 的边界。

如果有任何一项打印 `FAIL`，脚本会直接 `assert` 退出，把失败 case 信息写入异常。

形状覆盖的设计意图：

| `(m, m_total, n, k)`            | 覆盖目标 |
| --- | --- |
| `(1, 1, 2112, 7168)`            | 极小 m，验证 BLOCK_M padding 行 gather 越界保护 |
| `(128, 256, 4096, 7168)`        | block-aligned，最常用形状 |
| `(4096, 8192, 4096, 7168)`      | 大 m + 大 m_total，MoE-like |
| `(4096, 4096, 7168, 2048)`      | identity 路径回归（确保 None==identity 行为一致） |
| `(1023, 2048, 2048, 1024)`      | 非对齐 m / k，验证尾巴 tile |

### 4.2 性能部分

输出沿用 `test_fp8_fp4.py:test_gemm` 的格式，每个形状打印 4 行：

```text
 > Perf (m= 4096, n= 4096, k= 7168, 1D2D, layout=NT, BF16, baseline   ):   485.3 us |   988 TFLOPS |   xxx GB/s | diff=0.00045
 > Perf (m= 4096, n= 4096, k= 7168, 1D2D, layout=NT, BF16, gather=id  ):   487.0 us |   985 TFLOPS |   xxx GB/s | diff=0.00045
 > Perf (m= 4096, n= 4096, k= 7168, 1D2D, layout=NT, BF16, gather=perm):   492.7 us |   974 TFLOPS |   xxx GB/s | diff=0.00043
 > Perf (m= 4096, n= 4096, k= 7168, 1D2D, layout=NT, BF16, gather=pool x2):  493.2 us |   973 TFLOPS |   xxx GB/s | diff=0.00046
```

字段说明：

- `baseline`：`fp8_gemm_nt(A, B, D)`，A 是 `(m, k)`，无 gather。这是 GEMM 的
  硬下限，等价于 `test_fp8_fp4.py:test_gemm` 同形状的输出。
- `gather=id`：`gather_index = arange(m, dtype=int32)`。kernel 走 gather 路径
  但每行选回自己；正确实现下应当与 baseline 几乎相等（额外开销 ≤ 1%）。
- `gather=perm`：`gather_index = randperm(m)`，`A_pool == A`。验证 kernel 在
  打乱顺序读 A 行 / sfa 行时能维持带宽。
- `gather=pool x2`：`A_pool` 形状 `(2m, k)`，`gather_index` 从 2m 行里选 m 行。
  **真实的 MoE token shuffle 场景**。这一行的带宽数值会比 baseline 高，因为
  分母里我们用 `count_bytes(A_pool, B) + bytes(D)` 而不是被实际命中的行数。

每个形状下，期望看到的关系：`baseline ≈ gather=id ≈ gather=perm ≈ gather=pool`，
两两差距应该都在 ±3% 以内。如果出现 >5% 的恶化，参考第 5.3 节排查。

形状覆盖（`m_list × bf16_output_nk`）：

| 维度 | 值 | 说明 |
| --- | --- | --- |
| `m`        | `[128, 1024, 4096, 8192]` | 都是 16 对齐的，避开 m % 4 != 0 的预先存在 bug |
| `(n, k)`   | `(2112, 7168) / (4096, 7168) / (7168, 2048) / (24576, 1536) / (7168, 16384)` | 与 `test_fp8_fp4.py:test_gemm` 完全相同 |

## 5. 预期结果与验收标准

### 5.1 正确性

- 所有 case 都打印 `OK`。
- `kernel-vs-perm diff` 在 1e-6 量级或更小（几乎全为 0，来自 GEMM 本身次序差异）。
- `kernel-vs-bf16 diff` < `quant_config.max_diff()`（legacy 默认 1e-3）。

### 5.2 性能

下列指标为典型 H100/H800 单卡参考量级，实际数值与时钟、热稳定状态相关，
**不要硬性 assert**。

- **gather=id / gather=perm / gather=pool 相对 baseline 的开销**：`≤ 5%`，
  多数形状 `≤ 2%`。（主要开销是 producer WG 在 K loop 外的
  `kAItersPerThread + kSFAItersPerThread` 次 `__ldg(gather_index)`。）
- **绝对 TFLOPS 不要拿来和 m-grouped contiguous 比**：见第 4 节开头的提醒。
  在 `m=4096, n=4096, k=7168` 这种规模下，Normal GEMM 跑到 ~1000 TFLOPS 已经
  接近 H100 上的硬上限；m-grouped contiguous 用 `m_total≈36000` 同时调度 N
  方向 cluster_m=2 multicast，等价 wave 数多很多倍，所以能贴近 1300 TFLOPS。
  这个差距来自 **workload size 与 kernel 的差异**，不是 gather 实现的开销。

### 5.3 异常排查

| 现象 | 可能原因 |
| --- | --- |
| `gather=id` 与 `baseline` 输出 diff > 1e-6 | identity 路径回归被改坏；检查 `has_gather_index ? __ldg(...) : logical_m` 这条 branch |
| `gather=perm` 或 `gather=pool` 与 explicit-perm GEMM 输出 diff > 1e-6 | A 和 sfa 没有按同一个 `gather_index` 同步；检查 sfa load 中 `source_m_for_sfa` |
| 任意路径与 bf16 ref 的 diff 接近 1.0 | SFA stride 没用 `sfa.stride(-1)`；旧代码遇到 `m % 4 != 0` 形状会读到行间 padding 字节 |
| `gather=*` kernel 比 baseline 慢 >10% | producer WG 寄存器压力上升导致溢出；查看 SASS 是否有 `LDL/STL`，必要时调整 `kNumProducerRegisters` |
| 绝对 TFLOPS 比预期低 | 先确认 baseline 自己是不是同样低（即 workload 太小所致）。可以取消 `tests/test_fp8_fp4.py` 末尾的 `# test_gemm()` 跑一遍同形状对比 |

## 6. 与 CI 集成

最简形式：

```bash
python tests/test_gather_index.py --no-perf   # 仅跑精度，作为 PR gate
```

跑完会以非零 exit code 退出（assert 失败）。如果需要把性能一起纳入回归，
可以采集脚本 stdout 并按 fixed-shape 抽取 `kernel gather BW = ... GB/s`
字段做趋势监控。

## 7. 文件清单

- 测试脚本：`tests/test_gather_index.py`
- 实现文档：`docs/sm90_fp8_gemm_1d2d_gather_index.md`
- 本文档：  `docs/sm90_fp8_gemm_1d2d_gather_index_bench.md`
