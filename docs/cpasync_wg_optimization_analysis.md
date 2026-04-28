# cp.async WG 优化分析

针对 `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh` 中 cp.async warp-group 的资源占用、流水效率进行分析，并给出优化建议。

## 1. 当前架构概览

Kernel 将线程分为三个 warp-group，各自运行独立循环，通过 barrier 协调 producer-consumer 关系：


| 线程组                          | 线程范围                                         | 职责                            | 信号                  |
| ---------------------------- | -------------------------------------------- | ----------------------------- | ------------------- |
| TMA WG (128 threads)         | `[kNumMathThreads, kNumMathThreads+127]`     | 通过 TMA 加载 B 矩阵                | → `full_barriers_b` |
| cp.async WG (128 threads)    | `[kNumMathThreads+128, kNumMathThreads+255]` | 通过 cp.async 加载 A 矩阵 + scale_a | → `full_barriers_a` |
| Math WG (128 or 256 threads) | `[0, kNumMathThreads-1]`                     | WGMMA 计算 + epilogue           | → `empty_barriers`  |


Multi-stage pipeline 由 `full_barriers`（producer→consumer）和 `empty_barriers`（consumer→producer）管理。

## 2. 寄存器资源账本

```cpp
constexpr uint32_t kNumTMARegisters     = 24;
constexpr uint32_t kNumCpAsyncRegisters = 40;
constexpr uint32_t kNumMathRegisters    = kNumMathThreads == 128 ? 248 : 224;
```

SM90 每 SM 有 **65536** 个寄存器。各配置下的寄存器使用：


| 线程组                          | 线程数     | 寄存器/线程 | 总寄存器       |
| ---------------------------- | ------- | ------ | ---------- |
| TMA WG                       | 128     | 24     | 3,072      |
| cp.async WG                  | 128     | 40     | 5,120      |
| Math WG (128t)               | 128     | 248    | 31,744     |
| **总计 (kNumMathThreads=128)** | **384** | —      | **39,936** |



| 线程组                          | 线程数     | 寄存器/线程 | 总寄存器       |
| ---------------------------- | ------- | ------ | ---------- |
| TMA WG                       | 128     | 24     | 3,072      |
| cp.async WG                  | 128     | 40     | 5,120      |
| Math WG (256t)               | 256     | 224    | 57,344     |
| **总计 (kNumMathThreads=256)** | **512** | —      | **65,536** |


当 `kNumMathThreads=256` 时，寄存器已经**完全打满**。cp.async WG 的 5120 个寄存器成为显著开销。

## 3. cp.async 线程数分析

### 3.1 工作量

以典型配置 `BLOCK_M=64, BLOCK_K=128` 为例：

- A tile 大小：`BLOCK_M × BLOCK_K = 64 × 128 = 8192` 字节
- 每条 cp.async 搬运 16 字节
- 总指令数：`8192 / 16 = 512` 条

不同线程数下的工作分配：


| 线程数 | warp 数 | 每线程 cp.async 条数 |
| --- | ------ | --------------- |
| 128 | 4      | 4               |
| 64  | 2      | 8               |
| 32  | 1      | 16              |


### 3.2 发射速率

SM90 有 4 个 SMSP，每 SMSP 每 cycle 可从其分配的 warp 中发射 1 条指令：


| 线程数 | warp 数 | 峰值发射速率    | 发完 512 条所需 cycle |
| --- | ------ | --------- | ---------------- |
| 128 | 4 warp | 4 条/cycle | ~128 cycles      |
| 64  | 2 warp | 2 条/cycle | ~256 cycles      |
| 32  | 1 warp | 1 条/cycle | ~512 cycles      |


256 cycles（~0.17μs @1.5GHz）在绝大多数场景下远小于一次 WGMMA 计算耗时，**64 线程足够**。

### 3.3 TMA WG 的空闲浪费

TMA WG 每个 stage 只由 **1 个 elected thread** 发 1~2 条 TMA 指令，其余 127 个线程全程在 barrier 上等待。这 128 个线程占了 3072 个寄存器却几乎不做有效工作。

## 4. `wait_group 0` 的流水问题

### 4.1 当前流水时序

```
Stage N:
  empty_barriers[stage]->wait()   ← 阻塞：等 Math 消费完
  issue 512 条 cp.async            ← 发射 A tile + scale_a
  cp.async.commit_group            ← 提交
  cp.async.wait_group 0            ← 阻塞：等所有 cp.async 落地
  NamedBarrier::sync(128, 2)       ← 阻塞：WG 内同步
  full_barriers_a[stage]->arrive() ← 通知 Math: A 就绪
  ────── 然后才能进入 Stage N+1 ──────
```

`wait_group 0` 紧跟 `commit_group`，导致 cp.async WG 在任意时刻**最多只有 1 个 stage 的数据在途**。

### 4.2 在途数据量分析

以 `BLOCK_M=64, BLOCK_K=128` 为例，单 stage A tile = **8 KB**。

要隐藏 ~300–500 cycle 的内存延迟，理想在途数据量：

```
B_inflight ≈ 128 B/cycle × 400 cycles ≈ 50 KB
```

当前 8 KB 远低于目标。虽然 TMA 侧（B 矩阵）也有在途数据，但 A 侧的 cp.async 流水偏保守。

### 4.3 `wait_group 0` vs `wait_group 1`

- `wait_group 0`：等待所有未完成的 cp.async group 完成。当前 stage 必须完全落地后才能开始下一个 stage。
- `wait_group 1`：等待至"最多还剩 1 个 group 未完成"。允许当前 stage 的 cp.async 和下一个 stage 的发射重叠。

## 5. 优化方案

### 方案 A：合并 TMA WG 和 cp.async WG

将两个 producer WG 合并为一个 128 线程的 Producer WG：

```
// 优化前: 3 个 warp-group
TMA WG (128 threads, 24 regs)   → 只用 1 线程发 TMA，127 线程空转
CPA WG (128 threads, 40 regs)   → 全部发 cp.async
Math WG (128/256 threads)

// 优化后: 2 个 warp-group
Producer WG (128 threads, ~40 regs) → elected thread 发 TMA (for B)
                                      全部或部分线程发 cp.async (for A)
Math WG (128/256 threads)
```

**收益：**

- 省掉 128 线程 × 40 寄存器 = **5120 个寄存器**
- TMA WG 的 24 regs/thread 提升到 ~40 regs/thread，**总寄存器反而节省 3072**（因为少了一整个 WG）
- `__launch_bounds__` 的 `maxThreads` 减少 128，有利于编译器优化和 occupancy
- TMA 指令和 cp.async 指令在同一循环体内，减少跨 WG 同步开销

**需要注意的改动：**

- `full_barriers_a` 和 `full_barriers_b` 的信号需在同一个循环体内发出
- TMA 的 `arrive_and_expect_tx` 本身是异步的，与 cp.async 不冲突
- `warpgroup_reg_dealloc` 目标从 24 调高到 ~40

### 方案 B：仅缩减 cp.async 线程到 64

保持 TMA WG 和 cp.async WG 分离，但将 cp.async WG 缩减为 64 线程（2 个 warp），复用 TMA WG 中的空闲 warp。

**收益：**

- 省 2 warp × 32 × 40 = **2560 个寄存器**
- 代码改动较小：只需调整 `kCpAsyncThreads`、线程范围判断和 NamedBarrier 的参与线程数

**权衡：**

- 每线程 cp.async 条数翻倍（4→8），但仍可 `#pragma unroll`
- 发射速率减半（4→2 条/cycle），对 compute-bound 场景无影响

### 方案 C：Pipeline 化 `wait_group`

将 `wait_group 0` 改为延迟等待，实现 cp.async 的 stage 间重叠：

```
// Prologue: 第一个 stage 只发不等
wait empty_barrier[0]
issue cp.async for stage 0
commit_group

// Steady state (stage 1 ~ N-1):
wait empty_barrier[next_stage]
issue cp.async for next_stage
commit_group
wait_group 1                          // 等 prev_stage 完成
NamedBarrier::sync(threads, id)
arrive full_barrier_a[prev_stage]     // 通知 Math: prev_stage 就绪

// Epilogue: 最后一个 stage
wait_group 0
NamedBarrier::sync(threads, id)
arrive full_barrier_a[last_stage]
```

**收益：**

- 在途数据从 1 stage (~~8KB) 翻倍到 2 stage (~~16KB)
- 如果 `kNumStages >= 3`，可用 `wait_group 2` 进一步增加在途深度
- 对 memory-bound 场景（小 M、大 K）收益显著

**权衡：**

- 需要 prologue/epilogue 逻辑，代码复杂度增加
- 对 compute-bound 场景（大 M 大 N）收益有限，因为 Math WG 本身是瓶颈

## 6. 优化优先级建议


| 优先级 | 方案                         | 难度  | 适用场景                   |
| --- | -------------------------- | --- | ---------------------- |
| P0  | 方案 A：合并 TMA+CPA WG         | 中等  | 所有 shape，确定性收益         |
| P0  | 方案 B：CPA 缩减到 64 线程         | 低   | A 的替代方案，改动更小           |
| P1  | 方案 C：Pipeline `wait_group` | 中等  | memory-bound (小 M 大 K) |


方案 A/B 和方案 C 可以独立实施，也可以组合使用。建议先做 A 或 B 验证寄存器收益，再做 C 针对 memory-bound case 进一步调优。

## 7. 方案 D：`cp.async.mbarrier.arrive` 异步信号优化

### 7.1 当前瓶颈回顾

回顾 cp.async WG 的核心循环（L377-386）：

```cpp
// 1. 发射 cp.async 指令（A tile + scale_a）
#pragma unroll
for (uint32_t i = 0; i < kAItersPerThread; ++i) {
    cp_async4(dst_a + ..., gmem_a + ...);
}
// sfa 拷贝 (仅部分线程)
asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ...);

// 2. 提交并阻塞等待
asm volatile("cp.async.commit_group;\n" ::: "memory");
asm volatile("cp.async.wait_group 0;\n"  ::: "memory");  // ← 阻塞点

// 3. WG 内同步 + 单线程信号
cutlass::arch::NamedBarrier::sync(128, 2);               // ← 阻塞点
if (tid_in_cpasync_wg == 0)
    full_barriers_a[stage_idx]->arrive();
```

这里有 **3 次阻塞**：

1. `cp.async.wait_group 0` — 等待所有 cp.async 在 smem 落地
2. `NamedBarrier::sync(128, 2)` — 128 线程 WG 内同步
3. `arrive()` — 仅 1 个线程通知 Math WG

线程在每个 stage 被阻塞两次后才能进入下一个 stage，完全丧失了 cp.async 的异步流水优势。

### 7.2 `cp.async.mbarrier.arrive` 指令详解

#### 7.2.1 指令语法

```
cp.async.mbarrier.arrive{.noinc}.shared::cta.b64 [mbar];
```

- PTX ISA 7.0 引入，SM80+ 可用（SM90 完全支持）
- `[mbar]`：shared memory 中 mbarrier 对象的地址（64-bit，即 `ClusterTransactionBarrier` 底层对象）
- `.noinc`（可选）：是否在 arrive 时递减 mbarrier 的 pending arrive count

#### 7.2.2 语义

> 当执行线程**此前发射的所有 cp.async 操作**全部完成后，硬件自动对 `[mbar]` 执行一次 arrive 操作。
> **执行线程本身不阻塞**，可以立即继续执行后续指令。

关键特性：


| 特性           | `cp.async.wait_group 0` + 手动 `arrive()` | `cp.async.mbarrier.arrive` |
| ------------ | --------------------------------------- | -------------------------- |
| 线程是否阻塞       | 是，阻塞直到所有 cp.async 完成                    | **否**，线程立即继续               |
| arrive 时机    | 线程 resume 后手动执行                         | 硬件在 cp.async 完成时自动执行       |
| 需要 WG 内 sync | 是（`NamedBarrier::sync`）                 | **否**                      |
| 流水重叠         | 不可能（1 stage 在途）                         | 可能（多 stage 在途）             |


#### 7.2.3 `.noinc` vs 不带 `.noinc`

- **不带 `.noinc`**：每次 arrive 递减 mbarrier 的 pending count 1。若 128 个线程都发出此指令，需要 `init(128)`。
- **带 `.noinc`**（推荐）：arrive 时 **不递减** pending count，仅让 mbarrier "追踪"该线程的 cp.async 完成状态。需要配合独立的 arrive 机制来管理 count。

**推荐使用 `.noinc` 变体**，原因：

1. 保持 `full_barriers_a[i]->init(1)` 不变，减少改动
2. 与 TMA 侧的 barrier 协议一致
3. 结合 `NamedBarrier` + 单线程 `arrive()` 更直观

但如果要 **彻底消除 `NamedBarrier::sync`**（进一步减少同步开销），应使用**不带 `.noinc`** 的版本，让每个线程的 cp.async 完成时独立 arrive，将 init count 改为 `kCpAsyncThreads`（128）。

#### 7.2.4 内存序保证

PTX ISA 对 `mbarrier.try_wait` 的内存序规定：

> 当 `mbarrier.try_wait` 返回 true 时：
> 所有参与 CTA 线程在 `cp.async.mbarrier.arrive` **之前**发射的 cp.async 操作，
> 对执行 wait 的线程**可见**。

这意味着 Math WG 在 `full_barriers_a[stage]->wait(phase)` 返回后，可以安全读取该 stage 的 A tile 和 scale_a 数据——不需要额外的 fence。

### 7.3 优化方案·

#### 方案 D1：最小改动 — `.noinc` + 保留 NamedBarrier（消除 `wait_group 0`）

将 `wait_group 0` 替换为 `cp.async.mbarrier.arrive.noinc`，但保留 NamedBarrier 来保证所有线程的 cp.async 都已被 mbarrier 追踪：

```cpp
// After cp.async for A and sfa:
asm volatile("cp.async.commit_group;\n" ::: "memory");

// 每个线程：让 mbarrier 追踪我的 cp.async（不阻塞线程）
asm volatile(
    "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n"
    :: "r"(static_cast<uint32_t>(
           __cvta_generic_to_shared(full_barriers_a[stage_idx])))
    : "memory");

// WG 内同步：确保所有线程的 noinc arrive 已发出
cutlass::arch::NamedBarrier::sync(128, 2);

// 单线程执行真正的 arrive（递减 count）
if (tid_in_cpasync_wg == 0)
    full_barriers_a[stage_idx]->arrive();
```

**barrier 初始化不变**：`full_barriers_a[i]->init(1)`

**原理**：

- 每个线程的 `.noinc` arrive 让 mbarrier 追踪该线程的所有 cp.async
- `NamedBarrier::sync` 保证所有 128 个 `.noinc` arrive 都已发出
- 此后 thread 0 的 `arrive()` 递减 count，mbarrier 转 phase
- Math WG 的 `wait(phase)` 返回时，**内存序保证**所有 cp.async 数据可见

**收益**：消除了 `wait_group 0` 的阻塞，cp.async 硬件流水线不被打断。但 `NamedBarrier::sync` 仍存在。

#### 方案 D2：彻底消除同步 — 不带 `.noinc`（推荐）

让每个线程独立 arrive，完全去掉 `wait_group 0` 和 `NamedBarrier::sync`：

```cpp
// After cp.async for A and sfa:
asm volatile("cp.async.commit_group;\n" ::: "memory");

// 每个线程：当我的 cp.async 完成时，自动在 mbarrier 上 arrive
// 线程不阻塞，立即进入下一个 stage
asm volatile(
    "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n"
    :: "r"(static_cast<uint32_t>(
           __cvta_generic_to_shared(full_barriers_a[stage_idx])))
    : "memory");

// 不需要 NamedBarrier::sync(128, 2)
// 不需要 if (tid == 0) arrive()
```

**barrier 初始化改为**：`full_barriers_a[i]->init(kCpAsyncThreads)` （128）

**原理**：

- 128 个线程各自发出 `cp.async.mbarrier.arrive`
- 硬件异步地在每个线程的 cp.async 完成时执行 arrive（递减 count）
- 当所有 128 个 arrive 完成（即所有线程的所有 cp.async 数据落地），mbarrier 转 phase
- Math WG 的 `wait(phase)` 感知到 phase 翻转，读取数据
- 线程立即继续执行下一个 stage 的 `empty_barriers` wait + cp.async 发射

**关于 sfa 加载**：不是所有线程都发射 sfa 的 cp.async（仅 `tid_in_cpasync_wg < BLOCK_M && m_global_base + tid < shape_m` 的线程）。但所有线程都发射 A tile 的 cp.async（`kAItersPerThread >= 1`）。`cp.async.mbarrier.arrive` 覆盖"该线程此前所有 cp.async"，因此：

- 发了 A+sfa 的线程：arrive 在两者都完成后触发
- 只发了 A 的线程：arrive 在 A 完成后触发
- mbarrier 在所有 128 个 arrive 后转 phase → A tile 和 sfa 均已落地 ✓

### 7.4 流水效果对比

#### 当前时序（`wait_group 0`）

```
cp.async WG 时间线:
─── Stage 0 ──────────── Stage 1 ──────────── Stage 2 ───
│ wait_empty │ issue │ BLOCK │ sync│arr│ wait_empty │ issue │ BLOCK │ sync│arr│ ...
│            │cp.async│wait_0│     │   │            │cp.async│wait_0│     │   │
                       ~~~~~~                               ~~~~~~
                       阻塞!                                 阻塞!

在途 cp.async 数据: 最多 1 stage (8KB for BLOCK_M=64)
```

#### 方案 D2 时序（`cp.async.mbarrier.arrive`）

```
cp.async WG 时间线:
─── Stage 0 ──── Stage 1 ──── Stage 2 ──── Stage 3 ───
│wait_empty│issue│arr│wait_empty│issue│arr│wait_empty│issue│arr│ ...
│          │ cp  │   │          │ cp  │   │          │ cp  │   │
           ↑         ↑               ↑
           不阻塞！    不阻塞！         不阻塞！

在途 cp.async 数据: 最多 kNumStages stages
（由 empty_barriers 做背压，不会超发）
```

**核心改进**：线程在发完当前 stage 的 cp.async 后 **不等待落地**，立即去看下一个 stage 的 `empty_barriers` 是否就绪。如果 Math WG 已经消费完，可以立刻发射下一个 stage 的 cp.async。

### 7.5 cp.async commit group 与多 stage 在途的正确性

一个关键问题：如果 stage N 的 cp.async 尚未完成，线程开始发射 stage N+1 的 cp.async，两个 stage 的 `cp.async.mbarrier.arrive` 是否互相干扰？

**不会**。原因如下：

1. **commit_group 建立组边界**：stage N 的 cp.async 在 `cp.async.commit_group` 后形成 group 0；stage N+1 的 cp.async 在下一个 `commit_group` 后形成 group 1。
2. **组按 FIFO 顺序完成**：group 0 保证在 group 1 之前完成。
3. **每个 mbarrier.arrive 覆盖"所有先前 cp.async"**：
  - stage N 的 arrive（group 0 后发出）：在 group 0 完成时触发
  - stage N+1 的 arrive（group 0+1 后发出）：在 group 0 **和** group 1 都完成时触发
  - 由于 FIFO 顺序，stage N+1 的 arrive 实际上在 group 1 完成时触发（此时 group 0 必定已完成）

```
时间线:
T0: issue cp.async for stage N
T1: cp.async.commit_group                     → group 0
T2: cp.async.mbarrier.arrive [mbar_N]         → fires when group 0 done
T3: (线程不阻塞，继续)
T4: empty_barriers[N+1]->wait(phase ^ 1)
T5: issue cp.async for stage N+1
T6: cp.async.commit_group                     → group 1
T7: cp.async.mbarrier.arrive [mbar_{N+1}]    → fires when group 0+1 done
                                                (即 group 1 完成时)
...
TX: group 0 完成 → mbar_N arrives → Math WG 可读 stage N ✓
TY: group 1 完成 → mbar_{N+1} arrives → Math WG 可读 stage N+1 ✓
```

### 7.6 具体代码改动（方案 D2）

#### 7.6.1 PTX wrapper（建议加入 `ptx/tma.cuh`）

```cpp
CUTLASS_DEVICE void cp_async_mbarrier_arrive(
    cutlass::arch::ClusterTransactionBarrier* barrier) {
    asm volatile(
        "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(barrier)))
        : "memory");
}

// .noinc 版本（方案 D1 使用）
CUTLASS_DEVICE void cp_async_mbarrier_arrive_noinc(
    cutlass::arch::ClusterTransactionBarrier* barrier) {
    asm volatile(
        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(barrier)))
        : "memory");
}
```

#### 7.6.2 Barrier 初始化改动

```cpp
// 方案 D2: 每个 cp.async 线程独立 arrive
full_barriers_a[i]->init(kCpAsyncThreads);  // 128

// 方案 D1: 保持不变
full_barriers_a[i]->init(1);
```

#### 7.6.3 cp.async WG 主循环改动（方案 D2 完整 diff）

```diff
 // After cp.async for A tile and sfa:

-                // Commit and wait for all cp.async in this group to complete
-                asm volatile("cp.async.commit_group;\n" ::: "memory");
-                asm volatile("cp.async.wait_group 0;\n"  ::: "memory");
-                cutlass::arch::NamedBarrier::sync(128, 2);
-                // One thread signals that A is ready for this stage
-                if (tid_in_cpasync_wg == 0) {
-                    full_barriers_a[stage_idx]->arrive();
-                }
+                // Commit cp.async group; async arrive will fire when copies land
+                asm volatile("cp.async.commit_group;\n" ::: "memory");
+                cp_async_mbarrier_arrive(full_barriers_a[stage_idx]);
```

### 7.7 性能分析

#### 7.7.1 延迟节省

每个 stage 减少的阻塞：


| 阻塞项                          | 估算延迟                        | 是否消除 |
| ---------------------------- | --------------------------- | ---- |
| `cp.async.wait_group 0`      | 300-500 cycles（取决于 DRAM 延迟） | ✓ 消除 |
| `NamedBarrier::sync(128, 2)` | ~20-40 cycles               | ✓ 消除 |
| 合计                           | ~320-540 cycles/stage       |      |


对于 `num_total_k_blocks = shape_k / BLOCK_K` 个 stage，总节省：

```
~400 cycles × (shape_k / 128) stages × (1 / 1.5 GHz) 
≈ 0.27 μs × (shape_k / 128)
```

#### 7.7.2 第三方实测数据

ThunderKittens 项目（HazyResearch/ThunderKittens#97）在通用 matmul kernel 上的对比：

> "manually waiting on a semaphore with `cp.async.wait_all` plus an explicit `arrive(bar)` is **over 200 TFLOPS slower** than allowing `cp.async` to automatically signal the semaphore"

这证实了 `cp.async.mbarrier.arrive` 的显著性能优势，尤其在 producer-consumer 模式下。

#### 7.7.3 收益场景分析


| 场景                     | 收益    | 原因                               |
| ---------------------- | ----- | -------------------------------- |
| Memory-bound（小 M 大 K）  | **高** | cp.async 是关键路径，pipeline 重叠直接减少延迟 |
| Balanced（中等 M/K）       | **中** | Math WG 和 cp.async WG 接近平衡，减少气泡  |
| Compute-bound（大 M 大 N） | **低** | Math WG 是瓶颈，cp.async 延迟被隐藏       |


### 7.8 注意事项与风险

1. **寄存器开销**：`cp.async.mbarrier.arrive` 指令本身只需要 mbarrier 地址（1 个寄存器），不增加寄存器压力。当前 `kNumCpAsyncRegisters = 40` 足够。
2. **Barrier init count**：方案 D2 将 `full_barriers_a` 的 init count 从 1 改为 128。需要确认 `ClusterTransactionBarrier` 支持大于 1 的 arrive count（SM90 mbarrier 的 arrive count 上限为 2^11 - 1 = 2047，128 完全在范围内）。
3. **与 `kNumTMAMulticast > 1` 的交互**：当使用 TMA multicast 时，`empty_barriers` 的 arrive 涉及跨 CTA 信号。`full_barriers_a` 仅在 CTA 内部使用（cp.async WG → Math WG），不受 multicast 影响。
4. **Epilogue 阶段的 `empty_barriers` 额外 wait**（L388-392）：在 `kNumTMAMulticast > 1` 时，循环结束后有额外的 empty wait。这不受 full_barriers_a 改动影响。
5. `**commit_group` 仍然必须保留**：`cp.async.mbarrier.arrive` 覆盖"所有先前 cp.async"，但 commit_group 建立组边界，确保硬件按组 FIFO 完成。即使不需要 `wait_group`，commit_group 仍是正确性所需。
6. **与方案 A/B/C 的组合**：
  - 方案 D 可以与方案 A（合并 WG）或 B（缩减线程）**独立实施**
  - 与方案 C（pipeline `wait_group`）功能 **重叠**：方案 D2 直接实现了比方案 C 更彻底的 pipeline 化，因为线程完全不阻塞
  - **推荐优先实施方案 D2**，它替代了方案 C 的必要性

### 7.9 优先级更新


| 优先级    | 方案                                | 难度     | 改动量    | 适用场景                     |
| ------ | --------------------------------- | ------ | ------ | ------------------------ |
| **P0** | **D2：`cp.async.mbarrier.arrive`** | **低**  | ~10 行  | 所有 shape，尤其 memory-bound |
| P0     | A：合并 TMA+CPA WG                   | 中等     | ~100 行 | 所有 shape，省寄存器            |
| P0     | B：CPA 缩减到 64 线程                   | 低      | ~20 行  | A 的替代方案                  |
| P1     | ~~C：Pipeline `wait_group`~~       | ~~中等~~ | —      | 被 D2 取代                  |


方案 D2 改动最小（约 10 行），风险最低，且直接解决了文档第 4 节分析的 `wait_group 0` 流水瓶颈。建议作为**第一个实施的优化**。

## 8. 方案 E：cp.async 合并访存优化（iter-major 映射）

### 8.1 适用前提

方案 E 是在方案 A（合并 TMA+CPA WG → Producer WG）**已经实施**之后的增量优化。此时 Producer WG 的 128 线程在同一循环体内发射 cp.async，线程↔数据的映射方式就成为 gmem 访问效率的关键变量。

### 8.2 问题：tid-major 映射导致 warp 内访存分散

合并后 Producer WG 最自然的写法沿用了原 cp.async WG 的映射：

```cpp
#pragma unroll
for (uint32_t i = 0; i < kAItersPerThread; ++i) {
    const uint32_t linear = (tid_in_wg * kAItersPerThread + i) * kCpAsyncWidth;
    const uint32_t row = linear / BLOCK_K;
    const uint32_t col = linear % BLOCK_K;
    cp_async4(dst_a + row * BLOCK_K + (col ^ ((row % 8) * 16)),
              gmem_a + (m_global_base + row) * stride_a + k_idx + col);
}
```

关键参数：`BLOCK_K = 128`, `kCpAsyncWidth = 16`, `kCpAsyncThreads = 128`, `kAItersPerThread = BLOCK_M / 16`。

以 **BLOCK_M=64 (kAItersPerThread=4)** 为例，展开 `warp 0 在 iter 0` 的 lane → (row, col) 分布：

| lane | row | col (gmem bytes) |
|------|-----|------------------|
| 0    | 0   | 0–15             |
| 1    | 0   | 64–79            |
| 2    | 1   | 0–15             |
| 3    | 1   | 64–79            |
| …    | …   | …                |
| 30   | 15  | 0–15             |
| 31   | 15  | 64–79            |

一个 warp 的 32 条 cp.async 命中 **16 条不同的 cache line**（16 个 A 行），每条 line 只覆盖 2 个 16B 扇区（cols 0–15 和 64–79，共 32B / 128B ≈ **25% 利用率**）。

### 8.3 L2 能救带宽但救不了事务数

4 个 iter 的列覆盖如下：

| iter | 访问 cols          | 覆盖的 L2 sector（每行） | 是否触发 HBM |
|------|--------------------|----------------------|-------------|
| 0    | 0–15 + 64–79       | sector 0, sector 2   | 是          |
| 1    | 16–31 + 80–95      | sector 0, sector 2   | L2 hit      |
| 2    | 32–47 + 96–111     | sector 1, sector 3   | 是          |
| 3    | 48–63 + 112–127    | sector 1, sector 3   | L2 hit      |

**HBM 总流量**：16 行 × 128B = 2048B per warp —— 和理论最优相同。L2 把重复请求完全吸收掉了。

但代价是 **LSU → L2 事务数**：

- 当前 pattern：每 iter 16 条 L2 事务（按 32B sector 对齐），4 iter = **64 条**事务/warp
- iter-major pattern：每 iter 4 条 L2 事务（按 128B cache-line 粒度），4 iter = **16 条**事务/warp

事务数的 4 倍差距直接影响：

1. **L1TEX / L2 请求队列深度**：H100 每 SM 的 L1TEX 吞吐固定，更多请求会排队。
2. **LSU 发射阻塞**：cp.async 虽然 async，但 LSU 本身发射仍然受限。
3. **L2 tag lookup 次数**：每条请求都要走一次 tag lookup。
4. **对 L2 压力敏感**：当前 pattern 依赖 L2 跨 iter 缓存 2KB/warp。多 CTA 并发时与 B、CD、下游 SFA 竞争 60MB L2，被驱逐就得二次走 HBM。

### 8.4 优化：iter-major（row-group）映射

只改一行：

```cpp
// 原：
const uint32_t linear = (tid_in_wg * kAItersPerThread + i) * kCpAsyncWidth;
// 改为：
const uint32_t linear = (i * kCpAsyncThreads + tid_in_wg) * kCpAsyncWidth;
```

映射效果（BLOCK_M=64, kAItersPerThread=4），warp 0 iter 0 变为：

| lane   | row | col     | 备注                          |
|--------|-----|---------|------------------------------|
| 0      | 0   | 0       |                              |
| 1      | 0   | 16      |                              |
| …      | 0   | …       | 整行 128B 被 lane 0..7 完整覆盖 |
| 7      | 0   | 112     |                              |
| 8      | 1   | 0       |                              |
| 15     | 1   | 112     | 整行 128B 被 lane 8..15 覆盖   |
| 16..23 | 2   | 0..112  | 整行                          |
| 24..31 | 3   | 0..112  | 整行                          |

**一个 warp 一次 iter 仅命中 4 条 cache line**，每条被 8 条 lane 的 8 个 16B 连续覆盖 —— HW 完美 coalesce 为 **4 × 128B 请求**（vs. 当前 16 × 32B 请求）。

### 8.5 收益对比（per warp, 完整跑完 4 iter）

| 指标               | 当前 (tid-major) | 优化 (iter-major) | 比例        |
|-------------------|------------------|-------------------|-------------|
| LSU → L2 事务数    | 64               | 16                | **4×**      |
| 事务粒度            | 32B              | 128B              | 4×          |
| HBM 字节 (L2 命中) | 2048B            | 2048B             | 1×          |
| HBM 字节 (L2 miss) | 最多 8192B       | 2048B             | **≤ 4×**    |
| SMEM bank 冲突     | 无               | 无                | —           |
| SMEM 最终内容       | 相同             | 相同              | —（无副作用）|

- **L2 有余量** 时（小 CTA footprint），收益主要来自 LSU/L2 请求队列压力降低。
- **L2 压力大** 时（大 M/N 多波、多 SM 争用），当前 pattern 会因 L2 驱逐产生额外 HBM 访问；优化后稳定在 2048B/warp，最坏情况的收益更显著。

### 8.6 各 BLOCK_M 覆盖验证

新公式 `linear = (i * 128 + tid_in_wg) * 16`，row = linear / 128，col = linear % 128。跨所有 `(i, tid_in_wg)` 对的 `linear` 集合等于 `{0, 16, …, SMEM_A_SIZE_PER_STAGE - 16}`，和原公式**完全相同**，保证无重写、无遗漏：

| BLOCK_M | kAItersPerThread | Iter k 覆盖的 rows                 | 总 16B 写次数 | SMEM_A_SIZE_PER_STAGE |
|---------|------------------|------------------------------------|---------------|------------------------|
| 16      | 1                | i=0: rows 0..15                    | 128           | 2048B ✓                |
| 32      | 2                | i=0: 0..15, i=1: 16..31           | 256           | 4096B ✓                |
| 64      | 4                | i=0..3: 0..15, 16..31, 32..47, 48..63 | 512       | 8192B ✓                |
| 128     | 8                | 逐 iter 覆盖 0..15, …, 112..127   | 1024          | 16384B ✓               |
| 256     | 16               | 逐 iter 覆盖 0..15, …, 240..255   | 2048          | 32768B ✓               |

### 8.7 Swizzle / SMEM 读侧等价性

- swizzle 公式 `col ^ ((row % 8) * 16)` **完全不变**。
- 原公式与新公式枚举出的 `(row, col)` 对集合相同，只是"哪个线程写哪对"的分工不同。
- SMEM A tile 在所有写入完成后的位模式 **逐字节等价**。
- WGMMA 侧 `make_smem_desc(smem_a, /* layout_type = */ 1)` 无需任何改动。

### 8.8 代码改动（1 行）

```diff
 for (uint32_t i = 0; i < kAItersPerThread; ++i) {
-    const uint32_t linear = (tid_in_wg * kAItersPerThread + i) * kCpAsyncWidth;
+    const uint32_t linear = (i * kCpAsyncThreads + tid_in_wg) * kCpAsyncWidth;
     const uint32_t row = linear / BLOCK_K;
     const uint32_t col = linear % BLOCK_K;
     cp_async4(dst_a + row * BLOCK_K + (col ^ ((row % 8) * 16)),
               gmem_a + (m_global_base + row) * stride_a + k_idx + col);
 }
```

### 8.9 可观测指标（Nsight Compute）

预期 metric 变化：

| Metric                                                                | 预期方向        | 说明                                    |
|-----------------------------------------------------------------------|-----------------|---------------------------------------|
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`                     | **↓ 显著**（~4×） | LSU 发出的 sector 请求数减少             |
| `l1tex__throughput.avg.pct_of_peak_sustained_active`                 | ↓               | 请求少了，单位时间 L1TEX 占用下降         |
| `smsp__inst_executed_pipe_lsu.sum`                                   | ≈ 不变          | 每线程发射的 cp.async 条数不变           |
| `dram__bytes.sum`                                                    | ≈ 不变          | 数据总量相同（L2 命中时）                |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld_hits_in_l2`             | 相对更高        | 因为请求粒度 = cache line，L2 命中率高  |

端到端影响：
- LSU-bound 或 memory-latency-bound 形状：**kernel time 下降**。
- Compute-bound 形状（大 M 大 N）：端到端时间基本不变，但 Producer WG 更"闲"，stage 间 bubble 变小，为未来进一步提升 pipeline depth 留空间。

### 8.10 注意事项与风险

1. **依赖方案 A**：只有合并 WG 之后 Producer 的 128 线程才在同一循环体内，iter-major 重映射才有意义。
2. **无正确性风险**：数据集合与 swizzle 不变，只是线程分工换序；SMEM 终态、Math WG 视图均不变。
3. **无 SMEM bank 增量冲突**：128B swizzle 本身的 bank 分布由 swizzle 公式决定，不受 iter 顺序影响。
4. **sfa cp.async 未动**：sfa 每线程 4B，32 lane × 4B = 128B 本来就是 coalesced 的；且 sfa 总量小（256B~1KB/stage），不是瓶颈。
5. **不依赖 `stride_a` 的新对齐约束**：cp.async.cg 16B 的 src/dst 对齐要求是方案 A 就已经承担的前提，方案 E 不新增任何约束。

### 8.11 与其他方案的关系

- 与 **方案 A** 是"合并 → 再优化"的关系：E 必须在 A 之后。
- 与 **方案 D2** 正交：D2 改的是 **stage 之间**的信号机制（去除 `wait_group 0`），E 改的是 **stage 内部** cp.async 的线程分工。两者同时生效，没有耦合。
- 与 **方案 B** 冲突：B 保留两个 WG 但把 CPA 缩到 64 线程。如果选 B，iter-major 映射的 `kCpAsyncThreads` 需要改成 64，覆盖逻辑同样适用但每 iter 只覆盖 8 行（128/8 × 64/8 = ... 需重推），建议选 A+E 而不是 B。

## 9. 优化优先级（最终版）

| 优先级    | 方案                                     | 难度    | 改动量    | 适用场景                     |
| ------ | --------------------------------------- | ------ | ------- | ---------------------------- |
| **P0** | **D2：`cp.async.mbarrier.arrive`**      | **低** | ~10 行  | 所有 shape，尤其 memory-bound |
| **P0** | **A：合并 TMA+CPA WG**                  | 中等    | ~100 行 | 所有 shape，省寄存器           |
| **P0** | **E：iter-major 合并访存映射**（需 A） | **极低** | **1 行** | 所有 shape，LSU/L2 压力大时尤甚 |
| P1     | B：CPA 缩减到 64 线程                   | 低      | ~20 行  | A 的替代方案                   |
| P1     | ~~C：Pipeline `wait_group`~~           | ~~中等~~ | —       | 被 D2 取代                     |

推荐实施顺序：**D2 → A → E**。三者独立落地都能单独验证收益。

## 10. 附：关键代码位置

> 行号对应方案 A（合并 WG）+ D2（`cp.async.mbarrier.arrive.noinc`）+ E（iter-major 映射）落地后的状态。

| 内容                                     | 文件                       | 行号       |
| --------------------------------------- | ------------------------ | -------- |
| Barrier layout 注释                      | `sm90_fp8_gemm_1d2d.cuh` | L210-215 |
| Barrier 初始化（含 full_barriers_a init=128）| `sm90_fp8_gemm_1d2d.cuh` | L243-257 |
| Register 常量（含 kNumProducerRegisters）  | `sm90_fp8_gemm_1d2d.cuh` | L262-266 |
| cp.async 常量定义                         | `sm90_fp8_gemm_1d2d.cuh` | L268-274 |
| Producer WG 注释与入口                    | `sm90_fp8_gemm_1d2d.cuh` | L291-309 |
| Rotating-warp TMA B                      | `sm90_fp8_gemm_1d2d.cuh` | L332-347 |
| cp.async A（iter-major 映射）             | `sm90_fp8_gemm_1d2d.cuh` | L348-372 |
| cp.async sfa                             | `sm90_fp8_gemm_1d2d.cuh` | L374-389 |
| `cp.async.mbarrier.arrive.noinc` 调用    | `sm90_fp8_gemm_1d2d.cuh` | L391-398 |
| Math WG 主循环                            | `sm90_fp8_gemm_1d2d.cuh` | L407-643 |
| PTX wrapper (`cp_async_mbarrier_arrive_noinc`) | `ptx/tma.cuh`      | L34-43   |
| Launch 配置（`num_tma_threads=128`）      | `csrc/jit_kernels/heuristics/sm90.hpp` | L191-203 |


## 11. 参考资料

- [PTX ISA 9.2 - cp.async.mbarrier.arrive](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive)
- [NVIDIA/cccl#3602 - PTX: Add cp.async.mbarrier.arrive{.noinc}](https://github.com/NVIDIA/cccl/pull/3602)
- [HazyResearch/ThunderKittens#97 - Add Semaphore Support for cp.async loads](https://github.com/HazyResearch/ThunderKittens/pull/97) — 实测 200+ TFLOPS 提升
- [MLIR NVVM Dialect - CpAsyncMBarrierArriveOp](https://mlir.llvm.org/docs/Dialects/NVVMDialect/#nvvmcpasyncmbarrierarrive-nvvmcpasyncmbarrierarriveop)
- [CUDA Programming Guide - Global Memory Access Patterns](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-memory) — warp-level coalescing 与 128B cache-line 的对齐关系
- [Nsight Compute Metrics Reference - l1tex Section](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#l1tex) — 方案 E 推荐用于 A/B 测试的 metric

