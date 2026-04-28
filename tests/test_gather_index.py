"""
Accuracy + perf bench for the SM90 FP8 1D2D GEMM `gather_index` feature.

For each shape we compare three paths and present them as a single summary
table at the very end:

  [B] baseline  — pure GEMM. A is already in the layout the kernel wants
                  (pre-permuted). No gather work whatsoever. This is the
                  "upper bound" for the GEMM portion alone.

  [E] explicit  — `torch.index_select(A_pool, gather_index)` (and the same
                  for sfa) before the GEMM. This is the typical MoE
                  pre-processing path users write today.

  [K] kernel    — `<gemm_api>(A_pool, ..., gather_index=g)`. Kernel reads
                  A and sfa rows through `gather_index` while computing.

The end-to-end speedup of using `gather_index` is reported as `E_us / K_us`.

Two test functions are exposed:

  - `test_gather_gemm`            : GemmType::Normal
  - `test_gather_m_grouped_gemm_contiguous` : MGroupedContiguous
      (the real MoE shape).

Use `--tests` to pick which to run (see `--help`).

Notes:
  - We use `bench()` (wall time) for all paths so the explicit_gather path —
    which incurs an additional `index_select` kernel — is measured fairly
    against the single-kernel paths.
  - Each path is JIT-warmed with stdout suppressed; deep_gemm's JIT compile
    chatter (`num_tma_threads: ...`) doesn't pollute the table.
"""

import argparse
import os
import shutil

cache_dir = os.path.expanduser("~/.deep_gemm/cache")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("Clearing DeepGEMM JIT cache...")

import sys
import torch

import deep_gemm
from deep_gemm.testing import (
    bench, calc_diff, get_arch_major,
)
from deep_gemm.testing.bench import suppress_stdout_stderr
from deep_gemm.utils import (
    align, get_mk_alignment_for_contiguous_layout
)

sys.path.insert(0, os.path.dirname(__file__))
from generators import (
    KernelType, MajorTypeAB, QuantConfig,
    cast_fp8_fp4_with_major,
    grouped_cast_fp8_fp4_with_major,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _silent_warmup(*fns):
    """Run each fn once with stdout/stderr suppressed (silences JIT prints)."""
    with suppress_stdout_stderr():
        for fn in fns:
            fn()
    torch.cuda.synchronize()


def _bench_silent(fn) -> float:
    """`bench()` wrapped in stdout suppression; returns seconds (averaged)."""
    with suppress_stdout_stderr():
        return bench(fn)


def _fp8_a(a_bf16, quant_config, use_ue8m0):
    return cast_fp8_fp4_with_major(a_bf16, MajorTypeAB.KMajor,
                                   quant_config.gran_k_a, quant_config.is_fp4_a, use_ue8m0)


def _fp8_b(b_bf16, quant_config, kernel_type, use_ue8m0, *, grouped=False):
    if grouped:
        return grouped_cast_fp8_fp4_with_major(
            b_bf16, MajorTypeAB.KMajor,
            quant_config.gran_k_b, quant_config.is_fp4_b, use_ue8m0,
            use_block_cast_for_fp8=True)
    return cast_fp8_fp4_with_major(b_bf16, MajorTypeAB.KMajor,
                                   quant_config.gran_k_b, quant_config.is_fp4_b, use_ue8m0,
                                   use_block_cast_for_fp8=not kernel_type.is_1d1d())


def _format_table(title: str, headers, rows):
    if not rows:
        return
    cell_rows = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in cell_rows)) for i, h in enumerate(headers)]
    sep = ' | '
    header_line = sep.join(f'{h:>{w}}' for h, w in zip(headers, widths))
    bar = '=' * len(header_line)

    print()
    print(bar)
    print(f'  {title}')
    print(bar)
    print(header_line)
    print('-' * len(header_line))
    for r in cell_rows:
        print(sep.join(f'{c:>{w}}' for c, w in zip(r, widths)))
    print(bar)


def _us(t_seconds: float) -> str:
    return f'{t_seconds * 1e6:.0f}'


def _tflops(flops: int, t_seconds: float) -> str:
    return f'{flops / t_seconds / 1e12:.0f}'


# ---------------------------------------------------------------------------
# Normal GEMM (GemmType::Normal)
# ---------------------------------------------------------------------------
def test_gather_gemm() -> None:
    print('Testing FP8 GEMM with gather_index (SM90 1D2D, GemmType::Normal):')
    print('  3 paths per shape, m_pool = 2 * m, gather m of m_pool rows:')
    print('    [B] baseline — fp8_gemm_nt(A_perm, ...)                  (A already permuted)')
    print('    [E] explicit — index_select(A_pool, g) + fp8_gemm_nt     (current practice)')
    print('    [K] kernel   — fp8_gemm_nt(A_pool, ..., gather_index=g)  (this PR)')
    print()

    quant_config = QuantConfig()
    kernel_type = KernelType.Kernel1D2D
    use_ue8m0 = False
    disable_ue8m0_cast = not use_ue8m0
    recipe, recipe_a, recipe_b = quant_config.get_recipes()

    # nk pairs from `tests/test_fp8_fp4.py:test_gemm` fwd path (bf16 output).
    bf16_output_nk = [(2112, 7168), (4096, 7168), (7168, 2048),
                      (24576, 1536), (7168, 16384)]
    m_list = [128, 1024, 4096, 8192]

    summary = []
    for m in m_list:
        for n, k in bf16_output_nk:
            torch.manual_seed(0)
            m_pool = m * 2
            a_pool_bf16 = torch.randn((m_pool, k), device='cuda', dtype=torch.bfloat16)
            b_bf16 = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
            a_pool_fp8 = _fp8_a(a_pool_bf16, quant_config, use_ue8m0)
            b_fp8 = _fp8_b(b_bf16, quant_config, kernel_type, use_ue8m0)

            g = torch.randperm(m_pool, device='cuda')[:m].to(torch.int32).contiguous()
            g_long = g.long()

            a_perm_bf16 = a_pool_bf16[g_long]
            a_perm_fp8 = _fp8_a(a_perm_bf16, quant_config, use_ue8m0)

            d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
            ref_d = (a_perm_bf16.float() @ b_bf16.float().t()).to(torch.bfloat16)

            def fn_baseline():
                deep_gemm.fp8_gemm_nt(a_perm_fp8, b_fp8, d, recipe=recipe,
                                      recipe_a=recipe_a, recipe_b=recipe_b,
                                      disable_ue8m0_cast=disable_ue8m0_cast)

            def fn_explicit():
                a0 = a_pool_fp8[0].index_select(0, g_long)
                a1 = a_pool_fp8[1].index_select(0, g_long)
                deep_gemm.fp8_gemm_nt((a0, a1), b_fp8, d, recipe=recipe,
                                      recipe_a=recipe_a, recipe_b=recipe_b,
                                      disable_ue8m0_cast=disable_ue8m0_cast)

            def fn_kernel():
                deep_gemm.fp8_gemm_nt(a_pool_fp8, b_fp8, d, recipe=recipe,
                                      recipe_a=recipe_a, recipe_b=recipe_b,
                                      disable_ue8m0_cast=disable_ue8m0_cast,
                                      gather_index=g)

            _silent_warmup(fn_baseline, fn_explicit, fn_kernel)

            # ---- correctness ----
            fn_baseline(); d_baseline = d.clone()
            diff_b = calc_diff(d_baseline, ref_d)
            fn_explicit(); d_explicit = d.clone()
            diff_e = calc_diff(d_explicit, ref_d)
            fn_kernel(); diff_k = calc_diff(d, ref_d)
            diff_kvb = calc_diff(d, d_baseline)
            diff_kve = calc_diff(d, d_explicit)
            assert diff_b < quant_config.max_diff(), \
                f'baseline diff vs ref fails: m={m}, n={n}, k={k}, diff={diff_b:.5f}'
            assert diff_e < quant_config.max_diff(), \
                f'explicit diff vs ref fails: m={m}, n={n}, k={k}, diff={diff_e:.5f}'
            assert diff_k < quant_config.max_diff(), \
                f'kernel diff vs ref fails: m={m}, n={n}, k={k}, diff={diff_k:.5f}'
            assert diff_kvb < 1e-6, \
                f'kernel diverges from baseline: m={m}, n={n}, k={k}, diff={diff_kvb:.2e}'
            assert diff_kve < 1e-6, \
                f'kernel diverges from explicit: m={m}, n={n}, k={k}, diff={diff_kve:.2e}'

            # ---- bench (wall time, includes kernel launch overhead) ----
            t_b = _bench_silent(fn_baseline)
            t_e = _bench_silent(fn_explicit)
            t_k = _bench_silent(fn_kernel)
            flops = 2 * m * n * k
            speedup = t_e / t_k

            print(f'  > m={m:5}, n={n:5}, k={k:5} | '
                  f'B={_us(t_b):>5}us E={_us(t_e):>5}us K={_us(t_k):>5}us  e2e={speedup:.2f}x')

            summary.append((
                m, n, k,
                _us(t_b), _tflops(flops, t_b),
                _us(t_e), _tflops(flops, t_e),
                _us(t_k), _tflops(flops, t_k),
                f'{speedup:.2f}x',
                f'{diff_b:.0e}', f'{diff_k:.0e}',
            ))

    headers = ['m', 'n', 'k',
               'B us', 'B TFLOPS',
               'E us', 'E TFLOPS',
               'K us', 'K TFLOPS',
               'e2e (E/K)', 'diff_B', 'diff_K']
    _format_table(
        'Summary: Normal fp8 GEMM with gather_index (m_pool = 2m, randperm)',
        headers, summary)


# ---------------------------------------------------------------------------
# M-grouped contiguous GEMM (GemmType::MGroupedContiguous{,WithPsumLayout})
# ---------------------------------------------------------------------------
def _make_grouped_contiguous_inputs(num_groups, expected_m_per_group, n, k,
                                    use_psum_layout):
    import random
    actual_ms = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
    aligned_ms = [align(am, get_mk_alignment_for_contiguous_layout()) for am in actual_ms]
    m_total = sum(aligned_ms)

    a_bf16 = torch.randn((m_total, k), device='cuda', dtype=torch.bfloat16)
    b_bf16 = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    grouped_layout = torch.empty(num_groups, device='cuda', dtype=torch.int32) if use_psum_layout \
                     else torch.empty(m_total, device='cuda', dtype=torch.int32)
    ref_d = torch.empty((m_total, n), device='cuda', dtype=torch.bfloat16)

    start = 0
    for i, (am, alm) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end = start + am
        aligned_end = start + alm
        if use_psum_layout:
            grouped_layout[i] = actual_end
        else:
            grouped_layout[start: actual_end] = i
            grouped_layout[actual_end: aligned_end] = -1
        a_bf16[actual_end: aligned_end] = 0
        ref_d[start: aligned_end] = a_bf16[start: aligned_end].float() @ b_bf16[i].float().t()
        start = aligned_end

    return a_bf16, b_bf16, grouped_layout, m_total, ref_d


def test_gather_m_grouped_gemm_contiguous() -> None:
    print('Testing FP8 m-grouped contiguous GEMM with gather_index (SM90 1D2D):')
    print('  3 paths per shape:')
    print('    [B] baseline — A is pre-permuted into grouped layout                 (no gather)')
    print('    [E] explicit — index_select(A_pool, g) + m_grouped_gemm              (current MoE practice)')
    print('    [K] kernel   — m_grouped_gemm(A_pool, ..., gather_index=g)           (this PR)')
    print()

    quant_config = QuantConfig()
    kernel_type = KernelType.Kernel1D2D
    use_ue8m0 = False
    disable_ue8m0_cast = not use_ue8m0
    recipe, recipe_a, recipe_b = quant_config.get_recipes()

    deep_gemm.set_mk_alignment_for_contiguous_layout(
        deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout())

    # Same coverage as `test_fp8_fp4.py:test_m_grouped_gemm_contiguous`.
    m_group_list = [(4, 8192), (8, 4096)]
    n_k_list = [(7168, 3072), (4096, 4096), (4096, 2048)]

    summary = []
    for use_psum_layout in (False, True):
        for num_groups, expected_m_per_group in m_group_list:
            for n, k in n_k_list:
                torch.manual_seed(0)
                a_bf16, b_bf16, grouped_layout, m_total, ref_d = \
                    _make_grouped_contiguous_inputs(
                        num_groups, expected_m_per_group, n, k, use_psum_layout)

                a_fp8_grouped = _fp8_a(a_bf16, quant_config, use_ue8m0)
                b_fp8 = _fp8_b(b_bf16, quant_config, kernel_type, use_ue8m0, grouped=True)

                # Build a_pool such that a_pool[gather_index[i]] == a_bf16[i].
                gather_perm = torch.randperm(m_total, device='cuda')
                gather_index = gather_perm.to(torch.int32).contiguous()
                idx = gather_perm.long()
                a_pool_bf16 = torch.empty_like(a_bf16)
                a_pool_bf16[idx] = a_bf16
                a_pool_fp8 = _fp8_a(a_pool_bf16, quant_config, use_ue8m0)

                d = torch.empty((m_total, n), device='cuda', dtype=torch.bfloat16)
                em_kw = int(expected_m_per_group * 1.2) if use_psum_layout else None

                def fn_baseline():
                    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                        a_fp8_grouped, b_fp8, d, grouped_layout,
                        disable_ue8m0_cast=disable_ue8m0_cast,
                        recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
                        use_psum_layout=use_psum_layout,
                        expected_m_for_psum_layout=em_kw)

                def fn_explicit():
                    a0 = a_pool_fp8[0].index_select(0, idx)
                    a1 = a_pool_fp8[1].index_select(0, idx)
                    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                        (a0, a1), b_fp8, d, grouped_layout,
                        disable_ue8m0_cast=disable_ue8m0_cast,
                        recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
                        use_psum_layout=use_psum_layout,
                        expected_m_for_psum_layout=em_kw)

                def fn_kernel():
                    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                        a_pool_fp8, b_fp8, d, grouped_layout,
                        disable_ue8m0_cast=disable_ue8m0_cast,
                        recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
                        use_psum_layout=use_psum_layout,
                        expected_m_for_psum_layout=em_kw,
                        gather_index=gather_index)

                _silent_warmup(fn_baseline, fn_explicit, fn_kernel)

                # ---- correctness ----
                fn_baseline(); d_baseline = d.clone()
                diff_b = calc_diff(d_baseline, ref_d)
                fn_explicit(); d_explicit = d.clone()
                diff_e = calc_diff(d_explicit, ref_d)
                fn_kernel(); diff_k = calc_diff(d, ref_d)
                diff_kvb = calc_diff(d, d_baseline)
                assert diff_b < quant_config.max_diff(), \
                    f'baseline diff vs ref fails: ng={num_groups}, m={m_total}, n={n}, k={k}, diff={diff_b:.5f}'
                assert diff_e < quant_config.max_diff(), \
                    f'explicit diff vs ref fails: ng={num_groups}, m={m_total}, n={n}, k={k}, diff={diff_e:.5f}'
                assert diff_k < quant_config.max_diff(), \
                    f'kernel diff vs ref fails: ng={num_groups}, m={m_total}, n={n}, k={k}, diff={diff_k:.5f}'
                assert diff_kvb < 1e-6, \
                    f'kernel diverges from baseline: ng={num_groups}, m={m_total}, n={n}, k={k}, diff={diff_kvb:.2e}'

                t_b = _bench_silent(fn_baseline)
                t_e = _bench_silent(fn_explicit)
                t_k = _bench_silent(fn_kernel)
                flops = 2 * m_total * n * k
                speedup = t_e / t_k

                print(f'  > ng={num_groups}, m={m_total:5}, n={n:5}, k={k:5}, psum={int(use_psum_layout)} | '
                      f'B={_us(t_b):>5}us E={_us(t_e):>5}us K={_us(t_k):>5}us  e2e={speedup:.2f}x')

                summary.append((
                    num_groups, m_total, n, k, int(use_psum_layout),
                    _us(t_b), _tflops(flops, t_b),
                    _us(t_e), _tflops(flops, t_e),
                    _us(t_k), _tflops(flops, t_k),
                    f'{speedup:.2f}x',
                    f'{diff_b:.0e}', f'{diff_k:.0e}',
                ))

    headers = ['ng', 'm', 'n', 'k', 'psum',
               'B us', 'B TFLOPS',
               'E us', 'E TFLOPS',
               'K us', 'K TFLOPS',
               'e2e (E/K)', 'diff_B', 'diff_K']
    _format_table(
        'Summary: m-grouped contiguous fp8 GEMM with gather_index',
        headers, summary)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Rank-overlap (per-rank ready-flag) — accuracy-only single-GPU smoke test.
#
# Validates the new `rank_flags` / `tile_rank` / `num_ranks` path on top of
# `gather_index`. Single GPU, so we **pre-set all flags to 1** before launch
# (i.e. simulating "all-gather already finished") to focus on:
#
#   1. flag spin → NamedBarrier → tile compute path is wired correctly,
#   2. `gather_index[i] = -1` pad rows produce 0 in the corresponding output rows,
#   3. the layout-transformed SFA path still aligns with the un-permuted A pool.
#
# The end-to-end overlap latency benefit must be measured in a real multi-rank
# setup (left as future work; see docs/sm90_fp8_gemm_1d2d_gather_index_rank_overlap.md).
# ---------------------------------------------------------------------------
def _build_rank_aware_layout(num_ranks: int,
                             tokens_per_rank: int,
                             pad_ratio: float,
                             block_m: int):
    """Build the (gather_index, tile_rank, M_logical) triple for a single
    rank-aware tile layout.

    Each rank contributes between `(1 - pad_ratio) * tokens_per_rank` and
    `tokens_per_rank` real tokens; the rest of that rank's logical M slot is
    padded (gather_index = -1). Each rank's slot is rounded up to a multiple
    of `block_m`, so every M-tile (BLOCK_M rows) belongs to exactly one rank.
    """
    import random
    gather_chunks = []
    tile_rank_list = []
    m_logical = 0
    for r in range(num_ranks):
        n_real = max(1, int(tokens_per_rank * (1 - random.random() * pad_ratio)))
        n_real = min(n_real, tokens_per_rank)
        n_pad = ((n_real + block_m - 1) // block_m) * block_m - n_real
        n_slot = n_real + n_pad

        real_perm = torch.randperm(tokens_per_rank, device='cuda')[:n_real]
        # source row is rank-r token index in A_pool: range [r*T, (r+1)*T)
        real_g = (r * tokens_per_rank + real_perm).to(torch.int32)
        pad_g = torch.full((n_pad,), -1, dtype=torch.int32, device='cuda')

        gather_chunks.append(real_g)
        gather_chunks.append(pad_g)
        n_tiles_this_rank = n_slot // block_m
        tile_rank_list.extend([r] * n_tiles_this_rank)
        m_logical += n_slot

    gather_index = torch.cat(gather_chunks).contiguous()
    tile_rank = torch.tensor(tile_rank_list, dtype=torch.int32, device='cuda')
    assert gather_index.numel() == m_logical
    assert tile_rank.numel() * block_m == m_logical
    return gather_index, tile_rank, m_logical


def test_gather_rank_overlap_smoke() -> None:
    print('Testing FP8 GEMM with gather_index + rank_flags (smoke / accuracy only):')
    print('  Flags pre-set to 1; verifies pad-row → 0 and overlap dispatch path.')
    print()

    quant_config = QuantConfig()
    kernel_type = KernelType.Kernel1D2D
    use_ue8m0 = False
    disable_ue8m0_cast = not use_ue8m0
    recipe, recipe_a, recipe_b = quant_config.get_recipes()

    block_m = 128  # SM90 1d2d default for bf16-output normal GEMM

    cases = [
        # (num_ranks, tokens_per_rank, n, k, pad_ratio)
        (1,  512,  4096, 7168, 0.0),   # rank=1 ⇒ degenerates to plain gather_index
        (4,  512,  4096, 7168, 0.0),   # multi-rank, no padding (real_per_rank = T)
        (4,  512,  4096, 7168, 0.3),   # multi-rank, mild pad
        (8,  256,  2112, 7168, 0.5),   # NVL8, heavy pad
        (8,  512,  7168, 2048, 0.2),
    ]

    summary = []
    for num_ranks, tokens_per_rank, n, k, pad_ratio in cases:
        torch.manual_seed(0)
        import random; random.seed(0)

        gather_index, tile_rank, m_logical = _build_rank_aware_layout(
            num_ranks, tokens_per_rank, pad_ratio, block_m)
        m_pool = num_ranks * tokens_per_rank

        a_pool_bf16 = torch.randn((m_pool, k), device='cuda', dtype=torch.bfloat16)
        b_bf16 = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
        a_pool_fp8 = _fp8_a(a_pool_bf16, quant_config, use_ue8m0)
        b_fp8 = _fp8_b(b_bf16, quant_config, kernel_type, use_ue8m0)

        # Reference: pad rows materialize as 0 in A.
        g_long = gather_index.long()
        is_pad = g_long < 0
        safe_idx = torch.where(is_pad, torch.zeros_like(g_long), g_long)
        a_logical_bf16 = a_pool_bf16[safe_idx].clone()
        a_logical_bf16[is_pad] = 0
        ref_d = (a_logical_bf16.float() @ b_bf16.float().t()).to(torch.bfloat16)

        # Pre-set all flags to 1 (single-GPU smoke).
        rank_flags = torch.ones(num_ranks, dtype=torch.int64, device='cuda')

        d = torch.empty((m_logical, n), device='cuda', dtype=torch.bfloat16)

        def fn_kernel():
            deep_gemm.fp8_gemm_nt(
                a_pool_fp8, b_fp8, d, recipe=recipe,
                recipe_a=recipe_a, recipe_b=recipe_b,
                disable_ue8m0_cast=disable_ue8m0_cast,
                gather_index=gather_index,
                rank_flags=rank_flags,
                tile_rank=tile_rank,
                num_ranks=num_ranks,
            )

        _silent_warmup(fn_kernel)
        fn_kernel()
        torch.cuda.synchronize()

        diff_k = calc_diff(d, ref_d)

        # Pad-row outputs must be exactly 0 (no fp8 quant error contributes).
        if is_pad.any():
            d_pad = d[is_pad]
            assert torch.all(d_pad == 0), \
                (f'pad rows are not zero: ranks={num_ranks}, T={tokens_per_rank}, '
                 f'n={n}, k={k}, n_nonzero_pad={int((d_pad != 0).sum().item())}')

        max_diff = quant_config.max_diff()
        assert diff_k < max_diff, \
            (f'kernel diff vs reference fails: ranks={num_ranks}, T={tokens_per_rank}, '
             f'n={n}, k={k}, pad={pad_ratio}, diff={diff_k:.5f} >= {max_diff}')

        n_pad = int(is_pad.sum().item())
        print(f'  > ranks={num_ranks}, T={tokens_per_rank}, m_logical={m_logical:5}, '
              f'n_pad={n_pad:5}, n={n:5}, k={k:5} | diff={diff_k:.0e}  OK')

        summary.append((num_ranks, tokens_per_rank, m_logical, n_pad,
                        n, k, f'{diff_k:.0e}'))

    _format_table(
        'Summary: gather_index + rank_flags overlap (single-GPU smoke; flags pre-set to 1)',
        ['ranks', 'T/rank', 'm_logical', 'n_pad', 'n', 'k', 'diff_K'],
        summary)


_AVAILABLE_TESTS = {
    'normal':         test_gather_gemm,
    'grouped':        test_gather_m_grouped_gemm_contiguous,
    'rank_overlap':   test_gather_rank_overlap_smoke,
}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Bench the SM90 FP8 1D2D `gather_index` feature on Normal '
                    'and / or M-Grouped Contiguous GEMM.')
    parser.add_argument(
        '--tests', '-t',
        nargs='+',
        choices=list(_AVAILABLE_TESTS.keys()) + ['all'],
        default=['all'],
        help=("which tests to run. Repeatable / space-separated. "
              "`normal`: GemmType::Normal (test_gather_gemm). "
              "`grouped`: MGroupedContiguous (test_gather_m_grouped_gemm_contiguous). "
              "`all`: both (default)."))
    return parser.parse_args(argv)


def main(argv=None):
    if get_arch_major() != 9:
        print('gather_index is only supported on SM90; this run is not SM90.')
        return

    args = _parse_args(argv)
    selected = list(_AVAILABLE_TESTS.keys()) if 'all' in args.tests else args.tests
    selected = [name for name in _AVAILABLE_TESTS.keys() if name in selected]

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')
    print(f'Selected tests: {", ".join(selected)}\n')

    for name in selected:
        _AVAILABLE_TESTS[name]()


if __name__ == '__main__':
    torch.manual_seed(0)
    main()
