"""Bench single-node symmetric-memory all-gather overlapped with grouped GEMM.

Run example:
  python tests/bench_single_node_allgather_overlap.py --num-local-ranks 8

The timed all-gather payload is the FP8 activation tensor. SFA is prepared before
timing because the current Python/C++ GEMM API transforms SFA before launching
the rank-flag-aware GEMM kernel, so the GEMM-side flag wait cannot guard SFA
communication yet.
"""

import argparse
import os
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import deep_gemm
from deep_gemm.testing import get_arch_major
from deep_gemm.utils.dist import dist_print, init_dist

sys.path.insert(0, os.path.dirname(__file__))
from generators import (  # noqa: E402
    MajorTypeAB,
    QuantConfig,
    cast_fp8_fp4_with_major,
    grouped_cast_fp8_fp4_with_major,
)


def _generate_routing_topk(num_tokens: int, top_k: int, num_experts: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    return torch.randint(0, num_experts, (num_tokens, top_k),
                         dtype=torch.int32, device='cuda', generator=gen)


def _bench_ms(fn, group: dist.ProcessGroup, *, warmups: int, iters: int) -> float:
    times = []
    for _ in range(warmups):
        fn()
        torch.cuda.synchronize()
        dist.barrier(group=group)

    for _ in range(iters):
        dist.barrier(group=group)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def _max_across_ranks(value: float, group: dist.ProcessGroup) -> float:
    t = torch.tensor([value], dtype=torch.float64, device='cuda')
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return float(t.item())


def _worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    if get_arch_major() != 9:
        dist_print('SM90 is required for the current rank-flag grouped GEMM path.', once_in_node=True)
        return

    torch.manual_seed(0x1234 + rank)
    tokens_per_rank = args.tokens_per_rank
    total_tokens = num_ranks * tokens_per_rank
    hidden = args.hidden
    num_experts = args.num_experts
    top_k = args.top_k
    block_m = args.block_m

    quant_config = QuantConfig()
    recipe, recipe_a, recipe_b = quant_config.get_recipes()

    # Deterministic global activations let every rank precompute SFA outside the
    # timed region while the FP8 A payload itself is still gathered over NVLink.
    torch.manual_seed(0x2026)
    a_global_bf16 = torch.randn((total_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    a_global_fp8, sfa_global = cast_fp8_fp4_with_major(
        a_global_bf16, MajorTypeAB.KMajor, quant_config.gran_k_a,
        quant_config.is_fp4_a, use_ue8m0=False)
    a_local = a_global_fp8[rank * tokens_per_rank:(rank + 1) * tokens_per_rank].contiguous()

    # Symmetric output buffer for the all-gathered A pool.
    a_symm = symm_mem.empty((num_ranks, tokens_per_rank, hidden),
                            dtype=a_local.dtype, device='cuda')
    a_handle = symm_mem.rendezvous(a_symm, group=group)
    a_pool = a_symm.view(total_tokens, hidden)
    a_buffer_ptrs = [int(p) for p in a_handle.buffer_ptrs]

    # Grouped GEMM metadata and weights.
    routing_topk = _generate_routing_topk(total_tokens, top_k, num_experts, seed=0xBEEF)
    gather_index, tile_rank, grouped_layout, m_logical_t = \
        deep_gemm.build_gather_layout_for_rank_overlap(
            routing_topk, rank, num_ranks, tokens_per_rank, num_experts, block_m)
    m_logical = int(m_logical_t.item())
    num_m_tiles = (m_logical + block_m - 1) // block_m

    torch.manual_seed(0x5678 + rank)
    b_bf16 = torch.randn((num_experts, args.n, hidden), dtype=torch.bfloat16, device='cuda')
    b_fp8 = grouped_cast_fp8_fp4_with_major(
        b_bf16, MajorTypeAB.KMajor, quant_config.gran_k_b,
        quant_config.is_fp4_b, use_ue8m0=False, use_block_cast_for_fp8=True)

    d = torch.empty((m_logical, args.n), dtype=torch.bfloat16, device='cuda')
    rank_flags = torch.empty((num_ranks,), dtype=torch.int64, device='cuda')

    comm_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    zero_done = torch.cuda.Event()

    def launch_gemm() -> None:
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (a_pool, sfa_global), b_fp8, d, grouped_layout[:m_logical],
            recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
            disable_ue8m0_cast=True,
            gather_index=gather_index[:m_logical],
            rank_flags=rank_flags,
            tile_rank=tile_rank[:num_m_tiles],
            num_ranks=num_ranks,
        )

    def run_allgather_only() -> None:
        with torch.cuda.stream(comm_stream):
            rank_flags.zero_()
            deep_gemm.single_node_allgather_copy_local([a_local], [a_symm], rank, rank_flags)
            a_handle.barrier()
            deep_gemm.single_node_allgather_pull([a_symm], [a_buffer_ptrs], rank, num_ranks, rank_flags)
        torch.cuda.current_stream().wait_stream(comm_stream)

    def run_gemm_only() -> None:
        with torch.cuda.stream(compute_stream):
            rank_flags.fill_(1)
            launch_gemm()
        torch.cuda.current_stream().wait_stream(compute_stream)

    def run_serial() -> None:
        with torch.cuda.stream(comm_stream):
            rank_flags.zero_()
            deep_gemm.single_node_allgather_copy_local([a_local], [a_symm], rank, rank_flags)
            a_handle.barrier()
            deep_gemm.single_node_allgather_pull([a_symm], [a_buffer_ptrs], rank, num_ranks, rank_flags)
        with torch.cuda.stream(compute_stream):
            compute_stream.wait_stream(comm_stream)
            launch_gemm()
        torch.cuda.current_stream().wait_stream(comm_stream)
        torch.cuda.current_stream().wait_stream(compute_stream)

    def run_overlap() -> None:
        with torch.cuda.stream(comm_stream):
            rank_flags.zero_()
            zero_done.record()
            deep_gemm.single_node_allgather_copy_local([a_local], [a_symm], rank, rank_flags)
        with torch.cuda.stream(compute_stream):
            compute_stream.wait_event(zero_done)
            launch_gemm()
        with torch.cuda.stream(comm_stream):
            a_handle.barrier()
            deep_gemm.single_node_allgather_pull([a_symm], [a_buffer_ptrs], rank, num_ranks, rank_flags)
        torch.cuda.current_stream().wait_stream(comm_stream)
        torch.cuda.current_stream().wait_stream(compute_stream)

    # Compile/warm the JIT paths, fill A once, and optionally validate all-gather.
    run_allgather_only()
    torch.cuda.synchronize()
    if args.check:
        torch.testing.assert_close(a_pool, a_global_fp8, rtol=0, atol=0)

    comm_ms = _max_across_ranks(_bench_ms(run_allgather_only, group, warmups=args.warmups, iters=args.iters), group)
    gemm_ms = _max_across_ranks(_bench_ms(run_gemm_only, group, warmups=args.warmups, iters=args.iters), group)
    serial_ms = _max_across_ranks(_bench_ms(run_serial, group, warmups=args.warmups, iters=args.iters), group)
    overlap_ms = _max_across_ranks(_bench_ms(run_overlap, group, warmups=args.warmups, iters=args.iters), group)

    if rank == 0:
        payload_mb = a_local.numel() * a_local.element_size() / 1e6
        print('Single-node symmetric-memory all-gather overlap bench:', flush=True)
        print(f'  ranks={num_ranks}, tokens/rank={tokens_per_rank}, hidden={hidden}, '
              f'payload/rank={payload_mb:.1f} MB', flush=True)
        print(f'  experts={num_experts}, top_k={top_k}, m_logical={m_logical}, n={args.n}', flush=True)
        print(f'  allgather only : {comm_ms * 1e3:8.2f} us', flush=True)
        print(f'  grouped GEMM   : {gemm_ms * 1e3:8.2f} us', flush=True)
        print(f'  serial         : {serial_ms * 1e3:8.2f} us', flush=True)
        print(f'  overlap        : {overlap_ms * 1e3:8.2f} us', flush=True)
        if serial_ms > 0:
            print(f'  speedup        : {serial_ms / overlap_ms:8.3f}x', flush=True)

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-local-ranks', type=int, default=None)
    parser.add_argument('--tokens-per-rank', type=int, default=1024)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--n', type=int, default=4096)
    parser.add_argument('--num-experts', type=int, default=8)
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--block-m', type=int, default=128)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()

    num_local_ranks = args.num_local_ranks or torch.cuda.device_count()
    assert num_local_ranks > 0
    torch.multiprocessing.spawn(_worker, args=(num_local_ranks, args), nprocs=num_local_ranks)


if __name__ == '__main__':
    main()
