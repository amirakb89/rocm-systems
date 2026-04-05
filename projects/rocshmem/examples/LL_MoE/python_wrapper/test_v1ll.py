"""
Test script for the mori V1LL dispatch + combine Python wrapper.

Run with:
    torchrun --nproc-per-node=8 test_v1ll.py [--num-tokens 128] [--hidden 7168] ...
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Union

import torch
import torch.distributed as dist
import numpy as np

import mori_v1ll_cpp


def init_shmem(rank: int, world_size: int, group: dist.ProcessGroup):
    """Bootstrap rocSHMEM using torch.distributed to broadcast the unique ID."""
    if rank == 0:
        uid_vec = mori_v1ll_cpp.shmem_get_uid()
    else:
        uid_vec = [0] * 128
    uid_tensor = torch.tensor(uid_vec, dtype=torch.uint8, device="cuda")
    dist.broadcast(uid_tensor, src=0, group=group)
    uid_list = uid_tensor.cpu().tolist()
    mori_v1ll_cpp.shmem_init(uid_list, rank, world_size)


def test_dispatch_correctness(buffer, x, topk_idx, topk_weights, rank, world_size,
                               num_tokens, hidden, num_expert_per_token, num_expert_per_rank,
                               max_tokens_per_rank):
    """Verify dispatch routes tokens correctly."""
    recv_x, recv_indices, recv_weights, num_recv_t, src_tok_map = \
        buffer.dispatch(x, topk_idx, topk_weights)

    num_recv = num_recv_t.item()
    expected_recv = num_tokens * world_size
    assert num_recv == expected_recv, \
        f"[rank {rank}] recv count mismatch: got {num_recv}, expected {expected_recv}"

    hidden_mismatches = 0
    index_mismatches = 0
    weight_mismatches = 0

    src_tok_map_cpu = src_tok_map.cpu()
    recv_x_cpu = recv_x.cpu().float()
    recv_idx_cpu = recv_indices.cpu()
    recv_wt_cpu = recv_weights.cpu()

    config_max_send = max_tokens_per_rank * world_size

    for t in range(num_recv):
        flat_src = src_tok_map_cpu[t].item()
        src_pe   = flat_src // config_max_send
        src_tok  = flat_src % config_max_send

        if src_pe < 0 or src_pe >= world_size or src_tok < 0 or src_tok >= num_tokens:
            hidden_mismatches += 1
            if hidden_mismatches <= 3:
                print(f"[rank {rank}] token {t}: invalid src_pe={src_pe} src_tok={src_tok}")
            continue

        for h in range(min(hidden, 64)):
            expected = 1.0 + 0.001 * src_tok + 0.0001 * (h % 64)
            actual = recv_x_cpu[t, h].item()
            if abs(actual - expected) > 0.02:
                hidden_mismatches += 1
                break

        for e in range(num_expert_per_token):
            expected_idx = ((src_pe + e) % world_size) * num_expert_per_rank
            actual_idx = recv_idx_cpu[t, e].item()
            if actual_idx != expected_idx:
                index_mismatches += 1
                break

        for e in range(num_expert_per_token):
            w = recv_wt_cpu[t, e].item()
            if abs(w - 1.0) > 1e-4:
                weight_mismatches += 1
                break

    total_errors = hidden_mismatches + index_mismatches + weight_mismatches
    if total_errors == 0:
        print(f"[rank {rank}] DISPATCH PASSED — verified {num_recv}/{num_recv} tokens "
              f"(hidden, indices, weights)", flush=True)
    else:
        print(f"[rank {rank}] DISPATCH FAILED — {hidden_mismatches} hidden, "
              f"{index_mismatches} index, {weight_mismatches} weight errors", flush=True)
    return total_errors == 0


def test_combine_correctness(buffer, recv_x, topk_weights_recv, x_orig, topk_weights_orig,
                              rank, world_size, num_tokens, hidden, num_expert_per_token):
    """Verify combine accumulates tokens back correctly."""
    combined = buffer.combine(recv_x, topk_weights_recv)

    combined_cpu = combined.cpu().float()
    x_cpu = x_orig.cpu().float()
    weights_cpu = topk_weights_orig.cpu().float()

    weight_sum = weights_cpu.sum(dim=1, keepdim=True)
    expected = x_cpu * weight_sum

    diff = (combined_cpu - expected).abs().max().item()
    bf16_tol = 0.05
    if diff < bf16_tol:
        print(f"[rank {rank}] COMBINE PASSED — max diff {diff:.6f}", flush=True)
        return True
    else:
        print(f"[rank {rank}] COMBINE FAILED — max diff {diff:.6f} (tol={bf16_tol})", flush=True)
        return False


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')
        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()
        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)
        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)
        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)
        self.outnull_file.close()
        self.errnull_file.close()


class empty_suppress:
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass


def bench(fn, num_warmups: int = 50, num_tests: int = 50, post_fn=None):
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')
    for _ in range(num_warmups):
        fn()
    cache.zero_()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        start_events[i].record()
        fn()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
    torch.cuda.synchronize()
    times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])[1:]
    return np.average(times), np.min(times), np.max(times)


def bench_kineto(fn,
                 kernel_names: Union[str, tuple],
                 num_tests: int = 30,
                 suppress_kineto_output: bool = False,
                 trace_path: Optional[str] = None,
                 barrier_comm_profiling: bool = False,
                 num_kernels_per_period: int = 1):
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=schedule,
        ) as prof:
            for _ in range(2):
                if barrier_comm_profiling:
                    lhs = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
                    rhs = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
                    lhs @ rhs
                    dist.all_reduce(torch.ones(1, dtype=torch.float, device='cuda'))
                for _ in range(num_tests):
                    fn()
                torch.cuda.synchronize()
                prof.step()

    assert isinstance(kernel_names, (str, tuple))
    is_tuple = isinstance(kernel_names, tuple)
    prof_lines = prof.key_averages().table(
        sort_by='cuda_time_total', max_name_column_width=100).split('\n')
    kernel_names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names
    assert all(isinstance(name, str) for name in kernel_names)

    if trace_path is not None:
        prof.export_chrome_trace(trace_path)

    units = {'ms': 1e3, 'us': 1e6}
    kernel_durations = []
    for name in kernel_names:
        found = False
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                for unit, scale in units.items():
                    if unit in time_str:
                        kernel_durations.append(float(time_str.replace(unit, '')) / scale)
                        found = True
                        break
                break
        if not found:
            kernel_durations.append(float('nan'))

    if num_kernels_per_period > 1:
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
            prof.export_chrome_trace(tmp.name)
            profile_data = json.loads(Path(tmp.name).read_text())
        for i, kernel_name in enumerate(kernel_names):
            events = [e for e in profile_data['traceEvents'] if f'::{kernel_name}' in e.get('name', '')]
            events = sorted(events, key=lambda e: e['ts'])
            durations = [e['dur'] / 1e6 for e in events]
            if len(durations) > 0 and len(durations) % num_kernels_per_period == 0:
                num_kernel_patterns = len(durations) // num_kernels_per_period
                kernel_durations[i] = [
                    sum(durations[j::num_kernels_per_period]) / num_kernel_patterns
                    for j in range(num_kernels_per_period)
                ]

    return kernel_durations if is_tuple else kernel_durations[0]


def run_once_bench(buffer, x, topk_idx, topk_weights,
                   hidden_dim, element_size, max_tokens_per_rank,
                   num_experts_per_token, graph_replay_iters=10):
    """Single benchmark iteration matching mori's run_once_bench methodology.

    Times dispatch and combine separately with CUDA events, computes BW
    using mori's formula: total_recv_tokens * hidden * elem_size / duration.
    """
    # Run one real dispatch to get total_recv_num_token
    buffer.reset_counters()
    recv_x, _, recv_wt, num_recv_t, _ = buffer.dispatch(x, topk_idx, topk_weights)
    total_recv_num_token = num_recv_t.item()
    buffer.combine(recv_x, recv_wt)
    torch.cuda.synchronize()

    # Separate dispatch / combine timing
    round_start = [torch.cuda.Event(enable_timing=True) for _ in range(graph_replay_iters)]
    mid_events  = [torch.cuda.Event(enable_timing=True) for _ in range(graph_replay_iters)]
    round_end   = [torch.cuda.Event(enable_timing=True) for _ in range(graph_replay_iters)]
    e2e_start   = [torch.cuda.Event(enable_timing=True) for _ in range(graph_replay_iters)]
    e2e_end     = [torch.cuda.Event(enable_timing=True) for _ in range(graph_replay_iters)]

    for i in range(graph_replay_iters):
        buffer.reset_counters()
        round_start[i].record()
        recv_x, _, recv_wt, _, _ = buffer.dispatch(x, topk_idx, topk_weights)
        mid_events[i].record()
        buffer.combine(recv_x, recv_wt)
        round_end[i].record()

    for i in range(graph_replay_iters):
        buffer.reset_counters()
        e2e_start[i].record()
        recv_x, _, recv_wt, _, _ = buffer.dispatch(x, topk_idx, topk_weights)
        buffer.combine(recv_x, recv_wt)
        e2e_end[i].record()

    torch.cuda.synchronize()

    disp_duration = sum(
        s.elapsed_time(e) for s, e in zip(round_start, mid_events)
    ) / graph_replay_iters
    comb_duration = sum(
        s.elapsed_time(e) for s, e in zip(mid_events, round_end)
    ) / graph_replay_iters
    e2e_duration = sum(
        s.elapsed_time(e) for s, e in zip(e2e_start, e2e_end)
    ) / graph_replay_iters

    disp_total_bytes = total_recv_num_token * hidden_dim * element_size
    comb_total_bytes = total_recv_num_token * hidden_dim * element_size
    ll_mode_scale = (
        max_tokens_per_rank * num_experts_per_token
        / (total_recv_num_token + 0.01)
    )
    disp_bandwidth = disp_total_bytes / (1000**3) / (disp_duration / 1e3)
    comb_bandwidth = comb_total_bytes / (1000**3) / (comb_duration / 1e3)

    return (
        disp_duration, comb_duration, e2e_duration,
        disp_bandwidth, comb_bandwidth,
        disp_total_bytes, comb_total_bytes,
        ll_mode_scale,
    )


def main():
    parser = argparse.ArgumentParser(description="Test V1LL dispatch/combine with Python+torch")
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts-per-rank", type=int, default=36)
    parser.add_argument("--max-tokens-per-rank", type=int, default=128)
    parser.add_argument("--block-num", type=int, default=32)
    parser.add_argument("--rdma-block-num", type=int, default=2)
    parser.add_argument("--warp-per-block", type=int, default=4)
    parser.add_argument("--num-iters", type=int, default=10,
                        help="Number of benchmark iterations (each all-gathered across ranks)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warmup iterations before benchmarking")
    parser.add_argument("--graph-replay-iters", type=int, default=10,
                        help="Number of event-timed replays per benchmark iteration")
    parser.add_argument("--num-kineto-tests", type=int, default=30,
                        help="Number of iterations per kineto profiling phase")
    parser.add_argument("--trace-dir", type=str, default=None,
                        help="Directory to save chrome trace JSON files (e.g. /tmp/traces)")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(torch.cuda.device_count() > 1 and rank % torch.cuda.device_count() or rank)
    torch.cuda.set_device(local_rank)

    group = dist.new_group(list(range(world_size)))

    print(f"[rank {rank}] Initializing rocSHMEM ...", flush=True)
    init_shmem(rank, world_size, group)

    print(f"[rank {rank}] Creating V1LLBuffer (tokens={args.num_tokens} hidden={args.hidden} "
          f"topk={args.num_topk} experts/rank={args.num_experts_per_rank}) ...", flush=True)

    buffer = mori_v1ll_cpp.V1LLBuffer(
        rank=rank,
        world_size=world_size,
        hidden_dim=args.hidden,
        num_expert_per_token=args.num_topk,
        num_expert_per_rank=args.num_experts_per_rank,
        max_tokens_per_rank=args.max_tokens_per_rank,
        warp_per_block=args.warp_per_block,
        block_num=args.block_num,
        rdma_block_num=args.rdma_block_num,
    )

    x = torch.empty(args.num_tokens, args.hidden, dtype=torch.bfloat16, device="cuda")
    for t in range(args.num_tokens):
        x[t, :] = 1.0 + 0.001 * t + 0.0001 * (torch.arange(args.hidden, device="cuda") % 64).to(torch.bfloat16)

    topk_idx = torch.empty(args.num_tokens, args.num_topk, dtype=torch.int32, device="cuda")
    for t in range(args.num_tokens):
        for e in range(args.num_topk):
            dest_pe = (rank + e) % world_size
            topk_idx[t, e] = dest_pe * args.num_experts_per_rank

    topk_weights = torch.ones(args.num_tokens, args.num_topk, dtype=torch.float32, device="cuda")

    mori_v1ll_cpp.shmem_barrier()

    # --- Correctness: dispatch ---
    dispatch_ok = test_dispatch_correctness(
        buffer, x, topk_idx, topk_weights,
        rank, world_size, args.num_tokens, args.hidden,
        args.num_topk, args.num_experts_per_rank,
        args.max_tokens_per_rank,
    )

    # --- Correctness: combine ---
    buffer.reset_counters()
    recv_x, recv_indices, recv_weights, num_recv_t, src_tok_map = \
        buffer.dispatch(x, topk_idx, topk_weights)
    combine_ok = test_combine_correctness(
        buffer, recv_x, recv_weights, x, topk_weights,
        rank, world_size, args.num_tokens, args.hidden, args.num_topk,
    )

    # ================================================================
    # Benchmark (mori-style): CUDA events, per-rank all-gather
    # ================================================================
    hidden_dim = args.hidden
    element_size = x.element_size()
    max_tokens_per_rank = args.max_tokens_per_rank
    num_experts_per_token = args.num_topk
    graph_replay_iters = args.graph_replay_iters
    iters = args.num_iters
    warmup = args.warmup

    mori_v1ll_cpp.shmem_barrier()

    # Warmup
    for _ in range(warmup):
        buffer.reset_counters()
        rx, _, rw, _, _ = buffer.dispatch(x, topk_idx, topk_weights)
        buffer.combine(rx, rw)
    torch.cuda.synchronize()
    mori_v1ll_cpp.shmem_barrier()

    disp_duration_us_list = []
    disp_bandwidth_GB_list = []
    comb_duration_us_list = []
    comb_bandwidth_GB_list = []
    e2e_duration_us_list = []
    disp_avg_bytes_MB_list = []
    comb_avg_bytes_MB_list = []

    for i in range(iters):
        mori_v1ll_cpp.shmem_barrier()
        (
            disp_dur, comb_dur, e2e_dur,
            disp_bw, comb_bw,
            disp_total_bytes, comb_total_bytes,
            ll_mode_scale,
        ) = run_once_bench(
            buffer, x, topk_idx, topk_weights,
            hidden_dim, element_size, max_tokens_per_rank,
            num_experts_per_token, graph_replay_iters=graph_replay_iters,
        )

        dev = torch.device("cuda")
        disp_dur_list = [torch.zeros(1, device=dev) for _ in range(world_size)]
        disp_bw_list = [torch.zeros(1, device=dev) for _ in range(world_size)]
        comb_dur_list = [torch.zeros(1, device=dev) for _ in range(world_size)]
        comb_bw_list = [torch.zeros(1, device=dev) for _ in range(world_size)]
        e2e_dur_list = [torch.zeros(1, device=dev) for _ in range(world_size)]
        disp_bytes_list = [torch.zeros(1, device=dev) for _ in range(world_size)]
        comb_bytes_list = [torch.zeros(1, device=dev) for _ in range(world_size)]

        dist.all_gather(disp_dur_list, torch.tensor([disp_dur * 1000], device=dev))
        dist.all_gather(disp_bw_list, torch.tensor([disp_bw], device=dev))
        dist.all_gather(comb_dur_list, torch.tensor([comb_dur * 1000], device=dev))
        dist.all_gather(comb_bw_list, torch.tensor([comb_bw], device=dev))
        dist.all_gather(e2e_dur_list, torch.tensor([e2e_dur * 1000], device=dev))
        dist.all_gather(disp_bytes_list, torch.tensor([disp_total_bytes / (1024**2)], device=dev))
        dist.all_gather(comb_bytes_list, torch.tensor([comb_total_bytes / (1024**2)], device=dev))

        disp_duration_us_list.append([int(t.item()) for t in disp_dur_list])
        disp_bandwidth_GB_list.append([int(t.item()) for t in disp_bw_list])
        comb_duration_us_list.append([int(t.item()) for t in comb_dur_list])
        comb_bandwidth_GB_list.append([int(t.item()) for t in comb_bw_list])
        e2e_duration_us_list.append([int(t.item()) for t in e2e_dur_list])
        disp_avg_bytes_MB_list.append(int(torch.stack(disp_bytes_list).mean().item()))
        comb_avg_bytes_MB_list.append(int(torch.stack(comb_bytes_list).mean().item()))

    max_disp_algo_bw = 0
    max_comb_algo_bw = 0
    min_disp_latency_us = float("inf")
    min_comb_latency_us = float("inf")
    for i in range(iters):
        disp_algo_bw = sum(disp_bandwidth_GB_list[i]) / world_size
        comb_algo_bw = sum(comb_bandwidth_GB_list[i]) / world_size
        max_disp_algo_bw = max(max_disp_algo_bw, disp_algo_bw)
        max_comb_algo_bw = max(max_comb_algo_bw, comb_algo_bw)
        disp_max_lat = max(disp_duration_us_list[i])
        comb_max_lat = max(comb_duration_us_list[i])
        min_disp_latency_us = min(min_disp_latency_us, disp_max_lat)
        min_comb_latency_us = min(min_comb_latency_us, comb_max_lat)

    if rank == 0:
        print("\nDispatch result:", flush=True)
        for i, duration_us in enumerate(disp_duration_us_list):
            algo_bw = sum(disp_bandwidth_GB_list[i]) / world_size
            lat = max(duration_us)
            print(
                f"Round {i} duration(us) {duration_us} "
                f"bandwidth(GB/s) {disp_bandwidth_GB_list[i]} "
                f"avg bytes(MB) {disp_avg_bytes_MB_list[i]} "
                f"lat {lat} bw {algo_bw} / {algo_bw * ll_mode_scale:.2f}",
                flush=True,
            )

        print("\nCombine result:", flush=True)
        for i, duration_us in enumerate(comb_duration_us_list):
            algo_bw = sum(comb_bandwidth_GB_list[i]) / world_size
            lat = max(duration_us)
            print(
                f"Round {i} duration(us) {duration_us} "
                f"bandwidth(GB/s) {comb_bandwidth_GB_list[i]} "
                f"avg bytes(MB) {comb_avg_bytes_MB_list[i]} "
                f"lat {lat} bw {algo_bw} / {algo_bw * ll_mode_scale:.2f}",
                flush=True,
            )

        print("\nEnd-to-end result:", flush=True)
        for i, duration_us in enumerate(e2e_duration_us_list):
            print(f"Round {i} e2e(us) {duration_us}", flush=True)

        print(f"\n{'=' * 60}", flush=True)
        print("Performance Summary:", flush=True)
        print(f"{'=' * 60}", flush=True)
        print(
            f"Best Dispatch (bf16): {max_disp_algo_bw:.2f} GB/s  "
            f"(min bottleneck lat {min_disp_latency_us} us)",
            flush=True,
        )
        print(
            f"Best Combine  (bf16): {max_comb_algo_bw:.2f} GB/s  "
            f"(min bottleneck lat {min_comb_latency_us} us)",
            flush=True,
        )
        print(
            f"Scaled (ll_mode_scale={ll_mode_scale:.2f}): "
            f"Dispatch {max_disp_algo_bw * ll_mode_scale:.2f} GB/s, "
            f"Combine {max_comb_algo_bw * ll_mode_scale:.2f} GB/s",
            flush=True,
        )
        print(f"{'=' * 60}", flush=True)

    # ================================================================
    # Kineto profiling: per-kernel breakdown
    # ================================================================
    mori_v1ll_cpp.shmem_barrier()
    if rank == 0:
        print("\n===== Kineto Profiling =====", flush=True)

    dispatch_kernel_names = (
        "EpDispatchCopyToStaging",
        "EpDispatchInterNodeV1KernelLowLatency",
    )
    combine_kernel_names = (
        "EpCombineSync",
        "EpCombineSyncBarrier",
        "EpCombineInterNodeV1KernelLowLatency",
        "EpCombineAll",
    )
    all_kernel_names = dispatch_kernel_names + combine_kernel_names

    num_kineto_tests = args.num_kineto_tests

    trace_dir = args.trace_dir
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)

    def run_dispatch_then_combine():
        buffer.reset_counters()
        rx, _, rw, _, _ = buffer.dispatch(x, topk_idx, topk_weights)
        buffer.combine(rx, rw)

    # Warmup kineto path
    for _ in range(10):
        run_dispatch_then_combine()
    torch.cuda.synchronize()

    full_trace = os.path.join(trace_dir, f"full_pipeline_rank{rank}.json") if trace_dir else None
    full_durations = bench_kineto(
        run_dispatch_then_combine,
        kernel_names=all_kernel_names,
        num_tests=num_kineto_tests,
        suppress_kineto_output=True,
        trace_path=full_trace,
        barrier_comm_profiling=True,
    )

    if rank == 0:
        # Use the last total_recv_num_token from the benchmark rounds
        buffer.reset_counters()
        _, _, _, num_recv_t, _ = buffer.dispatch(x, topk_idx, topk_weights)
        total_recv = num_recv_t.item()
        disp_bytes = total_recv * hidden_dim * element_size
        comb_bytes = total_recv * hidden_dim * element_size

        kineto_disp_dur_s = sum(
            d for d in full_durations[:2] if isinstance(d, float) and not np.isnan(d)
        )
        kineto_comb_dur_s = sum(
            d for d in full_durations[2:] if isinstance(d, float) and not np.isnan(d)
        )
        kineto_disp_bw = disp_bytes / (1000**3) / kineto_disp_dur_s if kineto_disp_dur_s > 0 else 0
        kineto_comb_bw = comb_bytes / (1000**3) / kineto_comb_dur_s if kineto_comb_dur_s > 0 else 0

        print(f"\n  Kernel timings (avg over {num_kineto_tests} calls, recv_tokens={total_recv}):",
              flush=True)
        for name, dur in zip(all_kernel_names, full_durations):
            if isinstance(dur, float):
                print(f"    {name}: {dur*1e6:.1f} us", flush=True)
            else:
                print(f"    {name}: {[f'{d*1e6:.1f} us' for d in dur]}", flush=True)
        print(f"  ---", flush=True)
        print(f"  Total dispatch kernels: {kineto_disp_dur_s*1e6:.1f} us  "
              f"BW={kineto_disp_bw:.2f} GB/s", flush=True)
        print(f"  Total combine kernels:  {kineto_comb_dur_s*1e6:.1f} us  "
              f"BW={kineto_comb_bw:.2f} GB/s", flush=True)
        total_s = kineto_disp_dur_s + kineto_comb_dur_s
        print(f"  Total pipeline kernels: {total_s*1e6:.1f} us", flush=True)
        if trace_dir:
            print(f"\n  Chrome traces saved to: {trace_dir}/", flush=True)

    dist.barrier(group=group)
    dist.destroy_process_group()
    try:
        mori_v1ll_cpp.shmem_finalize()
    except Exception:
        pass


if __name__ == "__main__":
    main()
