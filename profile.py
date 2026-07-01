"""
Profile cuTile vs Triton vs PyTorch for the 3-layer fused MLP megakernel.

out = softplus(softplus(x @ W1) @ W2) @ W3

Benchmarks multiple batch sizes, reports timings and speedups.
After autotune selects the best config, all three are benchmarked inside
CUDA graphs to eliminate Python dispatch / launch overhead.

Usage:
    python profile.py
    python profile.py --sizes 256,512,1024
    python profile.py --warmup 50 --iters 100
    MLP_FAST_AUTOTUNE=1 python profile.py --sizes 256
"""
import argparse
import time
import torch
import torch.nn.functional as F

# --- cuTile ---
from cutile_mlp import fused_mlp_fwd_cutile

# --- Triton ---
from kernel import _fused_mlp_fwd as fused_mlp_fwd_triton


def benchmark_fn(fn, args, warmup=50, iters=100):
    """Benchmark a function using CUDA events (includes launch overhead)."""
    for _ in range(warmup):
        out = fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        out = fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    return median, mean, times


def benchmark_graph(fn, args, warmup=50, iters=100):
    """Benchmark using CUDA graph replay — eliminates Python dispatch overhead.

    After warmup (which triggers autotune/compilation), the kernel launch is
    captured into a CUDA graph. Replays measure pure GPU execution time.
    """
    # Warmup (triggers compilation/autotune)
    for _ in range(warmup):
        out = fn(*args)
    torch.cuda.synchronize()

    # Capture into a CUDA graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn(*args)

    # Time graph replays
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    return median, mean, times


def correctness_check(x, w1, w2, w3):
    """Verify both implementations match PyTorch reference."""
    h1 = F.softplus(x @ w1)
    h2 = F.softplus(h1 @ w2)
    ref = h2 @ w3

    out_ct = fused_mlp_fwd_cutile(x, w1, w2, w3)
    torch.cuda.synchronize()
    ct_diff = (out_ct - ref).abs().max().item()
    ct_pass = torch.allclose(out_ct, ref, atol=1e-2, rtol=1e-2)

    out_tr = fused_mlp_fwd_triton(x, w1, w2, w3)
    torch.cuda.synchronize()
    tr_diff = (out_tr - ref).abs().max().item()
    tr_pass = torch.allclose(out_tr, ref, atol=1e-2, rtol=1e-2)

    return ct_pass, ct_diff, tr_pass, tr_diff


def pytorch_fwd(x, w1, w2, w3):
    """Plain PyTorch reference: 3 separate matmuls + softplus."""
    h1 = F.softplus(x @ w1)
    h2 = F.softplus(h1 @ w2)
    return h2 @ w3


def run_benchmark(sizes, D, H, OUT_F, warmup, iters):
    """Run benchmarks for all sizes and return results."""
    print(f"{'='*80}")
    print(f"3-Layer Fused MLP Megakernel: cuTile vs Triton vs PyTorch")
    print(f"Architecture: out = softplus(softplus(x @ W1) @ W2) @ W3")
    print(f"D={D}, H={H}, OUT={OUT_F}, FP16, FP32 accumulation")
    print(f"Warmup={warmup}, Iters={iters}")
    print(f"GPU: {torch.cuda.get_device_properties(0).name}")
    print(f"SMs: {torch.cuda.get_device_properties(0).multi_processor_count}")
    print(f"Benchmark: CUDA graph replay (eliminates launch overhead)")
    print(f"{'='*80}")
    print()

    results = []

    for M in sizes:
        print(f"--- M={M} ---")

        x = torch.randn(M, D, dtype=torch.float16, device='cuda') * 0.02
        w1 = torch.randn(D, H, dtype=torch.float16, device='cuda') * 0.02
        w2 = torch.randn(H, H, dtype=torch.float16, device='cuda') * 0.02
        w3 = torch.randn(H, OUT_F, dtype=torch.float16, device='cuda') * 0.02

        # Correctness check
        ct_pass, ct_diff, tr_pass, tr_diff = correctness_check(x, w1, w2, w3)
        print(f"  Correctness: cuTile {'PASS' if ct_pass else 'FAIL'} (diff={ct_diff:.6f}), "
              f"Triton {'PASS' if tr_pass else 'FAIL'} (diff={tr_diff:.6f})")

        if not ct_pass or not tr_pass:
            print("  WARNING: Correctness issue detected!")

        # Benchmark cuTile (autotune happens during warmup)
        print(f"  Autotuning cuTile (warmup triggers autotune)...")
        args_ct = (x, w1, w2, w3)

        # Graph-captured benchmark
        ct_g_median, ct_g_mean, _ = benchmark_graph(
            fused_mlp_fwd_cutile, args_ct, warmup=warmup, iters=iters
        )

        # Also run non-graph for comparison
        ct_ng_median, ct_ng_mean, _ = benchmark_fn(
            fused_mlp_fwd_cutile, args_ct, warmup=warmup, iters=iters
        )

        # Benchmark Triton
        tr_g_median, tr_g_mean, _ = benchmark_graph(
            fused_mlp_fwd_triton, (x, w1, w2, w3), warmup=warmup, iters=iters
        )
        tr_ng_median, tr_ng_mean, _ = benchmark_fn(
            fused_mlp_fwd_triton, (x, w1, w2, w3), warmup=warmup, iters=iters
        )

        # Benchmark plain PyTorch
        pt_g_median, pt_g_mean, _ = benchmark_graph(
            pytorch_fwd, (x, w1, w2, w3), warmup=warmup, iters=iters
        )
        pt_ng_median, pt_ng_mean, _ = benchmark_fn(
            pytorch_fwd, (x, w1, w2, w3), warmup=warmup, iters=iters
        )

        speedup_tr = tr_g_median / ct_g_median if ct_g_median > 0 else float('inf')
        speedup_pt = pt_g_median / ct_g_median if ct_g_median > 0 else float('inf')
        results.append({
            'M': M, 'D': D, 'H': H, 'OUT_F': OUT_F,
            'ct_g_median': ct_g_median, 'ct_g_mean': ct_g_mean,
            'tr_g_median': tr_g_median, 'tr_g_mean': tr_g_mean,
            'pt_g_median': pt_g_median, 'pt_g_mean': pt_g_mean,
            'ct_ng_median': ct_ng_median,
            'tr_ng_median': tr_ng_median,
            'pt_ng_median': pt_ng_median,
            'speedup_tr': speedup_tr, 'speedup_pt': speedup_pt,
        })

        print(f"  cuTile:   graph={ct_g_median:.4f}ms  no-graph={ct_ng_median:.4f}ms  (overhead={ct_ng_median-ct_g_median:.4f}ms)")
        print(f"  Triton:   graph={tr_g_median:.4f}ms  no-graph={tr_ng_median:.4f}ms  (overhead={tr_ng_median-tr_g_median:.4f}ms)")
        print(f"  PyTorch:  graph={pt_g_median:.4f}ms  no-graph={pt_ng_median:.4f}ms  (overhead={pt_ng_median-pt_g_median:.4f}ms)")
        print(f"  Speedup vs Triton (graph): {speedup_tr:.2f}x ({'cuTile wins' if speedup_tr > 1 else 'Triton wins'})")
        print(f"  Speedup vs PyTorch (graph): {speedup_pt:.2f}x")
        print()

    # Summary table
    print(f"{'='*80}")
    print("SUMMARY (CUDA graph replay — pure GPU time)")
    print(f"{'='*80}")
    print(f"{'M':>8} {'cuTile(ms)':>12} {'Triton(ms)':>12} {'PyTorch(ms)':>12} {'vs Triton':>10} {'vs PyTorch':>10}")
    print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
    for r in results:
        print(f"{r['M']:>8} {r['ct_g_median']:>12.4f} {r['tr_g_median']:>12.4f} "
              f"{r['pt_g_median']:>12.4f} {r['speedup_tr']:>9.2f}x {r['speedup_pt']:>9.2f}x")
    print()

    # Non-graph comparison
    print(f"{'='*80}")
    print("COMPARISON (non-graph — includes launch overhead)")
    print(f"{'='*80}")
    print(f"{'M':>8} {'cuTile(ms)':>12} {'Triton(ms)':>12} {'PyTorch(ms)':>12}")
    print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    for r in results:
        print(f"{r['M']:>8} {r['ct_ng_median']:>12.4f} {r['tr_ng_median']:>12.4f} "
              f"{r['pt_ng_median']:>12.4f}")
    print()

    # Overhead table
    print(f"{'='*80}")
    print("LAUNCH OVERHEAD (non-graph minus graph, in ms)")
    print(f"{'='*80}")
    print(f"{'M':>8} {'cuTile':>10} {'Triton':>10} {'PyTorch':>10}")
    print(f"{'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for r in results:
        ct_oh = r['ct_ng_median'] - r['ct_g_median']
        tr_oh = r['tr_ng_median'] - r['tr_g_median']
        pt_oh = r['pt_ng_median'] - r['pt_g_median']
        print(f"{r['M']:>8} {ct_oh:>10.4f} {tr_oh:>10.4f} {pt_oh:>10.4f}")
    print()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile cuTile vs Triton MLP megakernel")
    parser.add_argument('--sizes', type=str, default='64,128,256,512,1024,2048,4096',
                        help='Comma-separated batch sizes (default: 64,128,256,512,1024,2048,4096)')
    parser.add_argument('--D', type=int, default=128, help='Input dimension (default: 128)')
    parser.add_argument('--H', type=int, default=128, help='Hidden dimension (default: 128)')
    parser.add_argument('--OUT', type=int, default=128, help='Output dimension (default: 128)')
    parser.add_argument('--warmup', type=int, default=50, help='Warmup iterations (default: 50)')
    parser.add_argument('--iters', type=int, default=100, help='Benchmark iterations (default: 100)')
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(',')]
    run_benchmark(sizes, args.D, args.H, args.OUT, args.warmup, args.iters)
