"""
Profile cuTile vs Triton for the 3-layer fused MLP megakernel.

out = softplus(softplus(x @ W1) @ W2) @ W3

Benchmarks multiple small batch sizes, reports timings and speedups.
Uses the Docker cuda133-pytorch-arm64 image with cuTile 1.4 + Triton 3.7.1.

Usage:
    python profile.py
    python profile.py --sizes 256,512,1024
    python profile.py --warmup 50 --iters 100
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
    """Benchmark a function using CUDA events, returning median time in ms."""
    # Warmup (also triggers compilation/autotune)
    for _ in range(warmup):
        out = fn(*args)
    torch.cuda.synchronize()

    # Timed iterations using CUDA events
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


def correctness_check(x, w1, w2, w3):
    """Verify both implementations match PyTorch reference."""
    # PyTorch reference
    h1 = F.softplus(x @ w1)
    h2 = F.softplus(h1 @ w2)
    ref = h2 @ w3

    # cuTile
    out_ct = fused_mlp_fwd_cutile(x, w1, w2, w3)
    torch.cuda.synchronize()
    ct_diff = (out_ct - ref).abs().max().item()
    ct_pass = torch.allclose(out_ct, ref, atol=1e-2, rtol=1e-2)

    # Triton
    out_tr = fused_mlp_fwd_triton(x, w1, w2, w3)
    torch.cuda.synchronize()
    tr_diff = (out_tr - ref).abs().max().item()
    tr_pass = torch.allclose(out_tr, ref, atol=1e-2, rtol=1e-2)

    return ct_pass, ct_diff, tr_pass, tr_diff


def run_benchmark(sizes, D, H, OUT_F, warmup, iters):
    """Run benchmarks for all sizes and return results."""
    print(f"{'='*80}")
    print(f"3-Layer Fused MLP Megakernel: cuTile vs Triton")
    print(f"Architecture: out = softplus(softplus(x @ W1) @ W2) @ W3")
    print(f"D={D}, H={H}, OUT={OUT_F}, FP16, FP32 accumulation")
    print(f"Warmup={warmup}, Iters={iters}")
    print(f"GPU: {torch.cuda.get_device_properties(0).name}")
    print(f"SMs: {torch.cuda.get_device_properties(0).multi_processor_count}")
    print(f"{'='*80}")
    print()

    results = []

    for M in sizes:
        print(f"--- M={M} ---")

        # Create inputs
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

        # Benchmark cuTile (autotune happens on first call)
        print(f"  Autotuning cuTile...")
        ct_median, ct_mean, ct_times = benchmark_fn(
            fused_mlp_fwd_cutile, (x, w1, w2, w3), warmup=warmup, iters=iters
        )

        # Benchmark Triton
        tr_median, tr_mean, tr_times = benchmark_fn(
            fused_mlp_fwd_triton, (x, w1, w2, w3), warmup=warmup, iters=iters
        )

        speedup = tr_median / ct_median if ct_median > 0 else float('inf')
        results.append({
            'M': M, 'D': D, 'H': H, 'OUT_F': OUT_F,
            'ct_median': ct_median, 'ct_mean': ct_mean,
            'tr_median': tr_median, 'tr_mean': tr_mean,
            'speedup': speedup,
        })

        print(f"  cuTile:  median={ct_median:.4f}ms  mean={ct_mean:.4f}ms")
        print(f"  Triton:  median={tr_median:.4f}ms  mean={tr_mean:.4f}ms")
        print(f"  Speedup: {speedup:.2f}x ({'cuTile wins' if speedup > 1 else 'Triton wins'})")
        print()

    # Summary table
    print(f"{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'M':>8} {'cuTile(ms)':>12} {'Triton(ms)':>12} {'Speedup':>10} {'Winner':>12}")
    print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*12}")
    for r in results:
        winner = "cuTile" if r['speedup'] > 1 else "Triton"
        print(f"{r['M']:>8} {r['ct_median']:>12.4f} {r['tr_median']:>12.4f} "
              f"{r['speedup']:>9.2f}x {winner:>12}")
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
