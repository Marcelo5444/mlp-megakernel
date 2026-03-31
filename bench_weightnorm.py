"""Benchmark: weight-normalized fused MLP vs baselines.

Compares for both 3-layer and 5-layer:
  1. Eager PyTorch with weight_norm (reference)
  2. torch.compile'd eager with weight_norm
  3. Fused Triton + weight_norm (contiguous params, custom autograd)

Tests: correctness (cosine similarity), inference latency, training latency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fused_mlp import FusedMLPSoftplusWN
from fused_mlp5 import FusedMLP5SoftplusWN


# ═══════════════════════════════════════════════════════════════════════════
# Eager reference models with weight_norm
# ═══════════════════════════════════════════════════════════════════════════

class EagerMLPWN3(nn.Module):
    def __init__(self, D, H, O):
        super().__init__()
        self.v1 = nn.Parameter(torch.randn(H, D) * 0.1)
        self.g1 = nn.Parameter(torch.ones(H, 1))
        self.v2 = nn.Parameter(torch.randn(H, H) * 0.1)
        self.g2 = nn.Parameter(torch.ones(H, 1))
        self.v3 = nn.Parameter(torch.randn(O, H) * 0.1)
        self.g3 = nn.Parameter(torch.ones(O, 1))

    def _wn(self, v, g):
        return g * (v / v.norm(dim=1, keepdim=True))

    def forward(self, x):
        w1, w2, w3 = self._wn(self.v1, self.g1), self._wn(self.v2, self.g2), self._wn(self.v3, self.g3)
        h1 = F.softplus(x @ w1.t())
        h2 = F.softplus(h1 @ w2.t())
        return h2 @ w3.t()


class EagerMLPWN5(nn.Module):
    def __init__(self, D, H, O):
        super().__init__()
        self.v1 = nn.Parameter(torch.randn(H, D) * 0.1)
        self.g1 = nn.Parameter(torch.ones(H, 1))
        self.v2 = nn.Parameter(torch.randn(H, H) * 0.1)
        self.g2 = nn.Parameter(torch.ones(H, 1))
        self.v3 = nn.Parameter(torch.randn(H, H) * 0.1)
        self.g3 = nn.Parameter(torch.ones(H, 1))
        self.v4 = nn.Parameter(torch.randn(H, H) * 0.1)
        self.g4 = nn.Parameter(torch.ones(H, 1))
        self.v5 = nn.Parameter(torch.randn(O, H) * 0.1)
        self.g5 = nn.Parameter(torch.ones(O, 1))

    def _wn(self, v, g):
        return g * (v / v.norm(dim=1, keepdim=True))

    def forward(self, x):
        w1, w2, w3 = self._wn(self.v1, self.g1), self._wn(self.v2, self.g2), self._wn(self.v3, self.g3)
        w4, w5 = self._wn(self.v4, self.g4), self._wn(self.v5, self.g5)
        h1 = F.softplus(x @ w1.t())
        h2 = F.softplus(h1 @ w2.t())
        h3 = F.softplus(h2 @ w3.t())
        h4 = F.softplus(h3 @ w4.t())
        return h4 @ w5.t()


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _eager_v_names(layers):
    names = ["v1", "v2", "v3"]
    if layers == 5:
        names += ["v4", "v5"]
    return names


def _eager_g_names(layers):
    names = ["g1", "g2", "g3"]
    if layers == 5:
        names += ["g4", "g5"]
    return names


def copy_eager_to_fused(eager, fused, layers):
    """Copy individual v_i,g_i from eager model into fused's v_all,g_all."""
    with torch.no_grad():
        v_names = _eager_v_names(layers)
        g_names = _eager_g_names(layers)
        splits = fused._splits
        offset = 0
        for i, (vn, gn) in enumerate(zip(v_names, g_names)):
            vp = getattr(eager, vn)
            gp = getattr(eager, gn)
            r = splits[i]
            fused.v_all.data[offset:offset + r].copy_(vp.data)
            fused.g_all.data[offset:offset + r].copy_(gp.data)
            offset += r


def compare_grads(eager, fused, layers):
    """Compare per-layer gradients between eager (individual params) and fused (contiguous)."""
    v_names = _eager_v_names(layers)
    g_names = _eager_g_names(layers)
    splits = fused._splits

    results = []
    offset = 0
    for i, (vn, gn) in enumerate(zip(v_names, g_names)):
        r = splits[i]
        ge_v = getattr(eager, vn).grad
        gf_v = fused.v_all.grad[offset:offset + r] if fused.v_all.grad is not None else None
        ge_g = getattr(eager, gn).grad
        gf_g = fused.g_all.grad[offset:offset + r] if fused.g_all.grad is not None else None
        if ge_v is not None and gf_v is not None:
            results.append((vn, cosine_sim(ge_v, gf_v)))
        if ge_g is not None and gf_g is not None:
            results.append((gn, cosine_sim(ge_g, gf_g)))
        offset += r
    return results


def bench_fn(fn, warmup=20, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def cosine_sim(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


# ═══════════════════════════════════════════════════════════════════════════
# Correctness test
# ═══════════════════════════════════════════════════════════════════════════

def test_correctness():
    print("=" * 70)
    print("CORRECTNESS TEST")
    print("=" * 70)

    D, H, O = 128, 128, 128

    for M in [256, 1024, 4096]:
        for layers, EagerCls, FusedWNCls in [
            (3, EagerMLPWN3, FusedMLPSoftplusWN),
            (5, EagerMLPWN5, FusedMLP5SoftplusWN),
        ]:
            print(f"\n--- {layers}-layer, M={M} ---")
            eager = EagerCls(D, H, O).cuda().half()
            fused_wn = FusedWNCls(D, H, O).cuda().half()
            copy_eager_to_fused(eager, fused_wn, layers)

            x = torch.randn(M, D, device="cuda", dtype=torch.float16) * 0.1

            with torch.no_grad():
                y_eager = eager(x)
                y_fused = fused_wn(x)
            cs_fwd = cosine_sim(y_eager, y_fused)
            print(f"  Forward  cosine sim: {cs_fwd:.6f}  {'OK' if cs_fwd > 0.999 else 'FAIL'}")

            y_eager2 = eager(x)
            y_fused2 = fused_wn(x)
            y_eager2.sum().backward()
            y_fused2.sum().backward()

            for name, cs in compare_grads(eager, fused_wn, layers):
                ok = cs > 0.99
                print(f"  Grad {name:3s} cosine sim: {cs:.6f}  {'OK' if ok else 'FAIL'}")

            eager.zero_grad(set_to_none=True)
            fused_wn.zero_grad(set_to_none=True)
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════════════

def run_benchmarks():
    print("=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)

    D, H, O = 128, 128, 128
    batch_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                   50000, 65536, 100000, 131072, 200000, 262144, 300000]

    for layers in [3, 5]:
        print(f"\n{'=' * 70}")
        print(f" {layers}-LAYER MLP  (D={D}, H={H}, O={O}, fp16)")
        print(f"{'=' * 70}")

        if layers == 3:
            EagerCls, FusedWNCls = EagerMLPWN3, FusedMLPSoftplusWN
        else:
            EagerCls, FusedWNCls = EagerMLPWN5, FusedMLP5SoftplusWN

        eager_wn = EagerCls(D, H, O).cuda().half()
        compile_wn = torch.compile(EagerCls(D, H, O).cuda().half(), mode="max-autotune")
        fused_wn = FusedWNCls(D, H, O).cuda().half()

        # ── Inference ──────────────────────────────────────────────────
        print(f"\n{'─' * 70}")
        print(f" INFERENCE (forward only, no grad)")
        print(f"{'─' * 70}")
        header = f"{'N':>7s} | {'Eager':>10s} | {'Compile':>10s} | {'Fused':>10s} | {'F/Eager':>10s} | {'F/Compile':>10s}"
        print(header)
        print("-" * len(header))

        for M in batch_sizes:
            x = torch.randn(M, D, device="cuda", dtype=torch.float16) * 0.1
            with torch.no_grad():
                try:
                    compile_wn(x)
                except Exception:
                    pass
                t_eager = bench_fn(lambda: eager_wn(x))
                t_compile = bench_fn(lambda: compile_wn(x))
                t_fused = bench_fn(lambda: fused_wn(x))
            print(f"{M:>7d} | {t_eager:>8.3f}ms | {t_compile:>8.3f}ms | {t_fused:>8.3f}ms | {t_eager/t_fused:>8.2f}x | {t_compile/t_fused:>8.2f}x")

        # ── Training ──────────────────────────────────────────────────
        print(f"\n{'─' * 70}")
        print(f" TRAINING (forward + backward)")
        print(f"{'─' * 70}")
        header = f"{'N':>7s} | {'Eager':>10s} | {'Compile':>10s} | {'Fused':>10s} | {'F/Eager':>10s} | {'F/Compile':>10s}"
        print(header)
        print("-" * len(header))

        for M in batch_sizes:
            x = torch.randn(M, D, device="cuda", dtype=torch.float16) * 0.1

            def train_step(model, inp):
                model.zero_grad(set_to_none=True)
                out = model(inp)
                out.sum().backward()

            try:
                train_step(compile_wn, x)
            except Exception:
                pass

            t_eager = bench_fn(lambda: train_step(eager_wn, x))
            t_compile = bench_fn(lambda: train_step(compile_wn, x))
            t_fused = bench_fn(lambda: train_step(fused_wn, x))
            print(f"{M:>7d} | {t_eager:>8.3f}ms | {t_compile:>8.3f}ms | {t_fused:>8.3f}ms | {t_eager/t_fused:>8.2f}x | {t_compile/t_fused:>8.2f}x")

        del eager_wn, compile_wn, fused_wn
        torch.cuda.empty_cache()


if __name__ == "__main__":
    test_correctness()
    run_benchmarks()
