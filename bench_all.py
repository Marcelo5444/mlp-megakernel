"""Benchmark: PyTorch eager vs torch.compile vs Triton vs cuTile (fwd + bwd).

Uses triton.testing.do_bench with L2 flush for stable, skill-aligned timing.
Compares: torch eager, torch.compile (max-autotune-no-cudagraphs), Triton, cuTile.
"""
import os
import sys
os.environ.pop("MLP_FAST_AUTOTUNE", None)  # force full exhaustive autotune
sys.path.insert(0, "/work")
import torch
import torch.nn.functional as F
from triton.testing import do_bench

torch.manual_seed(0)
D, H, N3 = 128, 128, 128


def pytorch_fwd_ref(x, w1, w2, w3):
    return F.softplus(F.softplus(x @ w1) @ w2) @ w3


# torch.compile is applied lazily inside the benchmark loop to avoid
# graph-breaking the eager reference backward used for correctness checks.
_pytorch_fwd_compile = None

def pytorch_fwd_compile(x, w1, w2, w3):
    global _pytorch_fwd_compile
    if _pytorch_fwd_compile is None:
        _pytorch_fwd_compile = torch.compile(
            pytorch_fwd_ref, fullgraph=True, mode="max-autotune-no-cudagraphs"
        )
    return _pytorch_fwd_compile(x, w1, w2, w3)


from kernel import ModelNew as TritonMLP
from cutile_mlp import fused_mlp, fused_mlp_fwd_cutile, fused_mlp_cutile, _fwd_train

triton_model = TritonMLP()

Ms = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

# Matched-context forward comparison:
#  - "cutile-inf"   = inference-only forward (no activation storage) — fair vs
#                     torch.no_grad / deployment inference.
#  - "cutile-train" = forward WITH fp32 H0/H1_PRE activation storage — this is
#                     the forward actually paid in the fwd+bwd pass below, fair
#                     vs PyTorch eager forward with autograd state.
print("=== FORWARD PASS (do_bench, L2 flushed) ===")
print(f"{'M':>7} {'pytorch':>10} {'t.compile':>10} {'triton':>10} {'cut-inf':>10} {'cut-train':>10} {'T spd':>6} {'C spd':>6} {'PC spd':>7}")
rows_fwd = []
for M in Ms:
    x = torch.randn(M, D, dtype=torch.float16, device='cuda') * 0.02
    w1 = torch.randn(D, H, dtype=torch.float16, device='cuda') * 0.02
    w2 = torch.randn(H, H, dtype=torch.float16, device='cuda') * 0.02
    w3 = torch.randn(H, N3, dtype=torch.float16, device='cuda') * 0.02

    ref = pytorch_fwd_ref(x, w1, w2, w3)
    out_t = triton_model(x, w1, w2, w3)
    out_c = fused_mlp_fwd_cutile(x, w1, w2, w3)
    out_s, _, _ = _fwd_train(x, w1, w2, w3)  # training fwd (STORE_ACTS=1)
    torch.cuda.synchronize()
    ok_t = torch.allclose(out_t, ref, atol=1e-2, rtol=1e-2)
    ok_c = torch.allclose(out_c, ref, atol=1e-2, rtol=1e-2)
    ok_s = torch.allclose(out_s, ref, atol=1e-2, rtol=1e-2)

    t_p = do_bench(lambda: pytorch_fwd_ref(x, w1, w2, w3), warmup=25, rep=100) * 1000
    t_pc = do_bench(lambda: pytorch_fwd_compile(x, w1, w2, w3), warmup=25, rep=100) * 1000
    t_t = do_bench(lambda: triton_model(x, w1, w2, w3), warmup=25, rep=100) * 1000
    t_c = do_bench(lambda: fused_mlp_fwd_cutile(x, w1, w2, w3), warmup=25, rep=100) * 1000
    t_s = do_bench(lambda: _fwd_train(x, w1, w2, w3), warmup=25, rep=100) * 1000

    rows_fwd.append((M, t_p, t_pc, t_t, t_c, t_s, ok_t and ok_c and ok_s))
    print(f"{M:7d} {t_p:9.1f}u {t_pc:9.1f}u {t_t:9.1f}u {t_c:9.1f}u {t_s:9.1f}u {t_p/t_t:5.2f}x {t_p/t_c:5.2f}x {t_p/t_pc:6.2f}x"
          + ("" if (ok_t and ok_c and ok_s) else "  !!CORRECTNESS"))

print()
print("=== FORWARD + BACKWARD PASS (do_bench, L2 flushed) ===")
print(f"{'M':>7} {'pytorch':>10} {'t.compile':>10} {'cutile':>10} {'C spd':>6} {'PC spd':>7}")
rows_bwd = []
for M in Ms:
    # Correctness check first, before any torch.compile calls
    x = torch.randn(M, D, dtype=torch.float16, device='cuda') * 0.02
    w1 = torch.randn(D, H, dtype=torch.float16, device='cuda') * 0.02
    w2 = torch.randn(H, H, dtype=torch.float16, device='cuda') * 0.02
    w3 = torch.randn(H, N3, dtype=torch.float16, device='cuda') * 0.02
    x.requires_grad_(True)
    w1.requires_grad_(True)
    w2.requires_grad_(True)
    w3.requires_grad_(True)

    # Eager reference backward
    out_ref = pytorch_fwd_ref(x, w1, w2, w3)
    out_ref.sum().backward()
    ref_dx, ref_dw1 = x.grad.clone(), w1.grad.clone()

    # cuTile backward
    x.grad = None; w1.grad = None; w2.grad = None; w3.grad = None
    out_ct = fused_mlp_cutile(x, w1, w2, w3)
    out_ct.sum().backward()
    # fp16 accumulates ~65K entries in fp16 dot products -> inf/precision
    # loss; compute cosine in fp32 (same as bench_splitk.py).
    ok_dx = F.cosine_similarity(ref_dx.flatten().float(), x.grad.flatten().float(), dim=0).item() >= 0.99
    ok_dw1 = F.cosine_similarity(ref_dw1.flatten().float(), w1.grad.flatten().float(), dim=0).item() >= 0.99

    # Now benchmark (fresh tensors to avoid graph pollution)
    x = torch.randn(M, D, dtype=torch.float16, device='cuda') * 0.02
    w1 = torch.randn(D, H, dtype=torch.float16, device='cuda') * 0.02
    w2 = torch.randn(H, H, dtype=torch.float16, device='cuda') * 0.02
    w3 = torch.randn(H, N3, dtype=torch.float16, device='cuda') * 0.02
    x.requires_grad_(True)
    w1.requires_grad_(True)
    w2.requires_grad_(True)
    w3.requires_grad_(True)

    def pytorch_fwd_bwd():
        x.grad = None; w1.grad = None; w2.grad = None; w3.grad = None
        out = pytorch_fwd_ref(x, w1, w2, w3)
        out.sum().backward()

    def pytorch_compile_fwd_bwd():
        x.grad = None; w1.grad = None; w2.grad = None; w3.grad = None
        out = pytorch_fwd_compile(x, w1, w2, w3)
        out.sum().backward()

    def cutile_fwd_bwd():
        x.grad = None; w1.grad = None; w2.grad = None; w3.grad = None
        out = fused_mlp_cutile(x, w1, w2, w3)
        out.sum().backward()

    t_p = do_bench(pytorch_fwd_bwd, warmup=10, rep=50) * 1000
    t_pc = do_bench(pytorch_compile_fwd_bwd, warmup=10, rep=50) * 1000
    t_c = do_bench(cutile_fwd_bwd, warmup=10, rep=50) * 1000

    rows_bwd.append((M, t_p, t_pc, t_c, ok_dx and ok_dw1))
    print(f"{M:7d} {t_p:9.1f}u {t_pc:9.1f}u {t_c:9.1f}u {t_p/t_c:5.2f}x {t_p/t_pc:6.2f}x"
          + ("" if (ok_dx and ok_dw1) else "  !!CORRECTNESS"))

print()
print("=== BACKWARD ONLY (do_bench, L2 flushed) ===")
print(f"{'M':>7} {'pytorch':>10} {'t.compile':>10} {'cutile':>10} {'C spd':>6} {'PC spd':>7}")
rows_bwd_only = []
for M in Ms:
    x = torch.randn(M, D, dtype=torch.float16, device='cuda') * 0.02
    w1 = torch.randn(D, H, dtype=torch.float16, device='cuda') * 0.02
    w2 = torch.randn(H, H, dtype=torch.float16, device='cuda') * 0.02
    w3 = torch.randn(H, N3, dtype=torch.float16, device='cuda') * 0.02

    # Precompute activations for backward (same for all backends)
    h0 = x @ w1
    h1 = F.softplus(h0)
    h1_pre = h1 @ w2
    h2 = F.softplus(h1_pre)
    dout = torch.ones(M, N3, dtype=torch.float16, device='cuda')

    # PyTorch backward: manual chain rule (no autograd overhead)
    def pytorch_bwd_manual():
        dh2 = dout @ w3.T
        dh1_pre = dh2 * torch.sigmoid(h1_pre)
        dh1 = dh1_pre @ w2.T
        dh0 = dh1 * torch.sigmoid(h0)
        dx = dh0 @ w1.T
        dW3 = h2.T @ dout
        dW2 = h1.T @ dh1_pre
        dW1 = x.T @ dh0
        return dx, dW1, dW2, dW3

    # torch.compile backward (compiled chain rule)
    @torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
    def pytorch_bwd_compile(dout, x, w1, w2, w3, h0, h1_pre, h2):
        dh2 = dout @ w3.T
        dh1_pre = dh2 * torch.sigmoid(h1_pre)
        dh1 = dh1_pre @ w2.T
        dh0 = dh1 * torch.sigmoid(h0)
        dx = dh0 @ w1.T
        dW3 = h2.T @ dout
        dW2 = h1.T @ dh1_pre
        dW1 = x.T @ dh0
        return dx, dW1, dW2, dW3

    # cuTile backward: direct split-K kernel call with stored activations
    from cutile_mlp import fused_mlp_bwd_splitk
    H0_buf = h0.float().contiguous()
    H1PRE_buf = h1_pre.float().contiguous()

    def cutile_bwd_only():
        return fused_mlp_bwd_splitk(dout, x, w1, w2, w3, H0_buf, H1PRE_buf)

    t_p = do_bench(pytorch_bwd_manual, warmup=10, rep=50) * 1000
    t_pc = do_bench(lambda: pytorch_bwd_compile(dout, x, w1, w2, w3, h0, h1_pre, h2), warmup=10, rep=50) * 1000
    t_c = do_bench(cutile_bwd_only, warmup=10, rep=50) * 1000

    rows_bwd_only.append((M, t_p, t_pc, t_c))
    print(f"{M:7d} {t_p:9.1f}u {t_pc:9.1f}u {t_c:9.1f}u {t_p/t_c:5.2f}x {t_p/t_pc:6.2f}x")

print()
print("=== SUMMARY (do_bench, L2 flushed) ===")
print(f"{'M':>7} {'pytorch':>10} {'t.compile':>10} {'cutile':>10} {'C spd':>6} {'PC spd':>7}")
for M, tp, tpc, tc, ok in rows_bwd:
    print(f"{M:7d} {tp:9.1f}u {tpc:9.1f}u {tc:9.1f}u {tp/tc:5.2f}x {tp/tpc:6.2f}x"
          + ("" if ok else "  !!CORRECTNESS"))
