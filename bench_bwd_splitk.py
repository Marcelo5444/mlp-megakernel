"""Benchmark: fused backward vs split-K backward on GB10."""
import os, sys
os.environ.pop("MLP_FAST_AUTOTUNE", None)
sys.path.insert(0, "/work")
import torch
import torch.nn.functional as F
from triton.testing import do_bench

torch.manual_seed(0)
D, H, N3 = 128, 128, 128

from cutile_mlp import fused_mlp_bwd_splitk, fused_mlp_bwd_cutile

Ms = [256, 1024, 4096, 16384, 65536]

print("=== BACKWARD ONLY: fused vs split-K (GB10, do_bench, L2 flushed) ===")
print(f"{'M':>7} {'fused':>10} {'splitK':>10} {'splitK/fused':>12}")
for M in Ms:
    x = torch.randn(M, D, dtype=torch.float16, device='cuda') * 0.02
    w1 = torch.randn(D, H, dtype=torch.float16, device='cuda') * 0.02
    w2 = torch.randn(H, H, dtype=torch.float16, device='cuda') * 0.02
    w3 = torch.randn(H, N3, dtype=torch.float16, device='cuda') * 0.02

    # Precompute activations
    h0 = x @ w1
    h1 = F.softplus(h0)
    h1_pre = h1 @ w2
    h2 = F.softplus(h1_pre)
    dout = torch.ones(M, N3, dtype=torch.float16, device='cuda')

    H0_buf = h0.float().contiguous()
    H1PRE_buf = h1_pre.float().contiguous()

    t_fused = do_bench(lambda: fused_mlp_bwd_cutile(dout, x, w1, w2, w3, H0_buf, H1PRE_buf), warmup=10, rep=50) * 1000
    t_split = do_bench(lambda: fused_mlp_bwd_splitk(dout, x, w1, w2, w3, H0_buf, H1PRE_buf), warmup=10, rep=50) * 1000

    print(f"{M:7d} {t_fused:9.1f}u {t_split:9.1f}u {t_split/t_fused:11.2f}x")
