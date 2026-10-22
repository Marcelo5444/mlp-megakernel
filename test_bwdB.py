"""Strategy B validation + benchmark: fused backward megakernel (single launch).

Correctness vs PyTorch eager autograd (cos >= 0.99 on dx, dW1, dW2, dW3)
at M=256,1024,4096. Benchmark fwd+bwd at M=[256,1024,4096,16384,65536].
"""
import sys
sys.path.insert(0, "/work")
import torch
import torch.nn.functional as F
from triton.testing import do_bench

torch.manual_seed(0)
D, H, N3 = 128, 128, 128

from cutile_mlp import fused_mlp_cutile


def pytorch_ref(x, w1, w2, w3):
    return F.softplus(F.softplus(x @ w1) @ w2) @ w3


def make(M, req=False):
    x = torch.randn(M, D, dtype=torch.float16, device='cuda') * 0.02
    w1 = torch.randn(D, H, dtype=torch.float16, device='cuda') * 0.02
    w2 = torch.randn(H, H, dtype=torch.float16, device='cuda') * 0.02
    w3 = torch.randn(H, N3, dtype=torch.float16, device='cuda') * 0.02
    if req:
        for t in (x, w1, w2, w3):
            t.requires_grad_(True)
    return x, w1, w2, w3


# ---------- correctness ----------
print("=== CORRECTNESS (cos >= 0.99) ===")
all_pass = True
for M in [256, 1024, 4096]:
    x, w1, w2, w3 = make(M, req=True)
    out_ref = pytorch_ref(x, w1, w2, w3)
    out_ref.sum().backward()
    refs = (x.grad.clone(), w1.grad.clone(), w2.grad.clone(), w3.grad.clone())

    x.grad = w1.grad = w2.grad = w3.grad = None
    out_ct = fused_mlp_cutile(x, w1, w2, w3)
    out_ct.sum().backward()
    gots = (x.grad, w1.grad, w2.grad, w3.grad)

    # also check forward matches
    fcos = F.cosine_similarity(out_ref.flatten().float(),
                               out_ct.flatten().float(), dim=0).item()

    line = f"M={M:6d} fwd_cos={fcos:.4f} "
    for name, g, r in zip(("dx", "dW1", "dW2", "dW3"), gots, refs):
        cos = F.cosine_similarity(r.flatten().float(), g.flatten().float(),
                                  dim=0).item()
        ok = cos >= 0.99
        all_pass &= ok
        line += f"{name}={cos:.4f}{'P' if ok else 'F'} "
    print(line)
print("CORRECTNESS:", "ALL PASS" if all_pass else "FAIL")

# ---------- benchmark ----------
print()
print("=== FWD+BWD (do_bench, L2 flushed) ===")
print(f"{'M':>7} {'pytorch':>10} {'cutile-B':>10} {'speedup':>8}")
rows = []
for M in [256, 1024, 4096, 16384, 65536]:
    x, w1, w2, w3 = make(M, req=True)

    def pytorch_fwd_bwd():
        x.grad = w1.grad = w2.grad = w3.grad = None
        out = pytorch_ref(x, w1, w2, w3)
        out.sum().backward()

    def cutile_fwd_bwd():
        x.grad = w1.grad = w2.grad = w3.grad = None
        out = fused_mlp_cutile(x, w1, w2, w3)
        out.sum().backward()

    cutile_fwd_bwd()  # compile/warm
    torch.cuda.synchronize()

    t_p = do_bench(pytorch_fwd_bwd, warmup=10, rep=50) * 1000
    t_c = do_bench(cutile_fwd_bwd, warmup=10, rep=50) * 1000
    rows.append((M, t_p, t_c))
    print(f"{M:7d} {t_p:9.1f}u {t_c:9.1f}u {t_p/t_c:7.2f}x")

print()
print("MARKDOWN_TABLE")
print("| M | pytorch eager (us) | cutile strategy-B (us) | speedup |")
print("|---|---|---|---|")
for M, tp, tc in rows:
    print(f"| {M} | {tp:.1f} | {tc:.1f} | {tp/tc:.2f}x |")
