"""Reproduce bench_all's correctness check at large M: out.sum().backward()
through the autograd Function, dout=ones."""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/work")
from cutile_mlp import fused_mlp_cutile

def pytorch_fwd_ref(x, w1, w2, w3):
    return F.softplus(F.softplus(x @ w1) @ w2) @ w3

for M in (4096, 16384, 65536):
    torch.manual_seed(0)
    x = torch.randn(M, 128, dtype=torch.float16, device='cuda') * 0.02
    w1 = torch.randn(128, 128, dtype=torch.float16, device='cuda') * 0.02
    w2 = torch.randn(128, 128, dtype=torch.float16, device='cuda') * 0.02
    w3 = torch.randn(128, 128, dtype=torch.float16, device='cuda') * 0.02
    x.requires_grad_(True); w1.requires_grad_(True)
    w2.requires_grad_(True); w3.requires_grad_(True)

    out_ref = pytorch_fwd_ref(x, w1, w2, w3)
    out_ref.sum().backward()
    ref = (x.grad.clone(), w1.grad.clone(), w2.grad.clone(), w3.grad.clone())

    x.grad = w1.grad = w2.grad = w3.grad = None
    out_ct = fused_mlp_cutile(x, w1, w2, w3)
    out_ct.sum().backward()
    got = (x.grad, w1.grad, w2.grad, w3.grad)

    line = f"M={M}: "
    for n, a, b in zip(("dx", "dW1", "dW2", "dW3"), got, ref):
        c = F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()
        line += f"{n}={c:.5f} "
    print(line, flush=True)
