"""Isolate unified kernel correctness: test STORE_ACTS=1 stores + bwd separately."""
import sys
sys.path.insert(0, "/work")
import torch
import torch.nn.functional as F
from cutile_mlp import (fused_mlp_fwd_cutile, fused_mlp_fwd_store_acts,
                        fused_mlp_bwd_cutile)

torch.manual_seed(0)
D, H, N3 = 128, 128, 128

for M in [256, 1024, 4096]:
    x = torch.randn(M, D, dtype=torch.float16, device='cuda') * 0.02
    w1 = torch.randn(D, H, dtype=torch.float16, device='cuda') * 0.02
    w2 = torch.randn(H, H, dtype=torch.float16, device='cuda') * 0.02
    w3 = torch.randn(H, N3, dtype=torch.float16, device='cuda') * 0.02

    # Reference activations
    ref_H0 = x.float() @ w1.float()              # (M, N1) pre-activation 1
    ref_H1 = F.softplus(ref_H0)
    ref_H1PRE = ref_H1 @ w2.float()              # (M, N2) pre-activation 2
    ref_out = F.softplus(F.softplus(x @ w1) @ w2) @ w3

    # Unified kernel with STORE_ACTS=1
    out, H0, H1PRE = fused_mlp_fwd_store_acts(x, w1, w2, w3)
    torch.cuda.synchronize()

    ok_out = torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2)
    cos_h0 = F.cosine_similarity(ref_H0.flatten(), H0.flatten(), dim=0).item()
    cos_h1 = F.cosine_similarity(ref_H1PRE.flatten(), H1PRE.flatten(), dim=0).item()
    print(f"M={M}: out={'PASS' if ok_out else 'FAIL'} "
          f"H0 cos={cos_h0:.6f} {'PASS' if cos_h0 >= 0.99 else 'FAIL'} "
          f"H1PRE cos={cos_h1:.6f} {'PASS' if cos_h1 >= 0.99 else 'FAIL'}")

    # Backward with stored acts
    x.requires_grad_(True); w1.requires_grad_(True)
    w2.requires_grad_(True); w3.requires_grad_(True)
    ref = F.softplus(F.softplus(x @ w1) @ w2) @ w3
    ref.sum().backward()
    refs = (x.grad.clone(), w1.grad.clone(), w2.grad.clone(), w3.grad.clone())

    dout = torch.ones_like(ref_out)
    dx, dw1, dw2, dw3 = fused_mlp_bwd_cutile(dout, x.detach(), w1.detach(),
                                             w2.detach(), w3.detach(), H0, H1PRE)
    torch.cuda.synchronize()

    for name, got, ref_g in zip(['dx','dW1','dW2','dW3'], [dx,dw1,dw2,dw3], refs):
        cos = F.cosine_similarity(ref_g.flatten().float(), got.flatten().float(), dim=0).item()
        print(f"  {name}: cos={cos:.6f} {'PASS' if cos >= 0.99 else 'FAIL'}")
    print()
