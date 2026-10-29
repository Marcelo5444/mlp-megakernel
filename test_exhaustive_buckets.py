"""Correctness sweep across ALL 17 buckets at the merged exhaustive configs."""
import os
os.environ.setdefault("MLP_FAST_AUTOTUNE", "0")  # use disk/exhaustive configs
import torch, torch.nn.functional as F, sys
import cutile_mlp as cm

torch.manual_seed(0)
DEV, DT, D = "cuda", torch.float16, 128
w1, w2, w3 = [torch.randn(D, D, dtype=DT, device=DEV) * 0.02 for _ in range(3)]
gm = cm.GraphedBucketedMLP(w1, w2, w3, graph_max_bucket=4096)

all_ok = True
for b in cm.DEFAULT_BUCKETS:
    m_real = b - 17  # awkward M inside the bucket
    x = torch.randn(m_real, D, dtype=DT, device=DEV) * 0.02
    dout = torch.randn(m_real, D, dtype=DT, device=DEV) * 0.02
    xr = x.clone().requires_grad_(True)
    wr = [w.clone().requires_grad_(True) for w in (w1, w2, w3)]
    ref_out = F.softplus(F.softplus(xr @ wr[0]) @ wr[1]) @ wr[2]
    ref_out.backward(dout)
    out = gm.fwd(x)
    dx, dw1, dw2, dw3 = gm.bwd(dout)
    torch.cuda.synchronize()
    c_out = F.cosine_similarity(out.flatten().float(), ref_out.detach().flatten().float(), dim=0).item()
    c_dx = F.cosine_similarity(dx.flatten().float(), xr.grad.flatten().float(), dim=0).item()
    c_dw = min(F.cosine_similarity(a.flatten().float(), w.grad.flatten().float(), dim=0).item()
               for a, w in ((dw1, wr[0]), (dw2, wr[1]), (dw3, wr[2])))
    ok = c_out >= 0.99 and c_dx >= 0.99 and c_dw >= 0.99
    all_ok &= ok
    print(f"bucket {b:>6} (M_real={m_real}): out={c_out:.5f} dx={c_dx:.5f} dWmin={c_dw:.5f} {'PASS' if ok else 'FAIL'}")
print("ALL-BUCKET SWEEP:", "ALL PASS" if all_ok else "SOME FAILED")
sys.exit(0 if all_ok else 1)
