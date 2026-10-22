"""Iteration bench for the split-K wgrad backward.

Usage: python3 -u bench_splitk.py [old|new|both] [M,M,...]

do_bench (L2 flushed, warmup=10 rep=50) + cosine>=0.99 gate vs eager.
"""
import os, sys, json
import torch
import torch.nn.functional as F
from triton.testing import do_bench

sys.path.insert(0, "/work")
import cutile_mlp as cm

MS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
DEV = "cuda"


def make(M, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(M, 128, dtype=torch.float16, device=DEV, generator=g) * 0.02
    w1 = torch.randn(128, 128, dtype=torch.float16, device=DEV, generator=g) * 0.02
    w2 = torch.randn(128, 128, dtype=torch.float16, device=DEV, generator=g) * 0.02
    w3 = torch.randn(128, 128, dtype=torch.float16, device=DEV, generator=g) * 0.02
    return x, w1, w2, w3


def cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def run_once(M, variant, fwd_bench=False):
    x, w1, w2, w3 = make(M)
    out, H0, H1PRE = cm._fwd_train(x, w1, w2, w3)
    torch.cuda.synchronize()

    g = torch.Generator(device=DEV).manual_seed(42 + M)
    dout = torch.randn(M, 128, dtype=torch.float16, device=DEV, generator=g)

    xr = x.clone().requires_grad_(True)
    wr = [w.clone().requires_grad_(True) for w in (w1, w2, w3)]
    out_r = F.softplus(F.softplus(xr @ wr[0]) @ wr[1]) @ wr[2]
    out_r.backward(dout)
    refs = (xr.grad, wr[0].grad, wr[1].grad, wr[2].grad)

    bwd_fn = cm.fused_mlp_bwd_cutile if variant == "old" else cm.fused_mlp_bwd_splitk

    res = {"M": M, "variant": variant}
    if fwd_bench:
        t = do_bench(lambda: cm.fused_mlp_fwd_cutile(x, w1, w2, w3), warmup=10, rep=50)
        res["fwd_us"] = round(t * 1e3, 1)

    def bwd_call():
        return bwd_fn(dout, x, w1, w2, w3, H0, H1PRE)

    got = bwd_call()
    torch.cuda.synchronize()
    ok = True
    for n, a, b in zip(["dx", "dW1", "dW2", "dW3"], got, refs):
        c = cos(a, b)
        res[f"cos_{n}"] = round(c, 5)
        if c < 0.99:
            ok = False
    res["correct"] = ok

    t = do_bench(bwd_call, warmup=10, rep=50)
    res["bwd_us"] = round(t * 1e3, 1)

    def eager_fbwd():
        xe = x.clone().requires_grad_(True)
        we = [w.clone().requires_grad_(True) for w in (w1, w2, w3)]
        o = F.softplus(F.softplus(xe @ we[0]) @ we[1]) @ we[2]
        o.backward(dout)
    t2 = do_bench(eager_fbwd, warmup=5, rep=20)
    res["eager_fwd_bwd_us"] = round(t2 * 1e3, 1)
    return res


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    ms = MS if len(sys.argv) <= 2 else [int(v) for v in sys.argv[2].split(",")]
    variants = ["old", "new"] if mode == "both" else [mode]
    rows = []
    for M in ms:
        for i, v in enumerate(variants):
            r = run_once(M, v, fwd_bench=(i == 0))
            rows.append(r)
            print(json.dumps(r), flush=True)
    print("\nSUMMARY")
    for r in rows:
        print(f"M={r['M']:>6} [{r['variant']:>3}]  fwd={r.get('fwd_us','-'):>7}  bwd={r['bwd_us']:>7}us  "
              f"eager={r['eager_fwd_bwd_us']:>7}us  correct={r['correct']}  "
              f"cos=({r['cos_dx']},{r['cos_dW1']},{r['cos_dW2']},{r['cos_dW3']})")
