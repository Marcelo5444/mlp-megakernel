"""Autotune bwd_fused_kernel hints (TM x occupancy x num_worker_warps x num_ctas)
on GB10, per M-shape. D=H=OUT=128 fixed (kernel contract: square 128 dims).

Sweeps TM in {8,16,32,64,128} x occ {1,2,4} x nww {None,4,8} x num_ctas {1,2}
via cuda.tile.tune.exhaustive_search (flush_l2, compiler_timeout 5s), validates
each candidate's 4 grads (cos >= 0.99 in fp32 vs eager autograd), and records
the best VALID config per M.
"""
import sys
sys.path.insert(0, "/work")
import json
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import cuda.tile as ct
from cuda.tile.tune import exhaustive_search
from triton.testing import do_bench

torch.manual_seed(0)
D, H, N3 = 128, 128, 128

from cutile_mlp import (
    bwd_fused_kernel, fused_mlp_fwd_store_acts, _kernel_hints, _get_num_sms,
)

# kernels tuned per (occ, nww, nctas) — replace_hints recompiles, so cache
_KCACHE = {}


def _kvar(occ, nww, nctas):
    key = (occ, nww, nctas)
    if key not in _KCACHE:
        h = {"occupancy": occ,
             "num_ctas": ct.ByTarget(sm_90=nctas, sm_100=nctas, sm_120=nctas,
                                     sm_121=nctas, sm_89=nctas)}
        if nww is not None:
            h["num_worker_warps"] = nww
        _KCACHE[key] = bwd_fused_kernel.replace_hints(**h)
    return _KCACHE[key]


def _launch(stream, M, TM, kern, x, w1, w2, w3, dout, H0, H1PRE,
            dx32, dw1_32, dw2_32, dw3_32):
    num_sms = _get_num_sms()
    grid = (min(num_sms, (M + TM - 1) // TM),)
    ct.launch(stream, grid, kern, (
        x, w1, w2, w3, dout, H0, H1PRE, dx32, dw1_32, dw2_32, dw3_32,
        M, D, H, H, N3, TM, 128, 128,
    ))


def run_bwd(M, TM, occ, nww, nctas, tensors, zero=True):
    (x, w1, w2, w3, dout, H0, H1PRE, dx32, dw1_32, dw2_32, dw3_32) = tensors
    if zero:
        dx32.zero_(); dw1_32.zero_(); dw2_32.zero_(); dw3_32.zero_()
    stream = torch.cuda.current_stream()
    _launch(stream, M, TM, _kvar(occ, nww, nctas), *tensors[:11])


def check_grads(M, TM, occ, nww, nctas, tensors, refs):
    run_bwd(M, TM, occ, nww, nctas, tensors, zero=True)
    torch.cuda.synchronize()
    (_, _, _, _, _, _, _, dx32, dw1_32, dw2_32, dw3_32) = tensors
    gots = (dx32, dw1_32, dw2_32, dw3_32)
    cos = []
    for g, r in zip(gots, refs):
        c = F.cosine_similarity(r.flatten().float(), g.flatten().float(),
                                dim=0).item()
        cos.append(c)
    return min(cos), cos


def bench_config(M, TM, occ, nww, nctas, tensors):
    fn = lambda: run_bwd(M, TM, occ, nww, nctas, tensors, zero=True)
    fn(); torch.cuda.synchronize()
    return do_bench(fn, warmup=10, rep=50) * 1000


def main():
    Ms = [256, 1024, 4096, 16384, 65536]
    TMs = [8, 16, 32, 64, 128]
    OCCS = [1, 2, 4]
    NWWS = [None, 4, 8]
    NCTAS = [1, 2]

    results = {}

    for M in Ms:
        print(f"\n================ M={M} ================", flush=True)
        x = torch.randn(M, D, dtype=torch.float16, device='cuda') * 0.02
        w1 = torch.randn(D, H, dtype=torch.float16, device='cuda') * 0.02
        w2 = torch.randn(H, H, dtype=torch.float16, device='cuda') * 0.02
        w3 = torch.randn(H, N3, dtype=torch.float16, device='cuda') * 0.02
        dout = torch.randn(M, N3, dtype=torch.float16, device='cuda') * 0.02

        _, H0, H1PRE = fused_mlp_fwd_store_acts(x, w1, w2, w3)

        dx32 = torch.zeros((M, D), device='cuda', dtype=torch.float32)
        dw1_32 = torch.zeros((D, H), device='cuda', dtype=torch.float32)
        dw2_32 = torch.zeros((H, H), device='cuda', dtype=torch.float32)
        dw3_32 = torch.zeros((H, N3), device='cuda', dtype=torch.float32)
        tensors = (x, w1, w2, w3, dout, H0, H1PRE,
                   dx32, dw1_32, dw2_32, dw3_32)

        # eager autograd reference grads
        xr = x.clone().requires_grad_(True)
        w1r = w1.clone().requires_grad_(True)
        w2r = w2.clone().requires_grad_(True)
        w3r = w3.clone().requires_grad_(True)
        out = F.softplus(F.softplus(xr @ w1r) @ w2r) @ w3r
        out.backward(dout)
        refs = tuple(t.grad.clone().float() for t in (xr, w1r, w2r, w3r))
        del xr, w1r, w2r, w3r, out

        stream = torch.cuda.current_stream()
        num_sms = _get_num_sms()

        space = []
        for TM in TMs:
            if TM > max(8, M):
                continue
            for occ in OCCS:
                for nww in NWWS:
                    for nctas in NCTAS:
                        space.append(SimpleNamespace(
                            TM=TM, occupancy=occ, num_worker_warps=nww,
                            num_ctas=nctas))

        def grid_fn(cfg):
            return (min(num_sms, (M + cfg.TM - 1) // cfg.TM),)

        def args_fn(cfg):
            return (x, w1, w2, w3, dout, H0, H1PRE,
                    dx32, dw1_32, dw2_32, dw3_32, M,
                    D, H, H, N3, cfg.TM, 128, 128)

        print(f"  [search] {len(space)} configs...", flush=True)
        with ct.compiler_timeout(5):
            res = exhaustive_search(
                space, stream, grid_fn, bwd_fused_kernel,
                args_fn, lambda cfg: _kernel_hints(cfg),
                quiet=True, flush_l2=True,
            )
        n_ok = len(res.successes)
        print(f"  [search] {n_ok}/{len(space)} compiled+ran", flush=True)

        # validate in speed order, keep best valid
        ranked = sorted(res.successes, key=lambda m: m.mean_us)
        best = None
        tried = 0
        for meas in ranked:
            cfg = meas.config
            tried += 1
            try:
                mincos, cos = check_grads(M, cfg.TM, cfg.occupancy,
                                          cfg.num_worker_warps, cfg.num_ctas,
                                          tensors, refs)
            except Exception as e:
                print(f"    TM={cfg.TM} occ={cfg.occupancy} "
                      f"nww={cfg.num_worker_warps} nctas={cfg.num_ctas}: "
                      f"VALIDATE-ERR {e}", flush=True)
                continue
            ok = mincos >= 0.99
            tag = "VALID" if ok else f"BAD cos={['%.3f'%c for c in cos]}"
            if tried <= 8 or (ok and best is None):
                print(f"    TM={cfg.TM:3d} occ={cfg.occupancy} "
                      f"nww={str(cfg.num_worker_warps):>4} "
                      f"nctas={cfg.num_ctas} {meas.mean_us:8.1f}us  {tag}",
                      flush=True)
            if ok:
                best = (cfg, meas.mean_us, cos)
                break

        if best is None:
            print(f"  !! no valid config for M={M}; keeping baseline", flush=True)
            results[M] = {"TM": 16, "occ": 1, "nww": 8, "nctas": 1,
                          "kernel_us": None, "valid": False,
                          "cos": None}
            continue

        cfg, kus, cos = best
        # proper do_bench timing of the winner
        bus = bench_config(M, cfg.TM, cfg.occupancy, cfg.num_worker_warps,
                           cfg.num_ctas, tensors)
        print(f"  BEST M={M}: TM={cfg.TM} occ={cfg.occupancy} "
              f"nww={cfg.num_worker_warps} nctas={cfg.num_ctas} "
              f"search={kus:.1f}us bench={bus:.1f}us "
              f"cos={['%.4f'%c for c in cos]}", flush=True)
        results[M] = {"TM": cfg.TM, "occ": cfg.occupancy,
                      "nww": cfg.num_worker_warps, "nctas": cfg.num_ctas,
                      "kernel_us": bus, "valid": True,
                      "cos": [round(c, 4) for c in cos]}
        with open("/work/tune_bwd_results.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\n===== FINAL TABLE =====")
    for M, r in results.items():
        print(f"M={M}: {r}")
    with open("/work/tune_bwd_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
