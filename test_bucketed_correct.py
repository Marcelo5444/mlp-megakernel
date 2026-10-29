"""Test A: bucketed-path correctness + padding canaries (fwd AND bwd)."""
import os, sys, time
import torch
import torch.nn.functional as F

os.environ.setdefault("MLP_FAST_AUTOTUNE", "1")  # fast path: heuristic cfgs, no exhaustive tune
import cutile_mlp as cm

torch.manual_seed(0)
DEV = "cuda"
DT = torch.float16
D = 128

def ref_fwd(x, w1, w2, w3):
    return F.softplus(F.softplus(x @ w1) @ w2) @ w3

def cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()

CANARY = 12345.0
w1 = torch.randn(D, D, dtype=DT, device=DEV) * 0.02
w2 = torch.randn(D, D, dtype=DT, device=DEV) * 0.02
w3 = torch.randn(D, D, dtype=DT, device=DEV) * 0.02

M_VALS = [257, 1000, 5000, 31000, 65535]
graph_max = int(os.environ.get("GRAPH_MAX", "4096"))
gm = cm.GraphedBucketedMLP(w1, w2, w3, graph_max_bucket=graph_max)

all_ok = True
for m_real in M_VALS:
    bucket = cm.bucket_for(m_real)
    # Warm cuTile config for this bucket (fast heuristic mode when env set)
    e = gm._get_entry(bucket, want_graph=True)
    print(f"\n=== M_real={m_real} -> bucket={bucket} (graph={e['use_graph']}) ===")

    # ---- exact-M reference through the fused kernels (narrow views) ----
    x_exact = torch.randn(m_real, D, dtype=DT, device=DEV) * 0.02
    out_exact, H0_e, H1_e = cm._fwd_train(x_exact, w1, w2, w3)
    dout = torch.randn(m_real, D, dtype=DT, device=DEV) * 0.02
    dx_e, dw1_e, dw2_e, dw3_e = cm.fused_mlp_bwd_splitk(dout, x_exact, w1, w2, w3, H0_e, H1_e)

    # eager PyTorch reference for grad cosines
    xr = x_exact.clone().requires_grad_(True)
    wr = [w.clone().requires_grad_(True) for w in (w1, w2, w3)]
    ref_fwd(xr, *wr).backward(dout)
    ref_out = ref_fwd(x_exact, w1, w2, w3)

    # ---- bucketed path with canaries in padding ----
    e["x"].fill_(CANARY); e["out"].fill_(CANARY); e["h0"].fill_(CANARY); e["h1pre"].fill_(CANARY)
    e["dout"].fill_(CANARY); e["dx"].fill_(CANARY); e["dh1"].fill_(CANARY)
    out = gm.fwd(x_exact)
    torch.cuda.synchronize()
    ok_fwd = cos(out, out_exact) >= 0.99
    ok_ref = cos(out, ref_out) >= 0.99
    # Canary semantics:
    #  - eager path (no graph): grid/loop derive from M_real, so padded rows
    #    must be COMPLETELY untouched.
    #  - graph path: tiles are frozen at bucket granularity; the last ACTIVE
    #    tile (straddling M_real) is computed and stored — out-of-range rows
    #    inside it land in the padded region but are never consumed (they
    #    contribute zero to the split-K wgrads... see below) and callers only
    #    read [:M_real]. Only rows BEYOND the last active tile must stay
    #    canary.
    dx, dw1, dw2, dw3 = gm.bwd(dout)
    torch.cuda.synchronize()
    ok_dx = cos(dx, dx_e) >= 0.99
    ok_dw = all(cos(a, b) >= 0.99 for a, b in
                ((dw1, dw1_e), (dw2, dw2_e), (dw3, dw3_e)))
    ok_dw_ref = all(cos(a, b.grad) >= 0.99 for a, b in ((dw1, wr[0]), (dw2, wr[1]), (dw3, wr[2])))
    ok_dx_ref = cos(dx, xr.grad) >= 0.99
    # Canary check AFTER both fwd+bwd: rows beyond the last active tile of
    # the COARSEST tile granularity used on that buffer must be canary.
    # fwd tiles: TM (fwd cfg); dgrad/dx/dh1 tiles: 16.
    pad_ok = True
    tm = e["cfg_tr"].TM
    # NB: x/dout straddle windows are deliberately zeroed by the wrapper
    straddle = ((m_real + 127) // 128) * 128
    checks = (("x", 1, 0.0), ("dout", 1, 0.0),
              ("out", tm, CANARY), ("h0", tm, CANARY), ("h1pre", tm, CANARY),
              ("dx", 16, CANARY), ("dh1", 16, CANARY))
    for name, gran, expect in checks:
        end = max(((m_real + gran - 1) // gran) * gran, straddle if expect == 0.0 else 0)
        if expect == 0.0:
            # zeroed straddle window: rows [m_real, straddle) must be 0,
            # rows [straddle, bucket) must still be canary
            if (e[name][m_real:straddle] != 0.0).any():
                pad_ok = False; print(f"  STRADDLE not zeroed in {name}")
            if straddle < bucket and (e[name][straddle:] != CANARY).any():
                pad_ok = False; print(f"  CANARY VIOLATION in {name} rows {straddle}..{bucket}")
        elif end < bucket and (e[name][end:] != expect).any():
            pad_ok = False
            print(f"  CANARY VIOLATION in {name} rows {end}..{bucket}")
    status = ok_fwd and ok_ref and ok_dx and ok_dw and ok_dw_ref and ok_dx_ref and pad_ok
    all_ok &= status
    print(f"  fwd vs exact cos>=0.99:{ok_fwd}  vs eager:{ok_ref}  "
          f"dx:{ok_dx}/{ok_dx_ref} dW:{ok_dw}/{ok_dw_ref} canary:{pad_ok}  "
          f"=> {'PASS' if status else 'FAIL'}")

print("\nOVERALL:", "ALL PASS" if all_ok else "SOME FAILED")
sys.exit(0 if all_ok else 1)
