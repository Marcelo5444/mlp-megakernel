"""cuTile 3-layer fused MLP megakernel — optimized with cached fast-path launcher.

Architecture: out = softplus(softplus(x @ W1) @ W2) @ W3

All GEMMs + activations in ONE persistent kernel. Intermediates in registers.
Autotuned via ct.tune.exhaustive_search sweeping:
  - Tile sizes (TM, TN3, TK1) — TK2/TK3 matched to hidden dims
  - GROUP_M sweep (4, 8, 16) — L2 cache locality
  - occupancy (1,2,3,4)
  - num_ctas (1,2) with ByTarget — arch-aware CGA
  - num_worker_warps (None,4,8) — gated on cuTile >= 1.4
  - latency hints per load (1,2,4,8) — pipeline depth
  - allow_tma per load — TMA vs LDGSTS
  - compiler_timeout cap
  - GPU capability-aware tile caps

Key optimization: cached fast-path launcher eliminates Python overhead after
the first call — precomputes grid, args tuple, and compiled kernel.
"""
import torch
import os
import math
from types import SimpleNamespace
import cuda.tile as ct
from cuda.tile import ByTarget
from cuda.tile.tune import exhaustive_search

_NUM_SMS = None
_BWD_FUSED_K8 = None  # cached nww=8/occ=1 variant of bwd_fused_kernel


def _get_num_sms():
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    return _NUM_SMS


def softplus_tile(x):
    """Numerically stable softplus: where(x > 20, x, log(exp(x) + 1))."""
    return ct.where(x > 20.0, x, ct.log(ct.exp(x) + 1.0))


# ============================================================
# Kernel
# ============================================================

@ct.kernel
def fused_mlp_kernel(
    X, W1, W2, W3, OUT, H0, H1PRE,
    M,
    K1: ct.Constant[int], N1: ct.Constant[int], N2: ct.Constant[int], N3: ct.Constant[int],
    TM: ct.Constant[int], TN3: ct.Constant[int],
    TK1: ct.Constant[int], TK2: ct.Constant[int], TK3: ct.Constant[int],
    GROUP_M: ct.Constant[int],
    LATENCY_X: ct.Constant[int], LATENCY_W1: ct.Constant[int],
    LATENCY_W2: ct.Constant[int], LATENCY_W3: ct.Constant[int],
    USE_TMA_X: ct.Constant[int], USE_TMA_W1: ct.Constant[int],
    USE_TMA_W2: ct.Constant[int], USE_TMA_W3: ct.Constant[int],
    STORE_ACTS: ct.Constant[int],
):
    """Unified persistent kernel for inference and training.

    STORE_ACTS=0 (inference): compute OUT only — H0/H1PRE stores are compiled
    out entirely (zero overhead vs the old inference-only kernel).

    STORE_ACTS=1 (training): additionally store pre-activations H0 (M,N1) and
    H1_PRE (M,N2) in fp32 for the backward pass.
    """
    raw_pid = ct.bid(0)
    num_pids = ct.num_blocks(0)

    num_m = ct.cdiv(M, TM)
    num_n3 = ct.cdiv(N3, TN3)
    num_tiles = num_m * num_n3

    for tile_id in range(raw_pid, num_tiles, num_pids):
        group_id = tile_id // (GROUP_M * num_n3)
        group_size = min(num_m - group_id * GROUP_M, GROUP_M)
        pm = group_id * GROUP_M + (tile_id % group_size)
        pn3 = (tile_id % (GROUP_M * num_n3)) // group_size

        out_acc = ct.zeros((TM, TN3), ct.float32)

        for k3 in range(ct.cdiv(N2, TK3)):
            h2_acc = ct.zeros((TM, TK3), ct.float32)

            for k2 in range(ct.cdiv(N1, TK2)):
                h1_acc = ct.zeros((TM, TK2), ct.float32)

                for k1 in range(ct.cdiv(K1, TK1)):
                    x_tile = ct.load(X, (pm, k1), shape=(TM, TK1),
                                     latency=LATENCY_X, allow_tma=bool(USE_TMA_X))
                    w1_tile = ct.load(W1, (k1, k2), shape=(TK1, TK2),
                                      latency=LATENCY_W1, allow_tma=bool(USE_TMA_W1))
                    h1_acc = ct.mma(x_tile, w1_tile, h1_acc)

                if STORE_ACTS and k3 == 0:
                    ct.store(H0, (pm, k2), h1_acc)  # fp32 pre-activation

                h1_act = softplus_tile(h1_acc)
                h1_fp16 = ct.astype(h1_act, ct.float16)

                w2_tile = ct.load(W2, (k2, k3), shape=(TK2, TK3),
                                  latency=LATENCY_W2, allow_tma=bool(USE_TMA_W2))
                h2_acc = ct.mma(h1_fp16, w2_tile, h2_acc)

            if STORE_ACTS:
                ct.store(H1PRE, (pm, k3), h2_acc)  # fp32 pre-activation

            h2_act = softplus_tile(h2_acc)
            h2_fp16 = ct.astype(h2_act, ct.float16)

            w3_tile = ct.load(W3, (k3, pn3), shape=(TK3, TN3),
                              latency=LATENCY_W3, allow_tma=bool(USE_TMA_W3))
            out_acc = ct.mma(h2_fp16, w3_tile, out_acc)

        ct.store(OUT, (pm, pn3), ct.astype(out_acc, ct.float16))


# ============================================================
# GPU capability-aware settings (adasplash pattern)
# ============================================================

_MAX_TILE = {
    (8, 9): 128,   # sm_89 (RTX 4090)
    (9, 0): 128,   # sm_90 (H100)
    (10, 0): 128,  # sm_100 (B200)
    (12, 0): 128,  # sm_120
    (12, 1): 256,  # sm_121 (GB10)
}


def _next_pow2(n):
    return 1 << (n - 1).bit_length() if n > 0 else 1


def _cutile_version():
    try:
        return tuple(int(x) for x in ct.__version__.split('.')[:2])
    except Exception:
        return (0, 0)


# ============================================================
# Autotune search space (adasplash _configs pattern)
# ============================================================

def _kernel_hints(cfg):
    """Map config to cuTile compiler hints (ByTarget for arch portability)."""
    hints = {
        "num_ctas": ByTarget(
            sm_90=cfg.num_ctas, sm_100=cfg.num_ctas,
            sm_120=cfg.num_ctas, sm_121=cfg.num_ctas,
            sm_89=cfg.num_ctas,
        ),
        "occupancy": cfg.occupancy,
    }
    nww = getattr(cfg, "num_worker_warps", None)
    if nww is not None:
        hints["num_worker_warps"] = nww
    return hints


def _configs(K1, N1, N2, N3):
    """Generate the full autotune search space.

    Sweeps tile sizes x occupancy x num_ctas x num_worker_warps x GROUP_M.
    GPU capability-aware: tile sizes capped by _MAX_TILE, num_ctas restricted
    on non-CGA archs. Uses SimpleNamespace like adasplash.

    Search strategy (OFAT where interactions are weak, full cross where strong):
    - Phase 1: tile sizes x hints x GROUP_M — full cross
    - Phase 2: latency sweep at best tile+hints baseline (OFAT)
    - Phase 3: TMA mask sweep at best tile+hints baseline (OFAT)
    Phases 2/3 are small and fast. Phase 1 is the big one.
    """
    cc = torch.cuda.get_device_capability()
    max_block = _MAX_TILE.get(cc, 128)

    # Tile sizes — FULL space (user wants maximum autotune coverage).
    tm_choices = [16, 32, 64, 128]
    tm_choices = [v for v in tm_choices if v <= max_block]

    tn3_choices = [v for v in [64, 128, 256] if v <= max(N3, 32) and v <= max_block]

    tk1_choices = [v for v in [8, 16, 32, 64, 128] if v <= K1]
    if not tk1_choices:
        tk1_choices = [_next_pow2(K1)]

    if N1 >= 128:
        tk2_choices = [32, 64, 128]
    elif N1 >= 64:
        tk2_choices = [32, 64]
    else:
        tk2_choices = [_next_pow2(N1)]

    if N2 >= 128:
        tk3_choices = [32, 64, 128]
    elif N2 >= 64:
        tk3_choices = [32, 64]
    else:
        tk3_choices = [_next_pow2(N2)]

    # Hints — arch-aware
    if cc in ((9, 0), (10, 0), (12, 1)):
        occ_vals = (1, 2, 4)
        nctas_vals = (1, 2)
    else:
        occ_vals = (1, 2, 4)
        nctas_vals = (1,)

    # num_worker_warps — gated on cuTile >= 1.4
    nww_vals = (None, 4, 8) if _cutile_version() >= (1, 4) else (None,)

    # GROUP_M sweep — L2 cache locality. Full sweep: Triton's GROUP_M=16 swizzle
    # is what beats cuTile in the M=8192-32768 band, so this is the key axis.
    group_m_vals = (1, 8, 16)

    # Phase 1: tile x hints x GROUP_M (the big cross-product)
    for tm in tm_choices:
        for tn3 in tn3_choices:
            for tk1 in tk1_choices:
                for tk2 in tk2_choices:
                    for tk3 in tk3_choices:
                        for occ in occ_vals:
                            for nctas in nctas_vals:
                                for nww in nww_vals:
                                    for group_m in group_m_vals:
                                        yield SimpleNamespace(
                                            TM=tm, TN3=tn3,
                                            TK1=tk1, TK2=tk2, TK3=tk3,
                                            GROUP_M=group_m,
                                            occupancy=occ, num_ctas=nctas,
                                            num_worker_warps=nww,
                                            latency_x=4, latency_w1=4,
                                            latency_w2=4, latency_w3=4,
                                            use_tma=15,
                                        )


# ============================================================
# Autotune infrastructure (adasplash pattern)
# ============================================================

_tune_cache: dict = {}
_kernel_cache: dict = {}

# Persistent disk cache: full autotune runs once per (arch, shape), ever.
# Uses repo directory if writable, falls back to /work (Docker).
_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if os.access(_REPO_DIR, os.W_OK):
    _TUNE_CACHE_PATH = os.path.join(_REPO_DIR, ".cutile_tune_cache.json")
else:
    _TUNE_CACHE_PATH = os.path.join("/work", ".cutile_tune_cache.json")

def _load_disk_cache():
    try:
        import json
        with open(_TUNE_CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}

def _save_disk_cache(d):
    import json
    tmp = _TUNE_CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, _TUNE_CACHE_PATH)

_disk_cache = _load_disk_cache()

def _disk_key(cache_key):
    return "|".join(str(v) for v in cache_key)

def _cfg_to_dict(cfg):
    return {k: v for k, v in vars(cfg).items()}

def _cfg_from_dict(d):
    return SimpleNamespace(**d)

# Cached fast-path: after first autotune, precompute everything for zero-overhead launch
_launch_cache: dict = {}


def _build_kernel(cfg, store_acts=0):
    """Get or create a compiled kernel variant with the given hints (cached)."""
    key = (cfg.occupancy, getattr(cfg, "num_worker_warps", None),
           getattr(cfg, "num_ctas", 1), store_acts)
    if key not in _kernel_cache:
        hints = _kernel_hints(cfg)
        _kernel_cache[key] = fused_mlp_kernel.replace_hints(**hints)
    return _kernel_cache[key]


def _check_correctness(cfg, x, w1, w2, w3, ref, M, K1, N1, N2, N3):
    """Run the kernel with a config and check correctness against ref."""
    OUT_test = torch.empty((M, N3), device=x.device, dtype=x.dtype)
    num_sms = _get_num_sms()
    num_tiles = (M + cfg.TM - 1) // cfg.TM * ((N3 + cfg.TN3 - 1) // cfg.TN3)
    grid = (min(num_sms, num_tiles),)
    stream = torch.cuda.current_stream()

    kernel = _build_kernel(cfg)
    # Dummy fp32 tensors for H0/H1PRE (STORE_ACTS=0: stores are compiled out)
    h0_dummy = torch.empty(1, device=x.device, dtype=torch.float32)
    h1_dummy = torch.empty(1, device=x.device, dtype=torch.float32)
    ct.launch(stream, grid, kernel, (
        x, w1, w2, w3, OUT_test, h0_dummy, h1_dummy, M,
        K1, N1, N2, N3,
        cfg.TM, cfg.TN3, cfg.TK1, cfg.TK2, cfg.TK3, cfg.GROUP_M,
        cfg.latency_x, cfg.latency_w1, cfg.latency_w2, cfg.latency_w3,
        cfg.use_tma & 1, (cfg.use_tma >> 1) & 1,
        (cfg.use_tma >> 2) & 1, (cfg.use_tma >> 3) & 1,
        0,  # STORE_ACTS=0
    ))
    torch.cuda.synchronize()
    return torch.allclose(OUT_test, ref, atol=1e-2, rtol=1e-2)


def _autotune(x, w1, w2, w3, OUT, M, K1, N1, N2, N3):
    """Run 3-phase exhaustive autotune, validate correctness, cache best config.

    Phase 1: tile sizes x hints x GROUP_M — full cross
    Phase 2: latency sweep at best phase-1 config (OFAT)
    Phase 3: TMA mask sweep at best phase-2 config (OFAT)
    """
    cc = torch.cuda.get_device_capability()
    cache_key = (cc[0], cc[1], M, K1, N1, N2, N3)
    if cache_key in _tune_cache:
        return _tune_cache[cache_key]
    dk = _disk_key(cache_key)
    if dk in _disk_cache:
        cfg = _cfg_from_dict(_disk_cache[dk])
        _build_kernel(cfg)
        _tune_cache[cache_key] = cfg
        print(f"  [autotune] disk cache hit: TM={cfg.TM} TN3={cfg.TN3} TK1={cfg.TK1} occ={cfg.occupancy}")
        return cfg

    # Fast mode for testing
    if os.environ.get("MLP_FAST_AUTOTUNE", "0") == "1":
        # Shape-dependent heuristic based on exhaustive autotune + ablation results.
        #
        # RTX 4090 (sm_89, 128 SMs) — from full exhaustive autotune (648 configs/shape):
        #   M<=256:  TM=16, TN3=64,  occ=2, nww=None, gm=1  (6.0us)
        #   M=1024:  TM=16, TN3=128, occ=2, nww=4,    gm=1  (7.0us)
        #   M=4096:  TM=32, TN3=128, occ=2, nww=4,    gm=1  (11.4us, lat_x=2)
        #   M=8192:  TM=64, TN3=128, occ=4, nww=8,    gm=1  (16.0us, TK1=32, lat_w1=2)
        #   M=16384: TM=64, TN3=128, occ=1, nww=8,    gm=1  (25.7us, lat_w3=2)
        #   M=32768: TM=64, TN3=128, occ=1, nww=8,    gm=1  (44.1us)
        #   M=65536: TM=64, TN3=128, occ=1, nww=8,    gm=1  (76.5us)
        #
        # GB10 (sm_121, 48 SMs) — from full exhaustive autotune:
        #   M<=256:  TM=16, occ=1, nww=None, gm=8   (~8.6us)
        #   M=512:   TM=16, occ=2, nww=None, gm=8   (10.2us)
        #   M=1024:  TM=32, occ=2, nww=None, gm=16  (13.3us)
        #   M=2048:  TM=16, occ=1, nww=8,   gm=8   (16.4us)
        #   M=4096:  TM=32, occ=1, nww=8,   gm=16  (18.7us)
        #
        # nww=8 passes correctness at M>=2048 but fails at M<=256.
        if cc == (8, 9):
            # RTX 4090 (sm_89) — from full exhaustive autotune (648 configs/shape)
            if M <= 256:
                TM, TN3, occ, nww, gm = 16, 64, 2, None, 1
            elif M <= 1024:
                TM, TN3, occ, nww, gm = 16, 128, 2, 4, 1
            elif M <= 4096:
                TM, TN3, occ, nww, gm = 32, 128, 2, 4, 1
            elif M <= 8192:
                TM, TN3, occ, nww, gm = 64, 128, 4, 8, 1
            else:
                TM, TN3, occ, nww, gm = 64, 128, 1, 8, 1
            if TM > M:
                TM, TN3, occ, nww, gm = 16, 64, 2, None, 1
        else:
            # GB10 and other archs
            if M <= 256:
                TM, occ, nww, gm = 16, 1, None, 8
            elif M <= 512:
                TM, occ, nww, gm = 16, 2, None, 8
            elif M <= 1024:
                TM, occ, nww, gm = 32, 2, None, 16
            elif M <= 2048:
                TM, occ, nww, gm = 16, 1, 8, 8
            else:
                TM, occ, nww, gm = 32, 1, 8, 16
            TN3 = 128
            if TM > M:
                TM, occ, nww, gm = 16, 1, None, 8
                TN3 = 128

        # TK1 and latency — arch-aware per-shape overrides
        tk1 = min(64, K1)
        lat_x, lat_w1, lat_w2, lat_w3 = 4, 4, 4, 4
        if cc == (8, 9):
            if M <= 4096 and M > 1024:
                lat_x = 2  # M=4096: lat_x=2 wins (11.4us)
            elif M <= 8192:
                tk1 = min(32, K1)  # M=8192: TK1=32 wins (16.0us)
                lat_w1 = 2         # M=8192: lat_w1=2 wins
            elif M <= 16384:
                lat_w3 = 2         # M=16384: lat_w3=2 wins (25.7us)

        cfg = SimpleNamespace(
            TM=TM, TN3=TN3, TK1=tk1, TK2=min(128, N1),
            TK3=min(128, N2), GROUP_M=gm,
            occupancy=occ, num_ctas=1, num_worker_warps=nww,
            latency_x=lat_x, latency_w1=lat_w1,
            latency_w2=lat_w2, latency_w3=lat_w3,
            use_tma=15,
        )
        _build_kernel(cfg)
        _tune_cache[cache_key] = cfg
        return cfg

    stream = torch.cuda.current_stream()
    num_sms = _get_num_sms()
    import torch.nn.functional as F
    h1 = F.softplus(x @ w1)
    h2 = F.softplus(h1 @ w2)
    ref = h2 @ w3

    h0_dummy = torch.empty(1, device=x.device, dtype=torch.float32)
    h1_dummy = torch.empty(1, device=x.device, dtype=torch.float32)

    def _launch_args(cfg, out_tensor):
        return (
            x, w1, w2, w3, out_tensor, h0_dummy, h1_dummy, M,
            K1, N1, N2, N3,
            cfg.TM, cfg.TN3, cfg.TK1, cfg.TK2, cfg.TK3, cfg.GROUP_M,
            cfg.latency_x, cfg.latency_w1, cfg.latency_w2, cfg.latency_w3,
            cfg.use_tma & 1, (cfg.use_tma >> 1) & 1,
            (cfg.use_tma >> 2) & 1, (cfg.use_tma >> 3) & 1,
            0,  # STORE_ACTS=0 during autotune
        )

    def _grid_fn(cfg):
        return (min(num_sms, ct.cdiv(M, cfg.TM) * ct.cdiv(N3, cfg.TN3)),)

    def _run_search(space, label, keep=0):
        print(f"  [autotune {label}] {len(space)} configs...")
        with ct.compiler_timeout(5):
            result = exhaustive_search(
                space, stream, _grid_fn, fused_mlp_kernel,
                lambda cfg: _launch_args(cfg, OUT),
                lambda cfg: _kernel_hints(cfg),
                quiet=True,
                flush_l2=True,
            )
        # Sort by speed, pick fastest correct
        sorted_ok = sorted(result.successes, key=lambda m: m.mean_us)
        kept = []
        found = None
        for meas in sorted_ok:
            if _check_correctness(meas.config, x, w1, w2, w3, ref, M, K1, N1, N2, N3):
                if found is None:
                    found = meas
                    c = meas.config
                    print(f"  [autotune {label}] best: TM={c.TM} TN3={c.TN3} "
                          f"occ={c.occupancy} nctas={c.num_ctas} "
                          f"nww={getattr(c,'num_worker_warps',None)} "
                          f"lat=({c.latency_x},{c.latency_w1},{c.latency_w2},{c.latency_w3}) "
                          f"tma={c.use_tma} gm={c.GROUP_M} ({meas.mean_us:.1f}us)")
                if len(kept) < keep:
                    kept.append(meas)
        if found is None:
            if sorted_ok:
                print(f"  [autotune {label}] WARNING: fastest config failed correctness, "
                      f"using it anyway")
                return sorted_ok[0].config, sorted_ok[0].mean_us, []
            raise RuntimeError(f"Autotune {label}: no valid config found")
        return found.config, found.mean_us, kept

    # Phase 1: two-tier to keep full coverage without 3888 serial compiles.
    #  Tier A: all tile combos x GROUP_M at default hints (occ=1, nctas=1, nww=None)
    #  Tier B: top tile winners x full hint cross (occ x nctas x nww)
    tile_space = []
    seen_tiles = set()
    for cfg in _configs(K1, N1, N2, N3):
        tk = (cfg.TM, cfg.TN3, cfg.TK1, cfg.TK2, cfg.TK3, cfg.GROUP_M)
        if tk in seen_tiles:
            continue
        seen_tiles.add(tk)
        tile_space.append(SimpleNamespace(
            TM=cfg.TM, TN3=cfg.TN3, TK1=cfg.TK1, TK2=cfg.TK2, TK3=cfg.TK3,
            GROUP_M=cfg.GROUP_M, occupancy=1, num_ctas=1, num_worker_warps=None,
            latency_x=4, latency_w1=4, latency_w2=4, latency_w3=4,
            use_tma=15))
    best_cfg, best_us, tier_a = _run_search(tile_space, "phase1-tiles", keep=12)

    # Tier B: hint sweep on the top-12 tier-A winners (not just the single best —
    # hints interact with tile shape, so the #1 tile at default hints isn't
    # necessarily the #1 after hint tuning).
    hint_winners = []
    seen_hw = set()
    for meas in tier_a:
        c = meas.config
        key = (c.TM, c.TN3, c.TK1, c.TK2, c.TK3, c.GROUP_M)
        if key in seen_hw:
            continue
        seen_hw.add(key)
        hint_winners.append(c)

    phase1b_space = []
    occ_vals = (1, 2, 4)
    nctas_vals = (1, 2) if cc in ((9, 0), (10, 0), (12, 1)) else (1,)
    nww_vals = (None, 4, 8) if _cutile_version() >= (1, 4) else (None,)
    for base in hint_winners:
        for occ in occ_vals:
            for nctas in nctas_vals:
                for nww in nww_vals:
                    if (occ, nctas, nww) == (base.occupancy, base.num_ctas,
                                             getattr(base, 'num_worker_warps', None)):
                        continue
                    c = SimpleNamespace(**vars(base))
                    c.occupancy = occ
                    c.num_ctas = nctas
                    c.num_worker_warps = nww
                    phase1b_space.append(c)
    p1b_cfg, p1b_us, _ = _run_search(phase1b_space, "phase1-hints", keep=0)
    if p1b_us < best_us:
        best_cfg, best_us = p1b_cfg, p1b_us

    # Phase 2: per-load latency sweep at best tile+hints (OFAT per load)
    tma_masks = (15, 0, 14, 13, 11, 7)
    latencies = (1, 2, 4, 8)

    phase2_space = []
    for load_name in ("latency_x", "latency_w1", "latency_w2", "latency_w3"):
        for lat in latencies:
            if lat == getattr(best_cfg, load_name):
                continue
            c = SimpleNamespace(**vars(best_cfg))
            setattr(c, load_name, lat)
            phase2_space.append(c)

    if phase2_space:
        p2_cfg, p2_us, _ = _run_search(phase2_space, "phase2-latency")
        if p2_us < best_us:
            best_cfg = p2_cfg
            best_us = p2_us

    # Phase 3: per-load TMA sweep at best config (OFAT per load)
    # TMA bitmask: bit0=X, bit1=W1, bit2=W2, bit3=W3
    phase3_space = []
    for bit in range(4):
        current_val = (best_cfg.use_tma >> bit) & 1
        toggled = best_cfg.use_tma ^ (1 << bit)
        if toggled == best_cfg.use_tma:
            continue
        c = SimpleNamespace(**vars(best_cfg))
        c.use_tma = toggled
        phase3_space.append(c)

    # Also try full TMA off and all combinations
    for tma in tma_masks:
        if tma == best_cfg.use_tma:
            continue
        c = SimpleNamespace(**vars(best_cfg))
        c.use_tma = tma
        phase3_space.append(c)

    if phase3_space:
        p3_cfg, p3_us, _ = _run_search(phase3_space, "phase3-tma")
        if p3_us < best_us:
            best_cfg = p3_cfg
            best_us = p3_us

    print(f"  [autotune] FINAL best: TM={best_cfg.TM} TN3={best_cfg.TN3} "
          f"TK1={best_cfg.TK1} occ={best_cfg.occupancy} nctas={best_cfg.num_ctas} "
          f"nww={getattr(best_cfg,'num_worker_warps',None)} "
          f"lat=({best_cfg.latency_x},{best_cfg.latency_w1},"
          f"{best_cfg.latency_w2},{best_cfg.latency_w3}) "
          f"tma={best_cfg.use_tma} gm={best_cfg.GROUP_M} ({best_us:.1f}us)")

    _build_kernel(best_cfg)
    _tune_cache[cache_key] = best_cfg
    _disk_cache[dk] = _cfg_to_dict(best_cfg)

    # ---- Dual-variant autotune: check if training (STORE_ACTS=1) picks the
    # same config as inference. Time the training variant with the inference
    # winner vs a small set of alternatives (larger tiles offset the extra
    # fp32 store traffic). If a different config wins for training, record it
    # under a separate cache key; otherwise mark shared.
    cfg_train = _autotune_train_variant(
        best_cfg, x, w1, w2, w3, M, K1, N1, N2, N3, cc, num_sms, stream)
    _disk_cache[dk + "|train"] = _cfg_to_dict(cfg_train)
    _save_disk_cache(_disk_cache)
    same = all(getattr(cfg_train, k) == getattr(best_cfg, k)
               for k in ("TM", "TN3", "TK1", "TK2", "TK3", "GROUP_M",
                         "occupancy", "num_ctas", "num_worker_warps",
                         "latency_x", "latency_w1", "latency_w2", "latency_w3",
                         "use_tma"))
    print(f"  [autotune] train config {'SAME as inference' if same else 'DIFFERS'}")
    return best_cfg


def _autotune_train_variant(best_cfg, x, w1, w2, w3, M, K1, N1, N2, N3,
                            cc, num_sms, stream):
    """Time STORE_ACTS=1 with the inference winner vs nearby alternatives.

    Returns the config to use for the training forward. The search is small
    (the training variant differs only by extra fp32 store traffic, which is
    bandwidth-bound — only tile sizes that change SM occupancy of the
    persistent loop matter).
    """
    H0 = torch.empty((M, N1), device=x.device, dtype=torch.float32)
    H1PRE = torch.empty((M, N2), device=x.device, dtype=torch.float32)
    OUT_t = torch.empty((M, N3), device=x.device, dtype=x.dtype)

    def _train_args(cfg):
        return (
            x, w1, w2, w3, OUT_t, H0, H1PRE, M,
            K1, N1, N2, N3,
            cfg.TM, cfg.TN3, cfg.TK1, cfg.TK2, cfg.TK3, cfg.GROUP_M,
            cfg.latency_x, cfg.latency_w1, cfg.latency_w2, cfg.latency_w3,
            cfg.use_tma & 1, (cfg.use_tma >> 1) & 1,
            (cfg.use_tma >> 2) & 1, (cfg.use_tma >> 3) & 1,
            1,  # STORE_ACTS=1
        )

    def _grid_fn(cfg):
        return (min(num_sms, ct.cdiv(M, cfg.TM) * ct.cdiv(N3, cfg.TN3)),)

    # Candidates: inference winner + neighboring TM/occ/gm
    candidates = [best_cfg]
    for tm in (16, 32, 64):
        if tm == best_cfg.TM:
            continue
        c = SimpleNamespace(**vars(best_cfg))
        c.TM = tm
        candidates.append(c)
    for occ in (1, 2):
        if occ == best_cfg.occupancy:
            continue
        c = SimpleNamespace(**vars(best_cfg))
        c.occupancy = occ
        candidates.append(c)
    for gm in (1, 8, 16):
        if gm == best_cfg.GROUP_M:
            continue
        c = SimpleNamespace(**vars(best_cfg))
        c.GROUP_M = gm
        candidates.append(c)

    best_us = None
    winner = best_cfg
    for cfg in candidates:
        try:
            with ct.compiler_timeout(5):
                result = exhaustive_search(
                    [cfg], stream, _grid_fn, fused_mlp_kernel,
                    _train_args, lambda c: _kernel_hints(c),
                    quiet=True, flush_l2=True,
                )
            us = result.successes[0].mean_us if result.successes else float("inf")
        except Exception:
            us = float("inf")
        if best_us is None or us < best_us:
            best_us = us
            winner = cfg
    print(f"  [autotune train] best TM={winner.TM} occ={winner.occupancy} "
          f"gm={winner.GROUP_M} ({best_us:.1f}us vs inf-winner)")
    return winner


def fused_mlp_fwd_cutile(x, w1, w2, w3):
    """Inference forward: single kernel launch, no intermediate global memory.

    Autotunes on first call per shape, then caches the best config.
    Uses a cached fast-path launcher to minimize Python overhead.
    """
    M, K1 = x.shape
    N1 = w1.shape[1]
    N2 = w2.shape[1]
    N3 = w3.shape[1]

    # Cache key for the fast-path launcher
    cc = torch.cuda.get_device_capability()
    fast_key = (cc[0], cc[1], M, K1, N1, N2, N3)

    if fast_key in _launch_cache:
        # Fast path: precomputed kernel, grid, and args template
        cached = _launch_cache[fast_key]
        OUT = cached["out_alloc"]()
        stream = torch.cuda.current_stream()
        ct.launch(stream, cached["grid"], cached["kernel"],
                  cached["args_fn"](x, w1, w2, w3, OUT))
        return OUT

    # Slow path: first call — autotune and build cached launcher
    x = x.contiguous()
    w1 = w1.contiguous()
    w2 = w2.contiguous()
    w3 = w3.contiguous()

    OUT = torch.empty((M, N3), device=x.device, dtype=x.dtype)
    cfg = _autotune(x, w1, w2, w3, OUT, M, K1, N1, N2, N3)

    num_sms = _get_num_sms()
    num_tiles = (M + cfg.TM - 1) // cfg.TM * ((N3 + cfg.TN3 - 1) // cfg.TN3)
    grid = (min(num_sms, num_tiles),)
    kernel = _build_kernel(cfg)

    # Precompute static args (everything except tensor pointers)
    K1_c, N1_c, N2_c, N3_c = K1, N1, N2, N3
    TM_c = cfg.TM
    TN3_c = cfg.TN3
    TK1_c = cfg.TK1
    TK2_c = cfg.TK2
    TK3_c = cfg.TK3
    GM_c = cfg.GROUP_M
    LX_c = cfg.latency_x
    LW1_c = cfg.latency_w1
    LW2_c = cfg.latency_w2
    LW3_c = cfg.latency_w3
    TX_c = cfg.use_tma & 1
    TW1_c = (cfg.use_tma >> 1) & 1
    TW2_c = (cfg.use_tma >> 2) & 1
    TW3_c = (cfg.use_tma >> 3) & 1

    M_c = M

    # Dummy fp32 tensors for H0/H1PRE (STORE_ACTS=0: stores compiled out)
    h0_dummy = torch.empty(1, device=x.device, dtype=torch.float32)
    h1_dummy = torch.empty(1, device=x.device, dtype=torch.float32)

    # Pre-allocate args list template — only tensor pointers change per call
    _static_args = (
        M_c, K1_c, N1_c, N2_c, N3_c,
        TM_c, TN3_c, TK1_c, TK2_c, TK3_c, GM_c,
        LX_c, LW1_c, LW2_c, LW3_c,
        TX_c, TW1_c, TW2_c, TW3_c,
        0,  # STORE_ACTS=0
    )

    def args_fn(x, w1, w2, w3, out):
        return (x, w1, w2, w3, out, h0_dummy, h1_dummy) + _static_args

    def out_alloc():
        return torch.empty((M_c, N3_c), device=x.device, dtype=x.dtype)

    _launch_cache[fast_key] = {
        "grid": grid,
        "kernel": kernel,
        "args_fn": args_fn,
        "out_alloc": out_alloc,
    }

    # Launch on the first call with a FRESH output tensor — the OUT passed to
    # _autotune was trial-run on by exhaustive_search and contains garbage.
    stream = torch.cuda.current_stream()
    OUT = torch.empty((M, N3), device=x.device, dtype=x.dtype)
    ct.launch(stream, grid, kernel, args_fn(x, w1, w2, w3, OUT))
    return OUT


# ============================================================
# Unified entry point: inference (backward=False) and training (backward=True)
# ============================================================

class FusedMLPCutile(torch.autograd.Function):
    """Training path: forward stores fp32 activations (STORE_ACTS=1),
    backward is ONE fused megakernel launch (strategy B)."""

    @staticmethod
    def forward(ctx, x, w1, w2, w3):
        out, H0, H1PRE = _fwd_train(x, w1, w2, w3)
        ctx.save_for_backward(x, w1, w2, w3, H0, H1PRE)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, w1, w2, w3, H0, H1PRE = ctx.saved_tensors
        dx, dw1, dw2, dw3 = fused_mlp_bwd_cutile(dout, x, w1, w2, w3, H0, H1PRE)
        return dx, dw1, dw2, dw3


def _fwd_train(x, w1, w2, w3):
    """Training forward: unified kernel, STORE_ACTS=1, autotuned config.

    Uses the same autotuned config as inference when available (the configs
    are autotuned per variant; when they match, the variants share the entry).
    """
    M, K1 = x.shape
    N1 = w1.shape[1]
    N2 = w2.shape[1]
    N3 = w3.shape[1]

    x = x.contiguous()
    w1 = w1.contiguous()
    w2 = w2.contiguous()
    w3 = w3.contiguous()

    OUT = torch.empty((M, N3), device=x.device, dtype=x.dtype)
    H0 = torch.empty((M, N1), device=x.device, dtype=torch.float32)
    H1PRE = torch.empty((M, N2), device=x.device, dtype=torch.float32)

    # Autotune inference config (also triggers train-variant tuning on first call)
    _autotune(x, w1, w2, w3, OUT, M, K1, N1, N2, N3)

    # Use the training-specific config if the dual-variant autotune found a
    # different winner; otherwise fall back to the inference config.
    cc = torch.cuda.get_device_capability()
    dk = _disk_key((cc[0], cc[1], M, K1, N1, N2, N3))
    train_dk = dk + "|train"
    if train_dk in _disk_cache:
        cfg = _cfg_from_dict(_disk_cache[train_dk])
    else:
        cfg = _tune_cache[(cc[0], cc[1], M, K1, N1, N2, N3)]
    kernel = _build_kernel(cfg, store_acts=1)

    num_sms = _get_num_sms()
    num_tiles = (M + cfg.TM - 1) // cfg.TM * ((N3 + cfg.TN3 - 1) // cfg.TN3)
    grid = (min(num_sms, num_tiles),)
    stream = torch.cuda.current_stream()

    ct.launch(stream, grid, kernel, (
        x, w1, w2, w3, OUT, H0, H1PRE, M,
        K1, N1, N2, N3,
        cfg.TM, cfg.TN3, cfg.TK1, cfg.TK2, cfg.TK3, cfg.GROUP_M,
        cfg.latency_x, cfg.latency_w1, cfg.latency_w2, cfg.latency_w3,
        cfg.use_tma & 1, (cfg.use_tma >> 1) & 1,
        (cfg.use_tma >> 2) & 1, (cfg.use_tma >> 3) & 1,
        1,  # STORE_ACTS=1
    ))
    return OUT, H0, H1PRE


def fused_mlp(x, w1, w2, w3, backward=False, save_acts_path=None):
    """Single entry point for the fused cuTile MLP.

    backward=False (inference): autotuned forward, no activation storage.
    backward=True  (training):  forward stores fp32 H0/H1_PRE and wires the
                                fused backward via autograd.
    save_acts_path:  optional .pt path — with backward=True, also dumps the
                     intermediate activations (H0, H1_PRE, OUT) to disk.
    """
    if not backward:
        return fused_mlp_fwd_cutile(x, w1, w2, w3)

    out = FusedMLPCutile.apply(x, w1, w2, w3)

    if save_acts_path is not None:
        # Grab the activations saved on the autograd node
        ctx = out.grad_fn
        _x, _w1, _w2, _w3, H0, H1PRE = ctx.saved_tensors
        torch.save({"OUT": out.detach(), "H0": H0, "H1_PRE": H1PRE},
                   save_acts_path)
    return out


# Back-compat aliases
def fused_mlp_fwd_store_acts(x, w1, w2, w3):
    """Training forward (STORE_ACTS=1). Prefer fused_mlp(..., backward=True)."""
    return _fwd_train(x, w1, w2, w3)


def fused_mlp_cutile(x, w1, w2, w3):
    """Training path via autograd.Function. Prefer fused_mlp(..., backward=True)."""
    return FusedMLPCutile.apply(x, w1, w2, w3)


# ============================================================
# Backward kernels (fused, atomic weight-grad accumulation)
# ============================================================

def softplus_grad_tile(x):
    """sigmoid(x) = where(x > 20, 1, exp(x)/(1+exp(x)))."""
    ex = ct.exp(x)
    return ct.where(x > 20.0, 1.0, ex / (1.0 + ex))


@ct.kernel
def bwd_dgrad_kernel(
    X, W1, W2, W3, DOUT, DH1, DX,
    M,
    K1: ct.Constant[int], N1: ct.Constant[int], N2: ct.Constant[int], N3: ct.Constant[int],
    TM: ct.Constant[int], TK1: ct.Constant[int], TK2: ct.Constant[int], TK3: ct.Constant[int],
):
    """Persistent kernel: for each M-tile, compute dh1, dh2, dx."""
    raw_pid = ct.bid(0)
    num_pids = ct.num_blocks(0)
    num_m = ct.cdiv(M, TM)

    for pm in range(raw_pid, num_m, num_pids):
        # ---- Stage 1: dh1 = (dout @ W3^T) * sigmoid(h1) ----
        for k2 in range(ct.cdiv(N2, TK2)):
            dh1_acc = ct.zeros((TM, TK2), ct.float32)
            h1_acc = ct.zeros((TM, TK2), ct.float32)
            for k1 in range(ct.cdiv(K1, TK1)):
                x_t = ct.load(X, (pm, k1), shape=(TM, TK1), latency=4)
                w1_t = ct.load(W1, (k1, k2), shape=(TK1, TK2), latency=4)
                h1_acc = ct.mma(x_t, w1_t, h1_acc)
            for k3 in range(ct.cdiv(N3, TK3)):
                dout_t = ct.load(DOUT, (pm, k3), shape=(TM, TK3), latency=4)
                w3_t = ct.load(W3, (k2, k3), shape=(TK2, TK3), latency=4)
                w3_t = ct.transpose(w3_t)
                dh1_acc = ct.mma(dout_t, w3_t, dh1_acc)
            dh1 = dh1_acc * softplus_grad_tile(h1_acc)
            ct.store(DH1, (pm, k2), ct.astype(dh1, ct.float16))

        # ---- Stage 2: dx = dh2 @ W1^T where dh2 = (dh1 @ W2^T) * sigmoid(h0) ----
        for k2 in range(ct.cdiv(N1, TK2)):
            h0_acc = ct.zeros((TM, TK2), ct.float32)
            for kk in range(ct.cdiv(K1, TK1)):
                x_t = ct.load(X, (pm, kk), shape=(TM, TK1), latency=4)
                w1_t = ct.load(W1, (kk, k2), shape=(TK1, TK2), latency=4)
                h0_acc = ct.mma(x_t, w1_t, h0_acc)
            dh2_acc = ct.zeros((TM, TK2), ct.float32)
            for k2p in range(ct.cdiv(N2, TK2)):
                dh1_t = ct.load(DH1, (pm, k2p), shape=(TM, TK2), latency=4)
                w2_t = ct.load(W2, (k2, k2p), shape=(TK2, TK2), latency=4)
                w2_t = ct.transpose(w2_t)
                dh2_acc = ct.mma(dh1_t, w2_t, dh2_acc)
            dh2 = dh2_acc * softplus_grad_tile(h0_acc)
            dh2_f16 = ct.astype(dh2, ct.float16)
            for k1 in range(ct.cdiv(K1, TK1)):
                w1_t = ct.load(W1, (k1, k2), shape=(TK1, TK2), latency=4)
                w1_t = ct.transpose(w1_t)
                dx_acc = ct.zeros((TM, TK1), ct.float32)
                dx_acc = ct.mma(dh2_f16, w1_t, dx_acc)
                rows = ct.arange(TM, dtype=ct.int32) + pm * TM
                cols = ct.arange(TK1, dtype=ct.int32) + k1 * TK1
                row_idx = ct.reshape(rows, (TM, 1))
                col_idx = ct.reshape(cols, (1, TK1))
                ct.atomic_add(DX, (row_idx, col_idx), dx_acc)


@ct.kernel
def bwd_wgrad3_kernel(
    X, W1, W2, W3, DOUT, DW3,
    M,
    K1: ct.Constant[int], N1: ct.Constant[int], N2: ct.Constant[int], N3: ct.Constant[int],
    TM: ct.Constant[int], TK1: ct.Constant[int], TK2: ct.Constant[int], TK3: ct.Constant[int],
):
    """dW3 = H2^T @ dout, accumulated atomically over M-tiles."""
    raw_pid = ct.bid(0)
    num_pids = ct.num_blocks(0)
    num_k3 = ct.cdiv(N2, TK3)
    num_n3 = ct.cdiv(N3, TK3)
    num_m = ct.cdiv(M, TM)
    num_tiles = num_k3 * num_n3

    for tile_id in range(raw_pid, num_tiles, num_pids):
        pk3 = tile_id % num_k3
        pn3 = tile_id // num_k3
        dw3_acc = ct.zeros((TK3, TK3), ct.float32)
        for pm in range(num_m):
            h2_acc = ct.zeros((TM, TK3), ct.float32)
            for k2 in range(ct.cdiv(N1, TK2)):
                h1_acc = ct.zeros((TM, TK2), ct.float32)
                for k1 in range(ct.cdiv(K1, TK1)):
                    x_t = ct.load(X, (pm, k1), shape=(TM, TK1), latency=4)
                    w1_t = ct.load(W1, (k1, k2), shape=(TK1, TK2), latency=4)
                    h1_acc = ct.mma(x_t, w1_t, h1_acc)
                h1 = softplus_tile(h1_acc)
                h1_f16 = ct.astype(h1, ct.float16)
                w2_t = ct.load(W2, (k2, pk3), shape=(TK2, TK3), latency=4)
                h2_acc = ct.mma(h1_f16, w2_t, h2_acc)
            h2 = softplus_tile(h2_acc)
            h2_f16 = ct.astype(h2, ct.float16)
            h2_t = ct.transpose(h2_f16)
            dout_t = ct.load(DOUT, (pm, pn3), shape=(TM, TK3), latency=4)
            dw3_acc = ct.mma(h2_t, dout_t, dw3_acc)
        rows = ct.arange(TK3, dtype=ct.int32) + pk3 * TK3
        cols = ct.arange(TK3, dtype=ct.int32) + pn3 * TK3
        row_idx = ct.reshape(rows, (TK3, 1))
        col_idx = ct.reshape(cols, (1, TK3))
        ct.atomic_add(DW3, (row_idx, col_idx), dw3_acc)


@ct.kernel
def bwd_wgrad2_kernel(
    X, W1, DH1, DW2,
    M,
    K1: ct.Constant[int], N1: ct.Constant[int], N2: ct.Constant[int],
    TM: ct.Constant[int], TK1: ct.Constant[int], TK2: ct.Constant[int],
):
    """dW2 = H1^T @ dh1, accumulated atomically over M-tiles."""
    raw_pid = ct.bid(0)
    num_pids = ct.num_blocks(0)
    num_k2 = ct.cdiv(N1, TK2)
    num_k3 = ct.cdiv(N2, TK2)
    num_m = ct.cdiv(M, TM)
    num_tiles = num_k2 * num_k3

    for tile_id in range(raw_pid, num_tiles, num_pids):
        pk2 = tile_id % num_k2
        pk3 = tile_id // num_k2
        dw2_acc = ct.zeros((TK2, TK2), ct.float32)
        for pm in range(num_m):
            h1_acc = ct.zeros((TM, TK2), ct.float32)
            for k1 in range(ct.cdiv(K1, TK1)):
                x_t = ct.load(X, (pm, k1), shape=(TM, TK1), latency=4)
                w1_t = ct.load(W1, (k1, pk2), shape=(TK1, TK2), latency=4)
                h1_acc = ct.mma(x_t, w1_t, h1_acc)
            h1 = softplus_tile(h1_acc)
            h1_f16 = ct.astype(h1, ct.float16)
            h1_t = ct.transpose(h1_f16)
            dh1_t = ct.load(DH1, (pm, pk3), shape=(TM, TK2), latency=4)
            dw2_acc = ct.mma(h1_t, dh1_t, dw2_acc)
        rows = ct.arange(TK2, dtype=ct.int32) + pk2 * TK2
        cols = ct.arange(TK2, dtype=ct.int32) + pk3 * TK2
        row_idx = ct.reshape(rows, (TK2, 1))
        col_idx = ct.reshape(cols, (1, TK2))
        ct.atomic_add(DW2, (row_idx, col_idx), dw2_acc)


@ct.kernel
def bwd_wgrad1_kernel(
    X, W1, DH1, W2, DW1,
    M,
    K1: ct.Constant[int], N1: ct.Constant[int], N2: ct.Constant[int],
    TM: ct.Constant[int], TK1: ct.Constant[int], TK2: ct.Constant[int],
):
    """dW1 = X^T @ dh2, accumulated atomically over M-tiles."""
    raw_pid = ct.bid(0)
    num_pids = ct.num_blocks(0)
    num_k1 = ct.cdiv(K1, TK1)
    num_k2 = ct.cdiv(N1, TK2)
    num_m = ct.cdiv(M, TM)
    num_tiles = num_k1 * num_k2

    for tile_id in range(raw_pid, num_tiles, num_pids):
        pk1 = tile_id % num_k1
        pk2 = tile_id // num_k1
        dw1_acc = ct.zeros((TK1, TK2), ct.float32)
        for pm in range(num_m):
            h0_acc = ct.zeros((TM, TK2), ct.float32)
            for k1 in range(ct.cdiv(K1, TK1)):
                x_t = ct.load(X, (pm, k1), shape=(TM, TK1), latency=4)
                w1_t = ct.load(W1, (k1, pk2), shape=(TK1, TK2), latency=4)
                h0_acc = ct.mma(x_t, w1_t, h0_acc)
            dh2_acc = ct.zeros((TM, TK2), ct.float32)
            for k2 in range(ct.cdiv(N2, TK2)):
                dh1_t = ct.load(DH1, (pm, k2), shape=(TM, TK2), latency=4)
                w2_t = ct.load(W2, (pk2, k2), shape=(TK2, TK2), latency=4)
                w2_t = ct.transpose(w2_t)
                dh2_acc = ct.mma(dh1_t, w2_t, dh2_acc)
            dh2 = dh2_acc * softplus_grad_tile(h0_acc)
            dh2_f16 = ct.astype(dh2, ct.float16)
            x_t = ct.load(X, (pm, pk1), shape=(TM, TK1), latency=4)
            x_t = ct.transpose(x_t)
            dw1_acc = ct.mma(x_t, dh2_f16, dw1_acc)
        rows = ct.arange(TK1, dtype=ct.int32) + pk1 * TK1
        cols = ct.arange(TK2, dtype=ct.int32) + pk2 * TK2
        row_idx = ct.reshape(rows, (TK1, 1))
        col_idx = ct.reshape(cols, (1, TK2))
        ct.atomic_add(DW1, (row_idx, col_idx), dw1_acc)


@ct.kernel
def bwd_fused_kernel(
    X, W1, W2, W3, DOUT, H0, H1PRE, DX, DW1, DW2, DW3,
    M,
    K1: ct.Constant[int], N1: ct.Constant[int], N2: ct.Constant[int], N3: ct.Constant[int],
    TM: ct.Constant[int], TK1: ct.Constant[int], TN: ct.Constant[int],
):
    """ONE fused backward megakernel (strategy B).

    Contract: K1 = N1 = N2 = N3 = TN (one square weight tile per weight), and
    K1 % TK1 == 0. Activations are loaded from global fp32 H0 (M,N1) /
    H1_PRE (M,N2) stored by fwd_store_acts_kernel — NO recompute.

    Per M-tile pm (persistent round-robin):
      Phase A (input grads):
        dh1[m, :] = (dout @ W3^T) * sigmoid(H1_PRE)
        dh2[m, :] = (dh1 @ W2^T) * sigmoid(H0)
        dx[m, k1]  = dh2 @ W1^T           (atomic_add, fp32)
      Phase B (weight grads, accumulators held across all pm of this block):
        dW3 += softplus(H1_PRE)^T @ dout
        dW2 += softplus(H0)^T     @ dh1
        dW1 += X^T                @ dh2
    Weight-grad accumulators are flushed with one atomic_add at kernel end.
    """
    raw_pid = ct.bid(0)
    num_pids = ct.num_blocks(0)
    num_m = ct.cdiv(M, TM)
    num_n3 = ct.cdiv(N3, TN)
    num_k1 = ct.cdiv(K1, TK1)

    # Full weight-grad accumulators (held across all pm tiles of this block):
    # dW3 (N2,N3), dW2 (N1,N2), dW1 (K1,N1) — exactly (TN,TN) / (TN,TN) /
    # (TK1 * num_k1, TN) == (K1, TN) when K1=TN=TK1.
    dw3_acc = ct.zeros((TN, TN), ct.float32)
    dw2_acc = ct.zeros((TN, TN), ct.float32)
    dw1_acc = ct.zeros((TK1, TN), ct.float32)

    for pm in range(raw_pid, num_m, num_pids):
        # ---- dh1 = (dout @ W3^T) * sigmoid(H1_PRE) ----
        dh1_acc = ct.zeros((TM, TN), ct.float32)
        for k3 in range(num_n3):
            dout_t = ct.load(DOUT, (pm, k3), shape=(TM, TN), latency=4)
            w3_t = ct.load(W3, (0, k3), shape=(TN, TN), latency=4)
            w3_t = ct.transpose(w3_t)
            dh1_acc = ct.mma(dout_t, w3_t, dh1_acc)
        h1pre_t = ct.load(H1PRE, (pm, 0), shape=(TM, TN), latency=4)
        dh1 = dh1_acc * softplus_grad_tile(h1pre_t)   # fp32
        dh1_f16 = ct.astype(dh1, ct.float16)

        # ---- dW3 += softplus(H1_PRE)^T @ dout ----
        h1 = softplus_tile(h1pre_t)
        h1_f16 = ct.astype(h1, ct.float16)
        h1_t = ct.transpose(h1_f16)
        for pn3 in range(num_n3):
            dout_t = ct.load(DOUT, (pm, pn3), shape=(TM, TN), latency=4)
            dw3_acc = ct.mma(h1_t, dout_t, dw3_acc)

        # ---- dh2 = (dh1 @ W2^T) * sigmoid(H0) ----
        w2_t = ct.load(W2, (0, 0), shape=(TN, TN), latency=4)
        w2_t = ct.transpose(w2_t)
        dh2_acc = ct.zeros((TM, TN), ct.float32)
        dh2_acc = ct.mma(dh1_f16, w2_t, dh2_acc)
        h0_t = ct.load(H0, (pm, 0), shape=(TM, TN), latency=4)
        dh2 = dh2_acc * softplus_grad_tile(h0_t)      # fp32
        dh2_f16 = ct.astype(dh2, ct.float16)

        # ---- dW2 += softplus(H0)^T @ dh1 ----
        h0 = softplus_tile(h0_t)
        h0_f16 = ct.astype(h0, ct.float16)
        h0_t = ct.transpose(h0_f16)
        dw2_acc = ct.mma(h0_t, dh1_f16, dw2_acc)

        # ---- dW1 += X^T @ dh2 / dx = dh2 @ W1^T (TK1 chunks over K1) ----
        for k1t in range(num_k1):
            x_t = ct.load(X, (pm, k1t), shape=(TM, TK1), latency=4)
            x_tT = ct.transpose(x_t)
            dw1_acc = ct.mma(x_tT, dh2_f16, dw1_acc)

            dx_acc = ct.zeros((TM, TK1), ct.float32)
            w1_t = ct.load(W1, (k1t, 0), shape=(TK1, TN), latency=4)
            w1_t = ct.transpose(w1_t)
            dx_acc = ct.mma(dh2_f16, w1_t, dx_acc)
            # dx[pm, k1t]: each row tile is written by exactly one block
            # (pm tiles are partitioned round-robin), so a plain store works.
            ct.store(DX, (pm, k1t), dx_acc)

    # ---- flush weight grads (atomic_add, fp32) ----
    rows = ct.arange(TN, dtype=ct.int32)
    cols = ct.arange(TN, dtype=ct.int32)
    row_idx = ct.reshape(rows, (TN, 1))
    col_idx = ct.reshape(cols, (1, TN))
    ct.atomic_add(DW3, (row_idx, col_idx), dw3_acc)
    ct.atomic_add(DW2, (row_idx, col_idx), dw2_acc)

    rows1 = ct.arange(TK1, dtype=ct.int32)
    row_idx1 = ct.reshape(rows1, (TK1, 1))
    ct.atomic_add(DW1, (row_idx1, col_idx), dw1_acc)


def fused_mlp_bwd_cutile(dout, x, w1, w2, w3, H0=None, H1PRE=None):
    """Strategy B backward: ONE fused kernel launch. Returns (dx, dW1, dW2, dW3).

    Requires H0/H1_PRE fp32 activations from fused_mlp_fwd_store_acts.
    Contract: K1 == N1 == N2 == N3 (single-tile square weights).
    """
    assert H0 is not None and H1PRE is not None, \
        "strategy B needs stored activations H0/H1_PRE"
    M, K1 = x.shape
    N1 = w1.shape[1]
    N2 = w2.shape[1]
    N3 = w3.shape[1]
    assert K1 == N1 == N2 == N3, \
        f"bwd_fused_kernel requires square dims, got {K1},{N1},{N2},{N3}"

    dout = dout.contiguous()
    x = x.contiguous()
    w1 = w1.contiguous()
    w2 = w2.contiguous()
    w3 = w3.contiguous()
    H0 = H0.contiguous()
    H1PRE = H1PRE.contiguous()

    dx32 = torch.zeros((M, K1), device=x.device, dtype=torch.float32)
    dw1_32 = torch.zeros((K1, N1), device=x.device, dtype=torch.float32)
    dw2_32 = torch.zeros((N1, N2), device=x.device, dtype=torch.float32)
    dw3_32 = torch.zeros((N2, N3), device=x.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    num_sms = _get_num_sms()

    TM = 16 if M >= 16 else _next_pow2(M)  # TM=16 wins across M on GB10
    TN = K1  # single square tile per weight (dims 128 in this repo)
    TK1 = TN
    grid = (min(num_sms, (M + TM - 1) // TM),)

    # nww=8 + occ=1: 2-3x faster than default hints on GB10 (sweep in tune2.py)
    global _BWD_FUSED_K8
    if _BWD_FUSED_K8 is None:
        _BWD_FUSED_K8 = bwd_fused_kernel.replace_hints(occupancy=1,
                                                       num_worker_warps=8)
    kernel = _BWD_FUSED_K8

    # ONE launch: dh1, dh2, dx, dW1, dW2, dW3
    ct.launch(stream, grid, kernel, (
        x, w1, w2, w3, dout, H0, H1PRE, dx32, dw1_32, dw2_32, dw3_32, M,
        K1, N1, N2, N3, TM, TK1, TN,
    ))

    return (dx32.to(x.dtype), dw1_32.to(x.dtype),
            dw2_32.to(x.dtype), dw3_32.to(x.dtype))


if __name__ == "__main__":
    import sys
    import torch.nn.functional as F

    M = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    D = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    H = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    OUT_F = int(sys.argv[4]) if len(sys.argv) > 4 else 128

    x = torch.randn(M, D, dtype=torch.float16, device='cuda') * 0.02
    w1 = torch.randn(D, H, dtype=torch.float16, device='cuda') * 0.02
    w2 = torch.randn(H, H, dtype=torch.float16, device='cuda') * 0.02
    w3 = torch.randn(H, OUT_F, dtype=torch.float16, device='cuda') * 0.02

    h1 = F.softplus(x @ w1)
    h2 = F.softplus(h1 @ w2)
    ref = h2 @ w3

    out = fused_mlp_fwd_cutile(x, w1, w2, w3)
    torch.cuda.synchronize()

    max_diff = (out - ref).abs().max().item()
    cos = F.cosine_similarity(ref.flatten(), out.flatten(), dim=0).item()
    passed = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
    print(f"Shape: M={M}, D={D}, H={H}, OUT={OUT_F}")
    print(f"Forward: max diff={max_diff:.6f}, cos={cos:.6f} {'PASS' if passed else 'FAIL'}")

    # Test backward
    x.requires_grad_(True)
    w1.requires_grad_(True)
    w2.requires_grad_(True)
    w3.requires_grad_(True)

    out_ref = F.softplus(F.softplus(x @ w1) @ w2) @ w3
    out_ref.sum().backward()
    ref_dx, ref_dw1, ref_dw2, ref_dw3 = x.grad.clone(), w1.grad.clone(), w2.grad.clone(), w3.grad.clone()

    x.grad = w1.grad = w2.grad = w3.grad = None
    out_ct = fused_mlp_cutile(x, w1, w2, w3)
    out_ct.sum().backward()

    def check(name, got, ref):
        diff = (got - ref).abs().max().item()
        cos = F.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()
        ok = cos >= 0.99
        print(f"  {name}: diff={diff:.6f} cos={cos:.6f} {'PASS' if ok else 'FAIL'}")
        return ok

    print("Backward:")
    all_ok = True
    all_ok &= check("dx", x.grad, ref_dx)
    all_ok &= check("dW1", w1.grad, ref_dw1)
    all_ok &= check("dW2", w2.grad, ref_dw2)
    all_ok &= check("dW3", w3.grad, ref_dw3)
    print("ALL PASS" if (passed and all_ok) else "SOME FAILED")
