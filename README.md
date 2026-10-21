# MLP Megakernel

Triton megakernels that fuse entire multi-layer MLPs with Softplus activations into single GPU kernel launches. Intermediate activations live entirely in registers — zero global memory traffic between layers.

## Architecture

**3-layer:** `out = softplus(softplus(x @ W1) @ W2) @ W3`

**5-layer:** `out = softplus(softplus(softplus(softplus(x @ W1) @ W2) @ W3) @ W4) @ W5`

## What's here

| File | Description |
|------|-------------|
| `kernel.py` | Inference-only 3-layer Triton megakernel (register-tiled recompute fusion) |
| `cutile_mlp.py` | cuTile 3-layer megakernel with 3-phase autotuning + cached fast-path launcher |
| `profile.py` | Benchmark harness: cuTile vs Triton vs PyTorch (CUDA event timing) |
| `fused_mlp.py` | Full training 3-layer: forward + backward megakernels, `torch.autograd.Function`, weight normalization |
| `fused_mlp5.py` | Full training 5-layer: same architecture extended to 5 layers |
| `bench_weightnorm.py` | Benchmark harness comparing Eager, `torch.compile`, and Fused variants |

## Key ideas

- **Register-tiled recompute fusion**: all GEMMs + activations in ONE kernel. Nested loops tile each layer's GEMM, apply softplus in-register, and feed directly into the next layer's GEMM. No intermediate tensors ever hit global memory.

- **Persistent kernel**: launch `min(NUM_SMs, num_tiles)` programs that iterate tiles round-robin, amortizing launch overhead.

- **FP32 accumulation, FP16 Tensor Cores**: all `tl.dot` accumulate in fp32; conversion to fp16 only happens at tile boundaries for Tensor Core GEMMs.

- **FP32 activation storage for training**: forward pass stores activations in fp32 to avoid numerical loss in the backward sigmoid derivative. Backward loads fp32 activations and casts to fp16 only for GEMM operands.

- **Atomic backward for small batches**: below 32K batch size, weight gradients are accumulated via `tl.atomic_add` in a single fused kernel. Above that threshold, falls back to cuBLAS GEMMs interleaved with data-gradient computation for L2 cache reuse.

- **Fused weight normalization**: single Triton kernel normalizes all weight matrices (`W = g * v / ||v||`) in one launch, eliminating Python dispatch overhead that otherwise dominates at small matrix sizes.

## Reports

Detailed optimization journey and benchmark results:

- [`mlp_fused.md`](mlp_fused.md) — Main optimization report
- [`weight_norm.md`](weight_norm.md) — Weight normalization integration
- [`split_k_atomic_backward.md`](split_k_atomic_backward.md) — Atomic backward kernel design
- [`generalizing_to_n_layers.md`](generalizing_to_n_layers.md) — Scaling to N layers
- [`comparison_tinycudann.md`](comparison_tinycudann.md) — Comparison with tiny-cuda-nn
- [`deep_analysis_tinycudann.md`](deep_analysis_tinycudann.md) — Deep analysis of tiny-cuda-nn internals

## Requirements

- PyTorch >= 2.0
- Triton >= 2.1
- CUDA GPU with Tensor Core support (sm_80+)

For cuTile: CUDA 13.4+, cuTile 1.4+, run inside the `nvcr.io/nvidia/pytorch:26.09-py3`
(or `nvcr.io/nvidia/cuda-dl-base:26.09-cuda13.4-devel-ubuntu24.04`) Docker image —
multi-arch (arm64 for GB10, amd64 for x86_64).

## Quick start

```python
from fused_mlp import FusedMLPSoftplus, FusedMLPSoftplusWN

# Inference + training, 3-layer
model = FusedMLPSoftplus(D=128, H=128, out_features=128).cuda().half()
out = model(x)  # single kernel launch
loss = out.sum()
loss.backward()  # fused backward

# With weight normalization
model_wn = FusedMLPSoftplusWN(D=128, H=128, out_features=128).cuda().half()
out = model_wn(x)
```

## cuTile Implementation

`cutile_mlp.py` provides a cuTile (CUDA Tile) port of the 3-layer fused MLP
megakernel. It uses the same register-tiled recompute fusion architecture as
the Triton kernel — all GEMMs and softplus activations execute in a single
persistent kernel launch with intermediates in registers.

### Autotuning

Autotuning uses `ct.tune.exhaustive_search` with a 3-phase
search strategy:

- **Phase 1** (two-tier): tier A sweeps all tile combos (TM, TN3, TK1, TK2,
  TK3) at default hints; tier B sweeps the full hint cross (occupancy x
  num_ctas x num_worker_warps) on the tier-A winner plus alternative tile
  shapes. Keeps effective full coverage without 3888 serial compiles.
- **Phase 2** (12 configs): per-load latency OFAT sweep (latency_x, latency_w1,
  latency_w2, latency_w3) over (1,2,4,8) at the best phase-1 config.
- **Phase 3** (9 configs): per-load TMA mask sweep — toggle TMA on/off for
  each of X, W1, W2, W3 individually, plus full mask combinations.

GPU capability-aware tile caps (`_MAX_TILE` by sm_89/90/100/120/121),
`ct.compiler_timeout(5)` to cap compile time, and post-autotune correctness
validation (fastest config that passes `torch.allclose` vs PyTorch reference).
Winning configs persist to a disk cache (`.cutile_tune_cache.json`) keyed by
(arch, M, K1, N1, N2, N3) — full autotune runs once per shape, ever.

### Benchmark: cuTile vs Triton vs PyTorch

NVIDIA GB10 (sm_121, 48 SMs), D=H=OUT=128, FP16, FP32 accumulation.
CUDA event timing, 100 warmup + 200 iterations, median reported.

| M (batch) | cuTile (ms) | Triton (ms) | PyTorch (ms) | vs Triton | vs PyTorch |
|------------|-------------|-------------|--------------|-----------|------------|
| 64 | 0.0087 | 0.0089 | 0.0165 | 1.02x | 1.90x |
| 128 | 0.0088 | 0.0089 | 0.0130 | 1.01x | 1.47x |
| 256 | 0.0087 | 0.0090 | 0.0133 | 1.03x | 1.52x |
| 512 | 0.0093 | 0.0090 | 0.0151 | 0.97x | 1.63x |
| 1024 | 0.0092 | 0.0131 | 0.0198 | 1.42x | 2.15x |
| 2048 | 0.0152 | 0.0153 | 0.0267 | 1.00x | 1.76x |
| 4096 | 0.0192 | 0.0229 | 0.0397 | 1.19x | 2.07x |

cuTile wins or ties at every size except M=512 (0.97x, within measurement
noise). Best speedup at M=1024 (1.42x vs Triton, 2.15x vs PyTorch) where the
shape-dependent heuristic selects TM=32/occ=2, and at M=4096 (1.19x vs Triton,
2.07x vs PyTorch) where nww=8 (num_worker_warps) provides warp-specialized
parallelism.

#### NVIDIA GB10 (sm_121, 48 SMs) — Full M Sweep

Full autotune per shape (two-tier tile sweep incl. GROUP_M {1,8,16} + hint
cross + latency/TMA OFAT), D=H=OUT=128, FP16, FP32 accumulation. CUDA event
timing, 20 warmup + scaled iterations, mean reported. All configs from
`.cutile_tune_cache.json`.

| M (batch) | PyTorch (us) | Triton (us) | cuTile (us) | Triton speedup | cuTile speedup | cuTile vs Triton |
|-----------|--------------|-------------|-------------|----------------|----------------|------------------|
| 256 | 25.5 | 17.3 | 10.6 | 1.47x | 2.40x | 1.62x |
| 512 | 24.7 | 17.2 | 10.8 | 1.44x | 2.29x | 1.59x |
| 1024 | 24.9 | 17.5 | 11.3 | 1.42x | 2.20x | 1.55x |
| 2048 | 25.1 | 17.5 | 11.2 | 1.44x | 2.23x | 1.56x |
| 4096 | 30.8 | 18.2 | 16.6 | 1.69x | 1.85x | 1.09x |
| 8192 | 47.0 | 25.4 | 29.8 | 1.85x | 1.58x | 0.85x |
| 16384 | 78.3 | 47.2 | 46.5 | 1.66x | 1.68x | 1.01x |
| 32768 | 227.4 | 83.1 | 76.6 | 2.74x | 2.97x | 1.08x |
| 65536 | 554.4 | 193.5 | 155.8 | 2.87x | 3.56x | 1.24x |

cuTile beats PyTorch at every size (1.6x–3.6x) and beats Triton at every size
except M=8192 (0.85x). The GROUP_M sweep (1/8/16) + TK1=128 + TM=64 variants
were the key additions: they flipped M=16384 (0.84x → 1.01x) and M=32768
(0.86x → 1.08x) from losses to wins vs Triton — Triton's edge in that band
comes from its GROUP_M=16 L2 swizzle, and matching it recovers the gap.
M=8192 (25.4us Triton vs 29.8us cuTile) remains the one losing shape.

#### NVIDIA RTX 4090 (sm_89, 128 SMs) — CUDA 13.4, cuTile 1.7

Fast heuristic autotune (`MLP_FAST_AUTOTUNE=1`), D=H=OUT=128, FP16, FP32 accumulation.
CUDA event timing, 20 warmup + 50 iterations, median reported.

| M (batch) | cuTile (ms) | Triton (ms) | PyTorch (ms) | vs Triton | vs PyTorch |
|------------|-------------|-------------|--------------|-----------|------------|
| 64 | 0.0543 | 0.0562 | 0.0625 | 1.04x | 1.15x |
| 128 | 0.0379 | 0.0563 | 0.0625 | 1.49x | 1.65x |
| 256 | 0.0379 | 0.0555 | 0.0611 | 1.46x | 1.61x |
| 512 | 0.0358 | 0.0563 | 0.0635 | 1.57x | 1.77x |
| 1024 | 0.0369 | 0.0565 | 0.0625 | 1.53x | 1.69x |
| 2048 | 0.0336 | 0.0532 | 0.0597 | 1.58x | 1.78x |
| 4096 | 0.0366 | 0.0573 | 0.0604 | 1.57x | 1.65x |
| 8192 | 0.0369 | 0.0553 | 0.0594 | 1.50x | 1.61x |
| 16384 | 0.0451 | 0.0587 | 0.0562 | 1.30x | 1.25x |
| 32768 | 0.0603 | 0.0791 | 0.0663 | 1.31x | 1.10x |
| 65536 | 0.0911 | 0.1116 | 0.1034 | 1.23x | 1.13x |

cuTile wins at every size. At small M (<=256) the kernel is launch-overhead bound and all implementations are close. At M=128-8192 cuTile achieves 1.5-1.6x over Triton and 1.6-1.8x over PyTorch. At large M (>=16384) the advantage narrows but cuTile still wins (1.1-1.3x vs Triton, 1.1-1.3x vs PyTorch). The shape-dependent heuristic picks TM=16 TN3=64 occ=2 nww=None gm=1 for M<=256, TM=16 TN3=128 occ=2 nww=4 gm=1 for M=1024, TM=32 TN3=128 occ=2 nww=4 gm=1 for M=4096, and TM=64 TN3=128 occ=4 nww=8 gm=1 for M>=8192.

### Optimizations

- **Cached fast-path launcher**: after the first call per shape, precomputes
  the compiled kernel, grid, and args tuple — eliminates ~9us Python overhead
  per call (M=256: 17.2us → 8.6us, M=4096: 34us → 19.2us).
- **Shape-dependent heuristic** (`MLP_FAST_AUTOTUNE=1`): picks tile sizes,
  occupancy, and num_worker_warps based on M, tuned from exhaustive autotune
  results. nww=8 passes correctness at M>=2048 but fails at M<=256, so the
  heuristic gates it accordingly.
- **Expanded autotune search space**: TM up to 128, TN3 up to 256, GROUP_M
  sweep (4/8/16), TK1 includes 128, with GPU capability-aware tile caps.

### Running the benchmark

```bash
# Inside the CUDA 13.4 Docker container (multi-arch: arm64 for GB10, amd64 for x86_64):
docker run --rm --gpus all -v /path/to/mlp-megakernel:/work -w /work \
    nvcr.io/nvidia/pytorch:26.09-py3 python3 profile.py
# or
docker run --rm --gpus all -v /path/to/mlp-megakernel:/work -w /work \
    nvcr.io/nvidia/cuda-dl-base:26.09-cuda13.4-devel-ubuntu24.04 python3 profile.py

# Custom sizes:
docker run --rm --gpus all -v /path/to/mlp-megakernel:/work -w /work \
    nvcr.io/nvidia/pytorch:26.09-py3 python3 profile.py --sizes 64,256,1024 --warmup 50 --iters 100

# Fast autotune (skip sweep, use heuristic defaults):
MLP_FAST_AUTOTUNE=1 python3 profile.py --sizes 256
```

## Ablation Study

An ablation study was conducted to understand which optimizations actually
bring speedup in the cuTile megakernel. Full results in
[`ablation_report.md`](ablation_report.md).

### Methodology

24 individual parameter variants tested in isolation (OFAT) across 4 batch
sizes on the RTX 4090 (sm_89, 128 SMs), then 15 combined configurations tested
across 5 batch sizes. Each variant benchmarked with CUDA events (50 warmup +
100 iterations, median). Full exhaustive autotune (540 configs/shape) was
then run with CUDA graph replay to measure pure GPU kernel time.

### Key findings

- **M<=1024 is launch-overhead bound.** With 128 SMs and only 1-8 output
  tiles, all configs cluster at ~12.3us. No config beats the launch floor.
- **With CUDA graphs, cuTile and Triton tie at pure GPU time.** The previous
  1.5x advantage was entirely Python dispatch overhead (~20-37us per call).
  cuTile's advantage is its lower launch overhead, not faster kernel code.
- **Exhaustive autotune finds nww=4 (not nww=8) optimal for small M.** The
  fast heuristic was updated with per-arch configs from full autotune.
- **TN3=64 wins at M<=256.** Smaller output tiles create more parallelism
  when SM utilization is low.
- **Branchless softplus fails correctness.** `x + log(1 + exp(-|x|))` via
  `ct.abs` produces 0.06+ max diff — the `where(x>20, x, log(exp(x)+1))`
  formulation is numerically necessary.
- **TMA is critical.** Disabling TMA costs 9-20%. Disabling TMA on X only
  is fine, but disabling on weights hurts significantly.
- **GROUP_M has no effect at D=H=OUT=128.** With TN3=128 and N3=128, there's
  only 1 output tile column — the swizzle degenerates.
- **opt_level=3 (default) is essential.** opt_level=0 is 4x slower,
  opt_level=2 matches default.

### Benchmark: CUDA graph replay (pure GPU time)

RTX 4090 (sm_89, 128 SMs), D=H=OUT=128, FP16, FP32 accumulation.
Full exhaustive autotune (540 configs/shape, 3-phase search).
CUDA graph replay, 50 warmup + 100 iterations, median reported.

| M | cuTile (ms) | Triton (ms) | PyTorch (ms) | vs Triton | vs PyTorch |
|---|---|---|---|---|---|
| 64 | 0.0092 | 0.0092 | 0.0123 | 1.00x | 1.33x |
| 256 | 0.0092 | 0.0092 | 0.0123 | 1.00x | 1.33x |
| 1024 | 0.0307 | 0.0310 | 0.0348 | 1.01x | 1.13x |
| 4096 | 0.0113 | 0.0113 | 0.0195 | 1.00x | 1.73x |
| 8192 | 0.0155 | 0.0143 | 0.0246 | 0.92x | 1.58x |

With launch overhead included (non-graph):

| M | cuTile (ms) | Triton (ms) | PyTorch (ms) | cuTile overhead | Triton overhead |
|---|---|---|---|---|---|
| 64 | 0.0299 | 0.0471 | 0.0621 | 0.0206 | 0.0379 |
| 256 | 0.0297 | 0.0462 | 0.0625 | 0.0205 | 0.0370 |
| 1024 | 0.1107 | 0.1577 | 0.2058 | 0.0800 | 0.1267 |
| 4096 | 0.0328 | 0.0501 | 0.0666 | 0.0215 | 0.0388 |
| 8192 | 0.0358 | 0.0492 | 0.0642 | 0.0203 | 0.0348 |

cuTile's Python dispatch overhead is ~45% lower than Triton's (20-22us vs
35-38us). This is where cuTile wins in practice — the kernel itself is
comparable to Triton, but cuTile's `ct.launch` is significantly lighter than
Triton's Python launcher.

### Large batch sizes (M=8192–65536)

Full exhaustive autotune, CUDA graph replay, 50 warmup + 100 iterations, median.

| M | cuTile graph (ms) | Triton graph (ms) | PyTorch graph (ms) | vs Triton (graph) | vs PyTorch (graph) | Best cuTile config |
|---|---|---|---|---|---|---|
| 8192 | 0.0141 | 0.0140 | 0.0236 | 0.99x | 1.67x | TM=64 occ=4 nww=8 gm=4 |
| 16384 | 0.0237 | 0.0225 | 0.0338 | 0.95x | 1.43x | TM=64 occ=1 nww=8 gm=1 |
| 32768 | 0.0381 | 0.0399 | 0.0522 | 1.05x | 1.37x | TM=64 occ=4 nww=8 gm=1 |
| 65536 | 0.0696 | 0.0737 | 0.0901 | 1.06x | 1.29x | TM=64 occ=4 nww=8 gm=1 |

With launch overhead included (non-graph):

| M | cuTile (ms) | Triton (ms) | PyTorch (ms) | vs Triton | vs PyTorch |
|---|---|---|---|---|---|
| 8192 | 0.0335 | 0.0459 | 0.0584 | 1.37x | 1.74x |
| 16384 | 0.0430 | 0.0552 | 0.0554 | 1.28x | 1.29x |
| 32768 | 0.0574 | 0.0748 | 0.0674 | 1.30x | 1.17x |
| 65536 | 0.0891 | 0.1085 | 0.1044 | 1.22x | 1.17x |

Launch overhead is constant regardless of M: ~19us for cuTile, ~35us for
Triton. At small M this dominates total time; at M=65536 it's only ~22% of
cuTile's total.

cuTile's advantage scales with M: as the kernel becomes more compute-bound
(M>=32768), the exhaustive autotune finds configs that Triton's static
heuristic can't match. The crossover where cuTile starts winning on pure
GPU kernel time (not just launch overhead) is around M=32768. At M=65536
cuTile is 1.06x faster than Triton at the kernel level and 1.22x including
dispatch.

## License

MIT
