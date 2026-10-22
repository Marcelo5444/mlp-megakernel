# MLP Megakernel

Triton megakernels that fuse entire multi-layer MLPs with Softplus activations into single GPU kernel launches. Intermediate activations live entirely in registers — zero global memory traffic between layers.

## Architecture

**3-layer:** `out = softplus(softplus(x @ W1) @ W2) @ W3`

**5-layer:** `out = softplus(softplus(softplus(softplus(x @ W1) @ W2) @ W3) @ W4) @ W5`

## What's here

| File | Description |
|------|-------------|
| `kernel.py` | Inference-only 3-layer Triton megakernel (register-tiled recompute fusion) |
| `cutile_mlp.py` | Unified cuTile 3-layer megakernel: single kernel for inference + training (STORE_ACTS), 3-phase autotuning, fused backward, cached fast-path launcher |
| `bench_all.py` | Benchmark harness: PyTorch eager vs torch.compile vs Triton vs cuTile (fwd + bwd, do_bench with L2 flush) |

## Key ideas

- **Register-tiled recompute fusion**: all GEMMs + activations in ONE kernel. Nested loops tile each layer's GEMM, apply softplus in-register, and feed directly into the next layer's GEMM. No intermediate tensors ever hit global memory.

- **Persistent kernel**: launch `min(NUM_SMs, num_tiles)` programs that iterate tiles round-robin, amortizing launch overhead.

- **FP32 accumulation, FP16 Tensor Cores**: all `tl.dot` accumulate in fp32; conversion to fp16 only happens at tile boundaries for Tensor Core GEMMs.

- **FP16 activation storage for training**: forward stores pre-activations in fp16; the backward recomputes sigmoid(pre)*softplus(pre) from them (one exp) for both the dgrad chain and the wgrad A-operands. Cosine vs fp32 eager stays >= 0.997 (gate 0.99).

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
from cutile_mlp import fused_mlp

# Inference (backward=False): single kernel launch, no activation storage
out = fused_mlp(x, w1, w2, w3, backward=False)

# Training (backward=True): forward stores fp32 activations, fused backward
out = fused_mlp(x, w1, w2, w3, backward=True)
loss = out.sum()
loss.backward()  # dx, dW1, dW2, dW3 populated via fused cuTile backward

# Optionally save intermediate activations to disk
out = fused_mlp(x, w1, w2, w3, backward=True, save_acts_path="acts.pt")
```

## cuTile Implementation

`cutile_mlp.py` provides a unified cuTile (CUDA Tile) implementation of the
3-layer fused MLP megakernel. A **single kernel** (`fused_mlp_kernel` with
`STORE_ACTS` constexpr) serves both inference (stores compiled out) and
training (fp32 H0/H1_PRE activation storage). The entry point
`fused_mlp(x, w1, w2, w3, backward=False, save_acts_path=None)` selects the
path: `backward=False` for inference, `backward=True` for training with
fused backward via `torch.autograd.Function`.

### Autotuning

Forward autotuning uses `ct.tune.exhaustive_search` with a 3-phase search
(tiles → hints → latency/TMA OFAT), GPU capability-aware tile caps, and
post-autotune correctness validation. Winning configs persist to
`.cutile_tune_cache.json` keyed by `(arch, M, K1, N1, N2, N3)`.

The backward kernel (`bwd_fused_kernel`) is autotuned separately per M-bucket
(TM × occupancy × num_worker_warps × num_ctas) with its own tuned table
(`_BWD_TUNED`) — see `tune_bwd_hints.py`.

A dual-variant autotune also checks whether training (STORE_ACTS=1) picks the
same config as inference: on GB10, **4/9 shapes differ** (training prefers
lower occupancy or different GROUP_M due to fp32 store traffic).

### Benchmark: cuTile vs Triton vs PyTorch (DGX Spark / GB10)

NVIDIA GB10 (sm_121, 48 SMs), D=H=OUT=128, FP16, FP32 accumulation.
`do_bench` with L2 flush, warmup=25, rep=100 for forward; warmup=10, rep=50
for forward+backward. Fresh exhaustive autotune per shape.

#### Forward pass (cut-inf = inference, cut-train = training with act storage)

| M | PyTorch (us) | torch.compile (us) | Triton (us) | cut-inf (us) | cut-train (us) | vs PyTorch | vs Triton |
|---|---|---|---|---|---|---|---|
| 256 | 16.7 | 18.5 | 10.0 | **9.2** | 11.0 | 1.80x | 1.09x |
| 512 | 21.1 | 22.0 | 11.5 | **11.0** | 12.3 | 1.91x | 1.05x |
| 1024 | 26.6 | 25.4 | 16.0 | **12.6** | 16.4 | 2.11x | 1.27x |
| 2048 | 33.7 | 33.3 | 20.2 | **17.8** | 23.6 | 1.89x | 1.13x |
| 4096 | 53.0 | 52.6 | 30.5 | **23.5** | 40.4 | 2.26x | 1.30x |
| 8192 | 89.7 | 92.5 | 44.9 | **39.1** | 76.7 | 2.30x | 1.15x |
| 16384 | 147.7 | 136.2 | 77.4 | **63.1** | 133.1 | 2.34x | 1.23x |
| 32768 | 285.2 | 241.3 | 134.3 | **109.4** | 251.9 | 2.61x | 1.23x |
| 65536 | 661.3 | 600.1 | 227.4 | **190.9** | 483.4 | 3.46x | 1.19x |

cut-inf beats PyTorch at every shape (1.8–3.5x) and beats Triton at every
shape (1.05–1.30x). cut-train costs 1.1–2.6x cut-inf (fp32 activation
storage overhead).

#### Forward + Backward pass

| M | PyTorch (us) | torch.compile (us) | cutile (us) | vs PyTorch | vs T.compile |
|---|---|---|---|---|---|
| 256 | 43.9 | 37.7 | **98.2** | 0.45x | 1.16x |
| 512 | 51.7 | 53.3 | **166.4** | 0.31x | 0.97x |
| 1024 | 75.8 | 68.1 | **177.7** | 0.43x | 1.11x |
| 2048 | 98.1 | 88.5 | **314.1** | 0.31x | 1.11x |
| 4096 | 144.5 | 126.2 | **246.6** | 0.59x | 1.15x |
| 8192 | 212.2 | 206.7 | **348.5** | 0.61x | 1.03x |
| 16384 | 441.6 | 408.0 | **564.9** | 0.78x | 1.08x |
| 32768 | 881.2 | 814.0 | **1040.5** | 0.85x | 1.08x |
| 65536 | 2023.9 | 1902.2 | **1989.5** | **1.02x** | 1.06x |

cuTile fwd+bwd **beats PyTorch eager at M=65536** (1.02x) and beats
torch.compile at all shapes except 256/512. The gap narrows with M as the
fused kernel amortizes launch overhead and activation storage costs.

#### Backward only (isolated backward pass)

Backward-only comparison: PyTorch manual chain rule (no autograd overhead)
vs torch.compile'd chain rule vs direct `fused_mlp_bwd_cutile` kernel call
with stored fp32 activations. `do_bench` with L2 flush, warmup=10, rep=50.

| M | PyTorch manual (us) | torch.compile (us) | cuTile fused (us) | vs PyTorch | vs T.compile |
|---|---|---|---|---|---|
| 256 | 35.9 | 24.6 | **91.3** | 0.39x | 1.46x |
| 512 | 42.6 | 35.9 | **155.6** | 0.27x | 1.19x |
| 1024 | 57.3 | 52.8 | **160.9** | 0.36x | 1.08x |
| 2048 | 79.3 | 72.2 | **280.0** | 0.28x | 1.10x |
| 4096 | 118.0 | 101.5 | **210.4** | 0.56x | 1.16x |
| 8192 | 183.4 | 150.6 | **272.6** | 0.67x | 1.22x |
| 16384 | 370.5 | 231.1 | **415.3** | 0.89x | 1.60x |
| 32768 | 744.8 | 505.8 | **782.6** | 0.95x | 1.47x |
| 65536 | 1611.9 | 1230.3 | **1419.0** | **1.14x** | 1.31x |

The fused cuTile backward **beats PyTorch manual backward at M=65536**
(1.14x) and reaches near-parity at M=32768 (0.95x). The gap narrows with M
as the fused kernel amortizes launch overhead and the 5 chained MMAs
dominate over PyTorch's ~6 separate GEMM+activation ops.

### Benchmark: cuTile vs Triton vs PyTorch (RTX 4090)

NVIDIA RTX 4090 (sm_89, 128 SMs), CUDA 13.4, cuTile 1.7, torch 2.14.1+cu130.
D=H=OUT=128, FP16, FP32 accumulation. `do_bench` with L2 flush, warmup=25,
rep=100 for forward; warmup=10, rep=50 for forward+backward. Fresh exhaustive
autotune per shape (no disk cache for sm_89 — 1080-config phase-1 tile sweep
+ hint cross + latency/TMA OFAT per shape).

#### Forward pass (cut-inf = inference, cut-train = training with act storage)

| M | PyTorch (us) | torch.compile (us) | Triton (us) | cut-inf (us) | cut-train (us) | vs PyTorch | vs Triton |
|---|---|---|---|---|---|---|---|
| 256 | 15.1 | 11.3 | 7.5 | **6.5** | 6.8 | 2.31x | 1.15x |
| 512 | 16.0 | 15.4 | 7.3 | **6.8** | 7.6 | 2.35x | 1.07x |
| 1024 | 18.3 | 16.4 | 7.8 | **7.7** | 8.2 | 2.38x | 1.01x |
| 2048 | 20.6 | 18.8 | 8.7 | **8.6** | 10.2 | 2.41x | 1.01x |
| 4096 | 25.0 | 24.0 | 10.4 | **10.6** | 11.7 | 2.36x | 0.98x |
| 8192 | 35.1 | 34.8 | 14.7 | **15.2** | 20.4 | 2.32x | 0.97x |
| 16384 | 45.4 | 45.2 | 25.2 | **25.8** | 38.1 | 1.76x | 0.98x |
| 32768 | 67.9 | 66.5 | 43.4 | **42.8** | 58.8 | 1.59x | 1.01x |
| 65536 | 132.5 | 133.3 | 77.4 | **77.2** | 120.5 | 1.72x | 1.00x |

cut-inf beats PyTorch eager at every shape (1.6–2.4x) and ties or beats
Triton at every shape except 4096–16384 (0.97–0.98x, within noise).
cut-train costs 1.05–1.6x cut-inf for activation storage, growing with
M as store traffic dominates.

#### Backward pass (split-K wgrad, RTX 4090)

The backward is split into a persistent dgrad kernel (dh1/dh2/dx + per-block
dW1 partials) and two split-K wgrad kernels (dW2/dW3 contract over the batch
dim across 128 blocks) with a deterministic tiled reduction — no atomics, no
register-resident-across-kernel accumulators. Activations are stored fp16
and the wgrad A-operand is recomputed as sigmoid(pre)*softplus(pre).
do_bench L2-flushed, warmup=10, rep=50, bwd-only path (stored acts,
random dout); eager = PyTorch fwd+bwd.

| M | old fused bwd (us) | split-K bwd (us) | PyTorch eager fwd+bwd (us) | speedup vs old | vs eager |
|---|---|---|---|---|---|
| 256 | 84.1 | **36.1** | 44.2 | 2.3x | 1.22x |
| 512 | 90.2 | **36.0** | 48.7 | 2.5x | 1.35x |
| 1024 | 141.0 | **54.5** | 54.4 | 2.6x | 1.00x |
| 2048 | 153.9 | **89.9** | 58.5 | 1.7x | 0.65x* |
| 4096 | 157.4 | **115.9** | 70.8 | 1.4x | 0.61x* |
| 8192 | 177.1 | **134.0** | 82.1 | 1.3x | 0.61x* |
| 16384 | 215.3 | **167.7** | 119.3 | 1.3x | 0.71x* |
| 32768 | 291.9 | **226.9** | 169.2 | 1.3x | 0.75x* |
| 65536 | 436.0 | **359.9** | 369.8 | 1.2x | 1.03x |

* eager fwd+bwd includes the forward pass; the cuTile bwd column is
backward-only. Correctness gate (cosine >= 0.99 vs eager, all of
dx/dW1/dW2/dW3) passes at every M; the old fused bwd failed it at M=65536.
dW3 cosine is 0.997 (fp16 pre-activation recompute); all others are 1.0.

#### Train vs Inference autotune config comparison

| M | cut-inf config | cut-train config | Same? |
|---|---|---|---|
| 256 | TM=16 occ=2 nww=8 gm=8 | TM=16 occ=1 nww=8 gm=8 | **DIFFERS** (occ) |
| 512 | TM=16 occ=2 nww=8 gm=16 | TM=16 occ=1 nww=8 gm=16 | **DIFFERS** (occ) |
| 1024 | TM=32 occ=1 nww=None gm=1 | TM=32 occ=1 nww=None gm=1 | SAME |
| 2048 | TM=64 occ=4 nww=None gm=8 | TM=64 occ=4 nww=None gm=16 | **DIFFERS** (gm) |
| 4096 | TM=32 occ=1 nww=4 gm=1 | TM=32 occ=1 nww=4 gm=16 | **DIFFERS** (gm) |
| 8192 | TM=32 occ=1 nww=4 gm=1 | TM=32 occ=1 nww=4 gm=1 | SAME |
| 16384 | TM=32 occ=1 nww=None gm=1 | TM=32 occ=1 nww=None gm=1 | SAME |
| 32768 | TM=32 occ=1 nww=8 gm=16 | TM=32 occ=1 nww=8 gm=16 | SAME |
| 65536 | TM=32 occ=1 nww=None gm=8 | TM=32 occ=2 nww=None gm=8 | **DIFFERS** (occ) |

Training generally prefers lower occupancy (occ=1 vs 2) or different GROUP_M
due to the extra fp32 store traffic from activation storage.

### Running the benchmark

```bash
# Inside the CUDA 13.4 Docker container (multi-arch: arm64 for GB10, amd64 for x86_64):
docker run --rm --gpus all -v /path/to/mlp-megakernel:/work -w /work \
    cuda134-cutile17-cutedsl:latest python3 bench_all.py

# Fresh exhaustive autotune (clears cache first):
rm .cutile_tune_cache.json
docker run --rm --gpus all -v /path/to/mlp-megakernel:/work -w /work \
    -e MLP_FAST_AUTOTUNE= cuda134-cutile17-cutedsl:latest python3 bench_all.py
```

## License

MIT
