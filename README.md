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

NVIDIA GB10 (sm_121, 48 SMs), CUDA 13.4, cuTile 1.7, torch 2.14.1+cu130.
D=H=OUT=128, FP16, FP32 accumulation. `do_bench` with L2 flush, warmup=25,
rep=100 for forward; warmup=10, rep=50 for forward+backward. Fresh exhaustive
autotune per shape (empty cache), split-K backward kernels.

#### Forward pass (cut-inf = inference, cut-train = training with act storage)

| M | PyTorch (us) | torch.compile (us) | Triton (us) | cut-inf (us) | cut-train (us) | vs PyTorch | vs Triton |
|---|---|---|---|---|---|---|---|
| 256 | 16.2 | 17.0 | 10.1 | **8.5** | 8.4 | 1.91x | 1.19x |
| 512 | 21.4 | 18.7 | 10.4 | **9.2** | 10.7 | 2.31x | 1.13x |
| 1024 | 25.6 | 23.5 | 16.3 | **12.5** | 14.8 | 2.05x | 1.30x |
| 2048 | 35.6 | 35.5 | 20.9 | **16.5** | 18.9 | 2.15x | 1.27x |
| 4096 | 51.9 | 57.4 | 29.7 | **24.5** | 33.8 | 2.12x | 1.21x |
| 8192 | 89.0 | 89.3 | 47.3 | **37.5** | 56.6 | 2.37x | 1.26x |
| 16384 | 149.2 | 127.0 | 80.6 | **69.0** | 104.4 | 2.16x | 1.17x |
| 32768 | 279.4 | 244.4 | 126.8 | **113.7** | 183.9 | 2.46x | 1.12x |
| 65536 | 635.2 | 580.8 | 221.8 | **188.6** | 336.6 | 3.37x | 1.18x |

cut-inf beats PyTorch at every shape (1.9–3.4x) and beats Triton at every
shape (1.12–1.30x). cut-train costs 1.0–1.8x cut-inf (fp16 activation
storage overhead).

#### Forward + Backward pass

| M | PyTorch (us) | torch.compile (us) | cutile (us) | vs PyTorch | vs T.compile |
|---|---|---|---|---|---|
| 256 | 54.2 | 85.2 | **72.1** | 0.75x | 0.64x |
| 512 | 57.0 | 216.7 | **100.9** | 0.57x | 0.26x |
| 1024 | 78.4 | 69.1 | **131.2** | 0.60x | 1.14x |
| 2048 | 96.6 | 166.0 | **225.7** | 0.43x | 0.58x |
| 4096 | 133.1 | 184.6 | **248.4** | 0.54x | 0.72x |
| 8192 | 220.6 | 269.9 | **324.8** | 0.68x | 0.82x |
| 16384 | 432.8 | 411.1 | **480.5** | 0.90x | 1.05x |
| 32768 | 896.0 | 833.5 | **871.3** | 1.03x | 1.07x |
| 65536 | 2103.9 | 1929.8 | **1603.9** | **1.31x** | **1.09x** |

With the split-K backward, cuTile fwd+bwd **beats PyTorch eager at M=65536
by 1.31x** and also beats torch.compile there (1.09x). Small/mid M still
favor cuBLAS-based eager — the backward pays activation re-reads and wgrad
reduction that only amortize at large M.

#### Backward only (isolated backward pass, GB10)

Backward-only comparison: PyTorch manual chain rule (no autograd overhead)
vs torch.compile'd chain rule vs direct `fused_mlp_bwd_splitk` kernel call
with stored fp16 activations. `do_bench` with L2 flush, warmup=10, rep=50.

| M | PyTorch manual (us) | torch.compile (us) | cuTile split-K (us) | vs PyTorch | vs T.compile |
|---|---|---|---|---|---|
| 256 | 30.6 | 31.1 | **58.4** | 0.52x | 0.98x |
| 512 | 40.9 | 38.4 | **82.8** | 0.49x | 1.07x |
| 1024 | 60.1 | 53.8 | **130.4** | 0.46x | 1.12x |
| 2048 | 84.8 | 72.3 | **208.0** | 0.41x | 1.17x |
| 4096 | 125.5 | 104.0 | **238.7** | 0.53x | 1.21x |
| 8192 | 182.6 | 154.3 | **327.0** | 0.56x | 1.18x |
| 16384 | 360.3 | 233.3 | **479.2** | 0.75x | 1.54x |
| 32768 | 750.0 | 500.7 | **763.9** | 0.98x | 1.50x |
| 65536 | 1597.1 | 1224.2 | **1307.8** | **1.22x** | **1.30x** |

The cuTile split-K backward **beats PyTorch manual backward at M=65536**
(1.22x) and **beats torch.compile at M=65536** (1.30x). Near-parity with
PyTorch at M=32768 (0.98x). Small M is dominated by the fixed launch
overhead of the 3-kernel split-K sequence vs PyTorch's cuBLAS GEMMs.

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
random dout).

| M | PyTorch manual bwd (us) | split-K bwd (us) | old fused bwd (us) | split-K vs torch | split-K vs old |
|---|---|---|---|---|---|
| 256 | 37.0 | **35.7** | 84.1 | 1.04x | 2.4x |
| 512 | 42.7 | **39.3** | 90.2 | 1.09x | 2.3x |
| 1024 | 48.1 | **54.4** | 141.0 | 0.88x | 2.6x |
| 2048 | 54.6 | **95.5** | 153.9 | 0.57x | 1.6x |
| 4096 | 64.1 | **113.0** | 157.4 | 0.57x | 1.4x |
| 8192 | 81.7 | **139.1** | 177.1 | 0.59x | 1.3x |
| 16384 | 105.2 | **157.4** | 215.3 | 0.67x | 1.4x |
| 32768 | 187.8 | **233.5** | 291.9 | 0.80x | 1.3x |
| 65536 | 388.0 | **362.7** | 436.0 | 1.07x | 1.2x |

PyTorch manual bwd = chain rule with preallocated cuBLAS GEMMs (no autograd
overhead), same stored-activations + random-dout protocol. split-K beats
PyTorch bwd-only at M<=512 and M=65536; cuBLAS split-K wgrads still win at
mid M (2048-32768). Correctness gate (cosine >= 0.99 vs eager, all of
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
