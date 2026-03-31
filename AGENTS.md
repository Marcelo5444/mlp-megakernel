# AGENTS.md — MLP Megakernel (Training + Weight Normalization)

Context file for AI agents working on this repo.

## What this repo is

Triton megakernels that fuse multi-layer MLPs with Softplus activations into
single GPU kernel launches. The core idea: **register-tiled recompute fusion**
— all GEMMs and activations happen in one kernel, with intermediate activations
living entirely in registers (never written to global memory).

This branch (`feature/weight-norm`) extends the base inference kernel with:
- Full forward + backward training support
- Weight normalization (fused Triton kernel)
- AMP / mixed-precision support with `torch.autograd.Function`
- 3-layer and 5-layer variants

Originated from [KernelBench](https://github.com/RightNow-AI/autokernel)
Problem L2_P900. This repo contains only the megakernel code and reports.

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Inference-only 3-layer megakernel |
| `feature/weight-norm` | **This branch.** Full training for 3-layer and 5-layer, weight normalization, AMP, benchmarks, reports |

## Architecture

```
3-layer: out = softplus(softplus(x @ W1) @ W2) @ W3
5-layer: out = softplus(softplus(softplus(softplus(x @ W1) @ W2) @ W3) @ W4) @ W5
```

Default dimensions: D=128, H=128. Batch sizes 256–65536. FP16 Tensor Cores
with FP32 accumulation. Activations stored in FP32 for training.

---

## File map

### Core kernels

| File | What it does |
|------|--------------|
| `kernel.py` | Base inference megakernel. `_fused_mlp_recompute` (Triton JIT), `_pick_config` (heuristic), `NUM_SMS`, `Model`/`ModelNew` |
| `fused_mlp.py` | **3-layer training.** Forward megakernel, backward megakernel, weight normalization kernels, `torch.autograd.Function`, `nn.Module` wrappers |
| `fused_mlp5.py` | **5-layer training.** Same architecture as `fused_mlp.py` but extended to 5 layers |

### Benchmarks & tools

| File | What it does |
|------|--------------|
| `bench_weightnorm.py` | Benchmark harness: Eager vs `torch.compile` vs Fused, forward/backward/full, across batch sizes |
| `profile_wn.py` | `torch.profiler` tracing for weight-norm variants |

### Reports

| File | Contents |
|------|----------|
| `mlp_fused.md` | Main optimization report — forward kernel design, backward kernel, heuristics, benchmarks |
| `weight_norm.md` | Weight normalization integration — 13 optimization approaches tested, bug chronicles, final benchmarks |
| `split_k_atomic_backward.md` | Atomic backward kernel design — split-K with `tl.atomic_add` for weight gradients |
| `generalizing_to_n_layers.md` | Analysis of scaling the megakernel to N layers |
| `comparison_tinycudann.md` | Performance comparison with NVIDIA tiny-cuda-nn |
| `deep_analysis_tinycudann.md` | Deep dive into tiny-cuda-nn internals and how our approach differs |

---

## How the kernels work

### Forward — inference (`kernel.py: _fused_mlp_recompute`)

Persistent kernel: `min(NUM_SMs, num_tiles)` programs iterate tiles
round-robin. Each tile computes a `[BM, BN3]` output block via nested loops:

```
for k3 (over N2, step BK3):      # h2 columns
  for k2 (over N1, step BK2):    # h1 columns
    for k1 (over K1, step BK1):  # input columns
      h1_chunk += X_tile @ W1_tile
    h1_chunk = softplus(h1_chunk)     # in-register
    h2_chunk += h1_chunk @ W2_tile
  h2_chunk = softplus(h2_chunk)       # in-register
  out_acc += h2_chunk @ W3_tile
store(OUT, out_acc)
```

### Forward — training (`fused_mlp.py: _fused_mlp_training_kernel`)

Same structure as inference but additionally **stores H1, H2 to global memory
in FP32** after each softplus. These saved activations are needed by the
backward pass for the sigmoid derivative.

FP32 storage is critical: storing in FP16 then reloading for backward causes
`fp32 → fp16 → fp32` roundtrip that loses precision in the sigmoid derivative
`σ(h) = 1/(1+exp(-h))`, leading to NaN gradients at large batch sizes.

### Forward — with weight normalization (`fused_mlp.py: _fused_mlp_wn_kernel`)

Fuses weight normalization into the forward kernel. Instead of loading
pre-computed `W`, loads `v` (direction) and applies `W = g * v * inv_norm`
inline before each `tl.dot`. The normalization factors (`g`, `inv_norm`) are
precomputed by a separate fused WN kernel (`_wn_fwd_kernel`) that normalizes
all weight matrices in a single launch.

### Backward — data gradients (`fused_mlp.py: _fused_mlp_bwd_full_kernel`)

Mirror of the forward: propagates gradients backward through layers using
the chain rule. Key operations per layer:

1. `dh = grad @ W^T` (data gradient via transposed weight GEMM)
2. `dz = dh * σ(h) * (1 - σ(h) + h·σ(h))` (softplus derivative, computed as
   `sigmoid(h) * dh` — this is why FP32 `h` is needed)
3. Passes `dz` to the next layer

### Backward — weight gradients (two strategies)

**Small batch (M ≤ threshold):** Atomic accumulation in the fused kernel.
Each thread block computes `tl.dot(tl.trans(h_tile), grad_tile)` and does
`tl.atomic_add` into a shared DW buffer. Single kernel launch.

- Threshold: 32768 for 3-layer, 16384 for 5-layer
- DW buffer is pre-allocated FP32 and zero'd before each backward

**Large batch (M > threshold):** cuBLAS fallback. Interleaves weight GEMMs
with data gradient computation for L2 cache reuse:

```python
dh2 = grad @ W3.T;    DW3 = H2.T @ grad        # layer 3
dz2 = sigmoid(h2)*dh2; DW2 = H1.T @ dz2         # layer 2
dz1 = sigmoid(h1)*dh1; DW1 = X.T @ dz1           # layer 1
```

When `return_fp32_dw=True` (for WN backward), the cuBLAS path forces FP32
GEMMs: `h.float().t() @ grad.float()` to prevent FP16 overflow in DW.

### Weight normalization backward

Two kernel variants:

1. `_wn_bwd_kernel`: standard — takes `dW`, `v`, `g`, `inv_norm` → `dv`, `dg`
2. `_wn_bwd_from_dw_T_kernel`: optimized — reads transposed DW directly from
   the atomic buffer with column-stride access, avoiding an explicit transpose

Both always output FP32 `dv`/`dg` to prevent overflow.

---

## Important classes and functions

### `fused_mlp.py` (3-layer)

| Symbol | Type | Purpose |
|--------|------|---------|
| `FusedMLPSoftplus` | `nn.Module` | Drop-in 3-layer MLP. Forward uses fused kernel, backward uses autograd |
| `FusedMLPSoftplusWN` | `nn.Module` | Same but with fused weight normalization |
| `FusedMLPSoftplusFunction` | `autograd.Function` | Forward/backward dispatch for plain variant |
| `FusedMLPSoftplusWNFunction` | `autograd.Function` | Forward/backward dispatch for WN variant |
| `_pick_config_bwd` | function | Backward heuristic (same pattern as `_pick_config`) |
| `_ATOMIC_M_THRESHOLD` | constant | 32768 — below this, use atomic backward; above, use cuBLAS |
| `_NoCast` | class | Wraps tensors to protect them from `torch.amp.custom_fwd(cast_inputs=...)` |

### `fused_mlp5.py` (5-layer)

Mirrors `fused_mlp.py` with `5` suffix. Key differences:
- `_ATOMIC_M_THRESHOLD = 16384` (lower due to more register pressure)
- `_pick_config_5`: BM capped at 64 (5 accumulator tiles approach 255-register limit)
- 5-level nested loops in forward, 5-level chain in backward

---

## Config heuristics

### Forward (`_pick_config` / `_pick_config_5`)

```
target = M * num_output_tiles / NUM_SMs
if target ≤ 24 → BM=16, warps=4    (high occupancy, small batches)
if target ≤ 48 → BM=32, warps=8    (balanced)
else           → BM=64, warps=8    (throughput, large batches)
FP32 mode: warps=4 throughout (register pressure)
```

BK values = `min(64 or 128, hidden_dim)`. Match hidden dims to avoid
recomputation redundancy.

### Backward (`_pick_config_bwd` / `_pick_config_bwd_5`)

Same pattern but BK directions are reversed (gradient flows backward).

---

## Critical bugs found & fixed (for future reference)

These are documented in detail in `weight_norm.md` Section 8.

### 1. `cast_inputs` corrupts FP32 buffers

`@torch.amp.custom_fwd(cast_inputs=torch.float16)` recursively casts ALL
tensor arguments, including FP32 buffers that must stay FP32. The atomic
backward kernel writes FP32 gradients into what it thinks is FP32 `dw_buf`,
but `cast_inputs` silently replaced it with an FP16 copy.

**Fix:** `_NoCast` wrapper class. PyTorch's `_cast` only traverses
pytree-registered types, so wrapping in a plain object hides it.

### 2. FP16 overflow in WN backward outputs

The WN backward kernels were storing `dv`/`dg` in the input dtype (FP16),
but gradient magnitudes can exceed 65504 (FP16 max).

**Fix:** Always store and allocate `dv`/`dg` as FP32.

### 3. FP16 overflow in cuBLAS DW path

For large M (cuBLAS fallback), `h.t() @ grad` was computed in FP16 even
when `return_fp32_dw=True`, causing overflow in DW values.

**Fix:** Explicit `.float()` cast: `h.float().t() @ grad.float()`.

### 4. FP32→FP16→FP32 activation roundtrip

Training forward stored activations in FP16 (matching input dtype), then
backward loaded them and needed FP32 for `sigmoid(h)`. The roundtrip lost
precision and caused NaN in sigmoid derivatives.

**Fix:** Store activations as FP32 in forward. Load FP32 in backward,
cast to FP16 only for `tl.dot` operands.

---

## Key constraints & gotchas

1. **Requires Tensor Core GPU** (sm_80+). `tl.dot` maps to HMMA.
2. **K1, N1, N2 must be `tl.constexpr`** — they control `static_range` loops.
3. **BK must divide hidden dims exactly** — no remainder handling.
4. **`NUM_SMS` computed at import** — requires CUDA device at import time.
5. **`_NoCast` is load-bearing** — removing it causes silent FP32 buffer
   corruption under AMP. Do not refactor without understanding the
   `cast_inputs` interaction.
6. **Atomic threshold is hardware-dependent** — 32K/16K works for A100.
   Different GPUs with different SM counts may need adjustment.
7. **Weight init scale matters** — `* 0.02` scaling prevents FP16 overflow
   in intermediate activations. Large init scales will NaN.

## Running

```bash
# Correctness test (3-layer, plain)
python -c "
import torch
from fused_mlp import FusedMLPSoftplus
m = FusedMLPSoftplus(128, 128, 128).cuda().half()
x = torch.randn(4096, 128, device='cuda', dtype=torch.float16)
out = m(x)
loss = out.sum()
loss.backward()
print('fwd ok, grad norm:', m.w1.grad.norm().item())
"

# Correctness test (3-layer, weight norm)
python -c "
import torch
from fused_mlp import FusedMLPSoftplusWN
m = FusedMLPSoftplusWN(128, 128, 128).cuda().half()
x = torch.randn(4096, 128, device='cuda', dtype=torch.float16)
out = m(x)
loss = out.sum()
loss.backward()
print('fwd ok, v1 grad norm:', m.v1.grad.norm().item())
"

# Benchmark
python bench_weightnorm.py
```

## Dependencies

- PyTorch >= 2.0
- Triton >= 2.1
- CUDA GPU with Tensor Core support (sm_80+)
