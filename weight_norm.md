# Weight Normalization for Fused MLP: Design & Benchmarks

## Overview

Weight normalization ([Salimans & Kingma 2016](https://arxiv.org/abs/1602.07868))
reparameterizes each weight matrix `W` as:

```
W = g * v / ||v||
```

where `v ∈ R^{out × in}` is the direction matrix, `g ∈ R^{out × 1}` is a
per-output-neuron scalar magnitude, and `||v||` is the L2 norm computed per
row. This decouples the magnitude from the direction, often improving
optimization dynamics.

The challenge: adding weight normalization to our fused megakernel MLP
introduces normalization overhead on every forward and backward call. This
document describes the optimization journey from naive implementation to a
fully-fused solution.

---

## 1. The Problem: Dispatch Overhead Dominates

For 128×128 weight matrices (D=H=128 in our KernelBench problem), the
normalization itself is trivially fast (~16K FLOPs, <1μs on GPU). The
bottleneck is entirely **Python-level dispatch overhead**:

- Each `torch` call (norm, clamp, reciprocal, multiply) launches a CUDA kernel
  at ~10–15μs per launch from Python
- For 3 weights × 4 ops = 12 kernel launches = ~120–180μs overhead
- For 5 weights × 4 ops = 20 kernel launches = ~200–300μs overhead

For context, the base fused megakernel takes ~67μs (3-layer) or ~93μs
(5-layer). The naive WN overhead exceeds the MLP compute itself.

---

## 2. Approach 1: torch.compile

First attempt: wrap the weight normalization in `torch.compile`:

```python
@torch.compile
def _weight_norm_3(v1, g1, v2, g2, v3, g3):
    w1 = g1 * v1 / v1.norm(dim=1, keepdim=True)
    w2 = g2 * v2 / v2.norm(dim=1, keepdim=True)
    w3 = g3 * v3 / v3.norm(dim=1, keepdim=True)
    return w1, w2, w3
```

**Result**: torch.compile with `max-autotune` generates efficient fused kernels
for the normalization (~80μs for all 3 weights), but the compiled graph has a
fixed dispatch floor. Combined with the megakernel, total inference:

| Approach | 3L Inference | 5L Inference |
|---|---|---|
| Compile (norm) + Fused (MLP) | ~0.150ms | ~0.175ms |
| torch.compile (everything) | ~0.085ms | ~0.080ms |

torch.compile wins because it fuses norm + GEMM + activation into a single
optimized graph. Our two-stage approach (compile norm → fused MLP) has double
the graph dispatch overhead.

---

## 3. Approach 2: Custom Triton Kernels

### Forward: `_wn_fwd_kernel`

A single Triton kernel computes `W[row] = g[row] * v[row] / ||v[row]||` and
stores `inv_norm[row] = 1/||v[row]||` for backward. One thread block per row.

```python
@triton.jit
def _wn_fwd_kernel(V, G, W, INVNORM, rows, cols, ...):
    row = tl.program_id(0)
    v = tl.load(V + row * stride + offs)
    norm_sq = tl.sum(v * v)
    inv_n = 1.0 / tl.sqrt(norm_sq + 1e-12)
    w = tl.load(G + row) * v * inv_n
    tl.store(W + row * stride + offs, w)
    tl.store(INVNORM + row, inv_n)
```

### Backward: `_wn_bwd_kernel`

Converts `dW` (gradient w.r.t. normalized weight) into `dv` and `dg`:

```
v_hat = v * inv_norm
dg = sum(dW * v_hat, dim=1)        # scalar per output neuron
dv = (g * inv_norm) * (dW - v_hat * dg)
```

Critical detail: `dW` arrives as `.t()` views (non-contiguous) from the
backward megakernel. The kernel must receive contiguous input, so
`.contiguous()` is called before launching. This was the source of a correctness
bug that caused completely wrong gradients before being fixed.

### Batching: Single Launch Per Direction

Instead of launching 3 (or 5) separate kernels, all weight matrices are
processed in a single kernel launch by concatenating them along the row
dimension:

```
v_cat = [v1; v2; v3]   → (384, 128) for 3 layers
g_cat = [g1; g2; g3]   → (384,)
```

One kernel launch processes all 384 rows. Results are sliced back into
individual weight views (zero-copy).

---

## 4. Approach 3: Contiguous Parameter Storage

### The torch.cat Problem

Even with batched kernels, `torch.cat` allocates new memory on every forward
call (~10–15μs for 384×128 fp16 tensor). For training at high frequency, this
allocation overhead accumulates.

### Solution: Store All Parameters as Single Blocks

```python
class FusedMLPSoftplusWN(nn.Module):
    def __init__(self, D, H, O):
        total = H + H + O  # 3 weight matrices stacked
        self.v_all = nn.Parameter(torch.randn(total, H))
        self.g_all = nn.Parameter(torch.ones(total, 1))
        self.register_buffer('_w_buf', torch.empty(total, H, dtype=torch.float16))
        self.register_buffer('_invn_buf', torch.empty(total, 1, dtype=torch.float16))
```

The `_w_buf` and `_invn_buf` are pre-allocated once at initialization.
Total forward with this approach: 2 kernel launches, 0 allocations.

---

## 5. Approach 4: Fused WN Megakernel (Final)

### Eliminating the Separate WN Kernel Launch

The contiguous approach still requires **2 kernel launches**: one WN kernel
(to compute `W = g*v/||v||`) and one MLP megakernel. At small batch sizes
(M ≤ 40K), kernel launch overhead (~15μs each) dominates. Can we merge them?

### Key Insight: Compute Norms Inside the Megakernel

The megakernel already loads weight tiles for the GEMM. Instead of loading
pre-computed W, it can load V tiles, compute the norm on the fly, and apply
`scale = g * inv_norm` per tile:

- **W2–W5** (BK = hidden_dim = 128): the full weight matrix fits in a single
  tile, so `norm² = sum(v²)` over axis=0 gives the complete row-wise norm.
  Single-pass, zero extra loads.

- **W1** (BK1 = 64 < K1 = 128): two-pass approach. First iterate the k1 loop
  to accumulate `norm_sq`, then iterate again for the actual GEMM with scaling.
  V1 tiles are served from L1 cache on the second pass (128×128 = 32KB fits
  easily in L1).

### Dynamic Dispatch

At small M (≤ 40K), the fused kernel saves ~34μs by eliminating the WN launch.
At large M (> 40K), the per-tile scaling overhead (extra FMAs + redundant norm
recomputation across M-tiles) exceeds the launch savings. The autograd function
dispatches dynamically:

```python
if M <= 40960:
    out, h1, h2 = _fused_mlp_wn_training_fwd(x, v_all, g_all, ...)  # 1 launch
else:
    _wn_fwd_contiguous(v_all, g_all, w_buf, invn_buf)                # 1 launch
    out, h1, h2 = _fused_mlp_training_fwd(x, w1, w2, w3)            # 1 launch
```

Total forward at M ≤ 40K: **1 kernel launch**.
Total forward at M > 40K: **2 kernel launches** (same as before).

### Zero-Copy WN Backward

The initial backward path had **9 kernel launches** for 3-layer (13 for
5-layer):

```
megakernel → DW1,DW2,DW3 (fp32)
→ 3× .to(fp16)                    # 3 cast kernels
→ 3× .t().contiguous()            # 3 copy kernels
→ torch.cat([...])                 # 1 concat kernel
→ _wn_bwd_contiguous              # 1 WN kernel
─────────────────────────────────
Total: 9 kernel launches + allocations
```

At ~15μs per Python-dispatched launch, this added ~105μs of pure overhead —
enough to lose to torch.compile at small N where the megakernel itself only
takes ~200μs.

**Solution**: `_wn_bwd_from_dw_T_kernel` reads DW directly in (K,N) fp32
layout using column-stride access. Since 128×128 × 4B = 64KB fits entirely
in L1 cache, the strided reads have negligible overhead. Each program handles
one output neuron row, dispatching to the correct DW buffer via split indices.

```
megakernel (return_fp32_dw=True) → DW1,DW2,DW3 (fp32, no cast)
→ _wn_bwd_from_dw_transposed      # 1 kernel, reads fp32 column-wise
─────────────────────────────────
Total: 2 kernel launches, 0 copies
```

This eliminates 7 kernel launches → saving ~105μs per backward call.

Total backward (all M): **2 kernel launches** (MLP bwd + WN bwd).

### Pre-Allocated DW Buffer

The atomic backward path (`M ≤ 32K`) originally allocated 3 (or 5) separate
`torch.zeros` tensors for DW on every backward call. Each `torch.zeros`
dispatches a CUDA memset from Python (~10–15μs each). For 3-layer: 3 calls =
~30–45μs of pure allocation overhead.

**Solution**: pre-allocate a single flat fp32 buffer (`_dw_buf`) in the module
at initialization. On each backward call, one `.zero_()` memsets the entire
buffer, then DW views are sliced out (zero-copy views).

```python
class FusedMLPSoftplusWN(nn.Module):
    def __init__(self, D, H, O):
        ...
        total_dw = D*H + H*H + H*O
        self.register_buffer('_dw_buf', torch.zeros(total_dw, dtype=torch.float32))
```

**Impact**: backward-only A/B test at M=4096 (3-layer):
- Before (3× `torch.zeros`): **0.190ms**
- After (1× `.zero_()` on pre-allocated): **0.137ms**
- Savings: **53μs (28%)** per backward call

The savings scale with the number of DW buffers — 5-layer saves even more
(4 fewer `torch.zeros` calls → ~70–100μs).

**Critical bug found**: `@torch.amp.custom_fwd(cast_inputs=torch.float16)`
recursively casts ALL tensor arguments to fp16 — including `dw_buf`. This
silently corrupted the fp32 buffer into fp16, causing the atomic backward
kernel to accumulate into wrong-dtype memory. For M ≥ 32K where DW values
exceed fp16 max (65504), this produced Inf/NaN in gradients. The fix: wrap
`dw_buf` in a `_NoCast` object (a custom class that PyTorch's pytree traversal
doesn't enter) before passing through the autograd Function.

### Approach 5: FP32 Activation Storage (Eliminating Precision Roundtrip)

The forward kernel computes activations h1, h2, ... in fp32 (register
arithmetic), then casts to fp16 for storage. The backward loads the fp16
values and casts back to fp32 for sigmoid computation:

```
Forward:  h = softplus(z)   [fp32 in registers]  → store as fp16
Backward: load fp16 h → cast to fp32 → sigmoid(h) = 1 - exp(-h)
```

This fp32 → fp16 → fp32 roundtrip loses 13 bits of mantissa precision.
For softplus values near zero, the sigmoid derivative `1 - exp(-h)` is
especially sensitive to rounding.

**Solution**: Store activations as fp32 directly in forward, eliminating the
roundtrip. The backward loads fp32 h for the sigmoid computation, and casts to
fp16 only for the GEMM `tl.dot()` operations (which need fp16 for Tensor Core):

```python
# Forward: store fp32 directly
tl.store(H1_OUT + ..., h1_chunk)  # was h1_chunk.to(INPUT_DTYPE)

# Backward: load fp32, cast to fp16 only for GEMM
h1_f32 = tl.load(H1 + ...)          # fp32 directly
h1_inp = h1_f32.to(INPUT_DTYPE)     # fp16 for tl.dot()
dz1 = dh1 * (1.0 - tl.exp(-h1_f32)) # full precision sigmoid
dw2 = tl.dot(tl.trans(h1_inp), dz2)  # fp16 Tensor Core
```

**Memory impact**: 2× activation memory (fp32 vs fp16). For M=4096, H=128:
h1+h2 = 2MB → 4MB. Negligible for small models.

**Precision impact**: The register-level `.to(INPUT_DTYPE)` for GEMM keeps
48 bits per element (same as before: fp16 raw + fp32 copy), so register
pressure is unchanged.

### FP32 Weight Gradient Accumulation in cuBLAS Path

For M > 32K, the cuBLAS fallback computes weight gradients as
`dW = h.t() @ grad`. With both h and grad in fp16, the accumulated sum over
M samples overflows fp16 at ~M=32K (128 * 8.8 * 32768 ≈ 37M >> 65504).

**Solution**: When `return_fp32_dw=True` (WN backward path), compute DW
GEMMs in fp32:

```python
if return_fp32_dw:
    dw3 = h2.float().t() @ grad_output.float()  # fp32, no overflow
else:
    dw3 = h2_hp.t() @ grad_output  # fp16 Tensor Core (may overflow)
```

**Performance impact**: fp32 matmul is ~8x slower than fp16 Tensor Core. For
the DW GEMM at M=65536: ~53μs (fp32) vs ~6μs (fp16). Three DW GEMMs add
~140μs. At M=65536 where total backward is ~1–2ms, this is ~7–14%.

### FP32 WN Backward Outputs

The WN backward kernel previously stored dv/dg in V's dtype. With
`cast_inputs=torch.float16`, V is fp16, so dv was stored as fp16 — causing
gradient overflow at M ≥ 32K (dv values up to ~70K exceed fp16 max).

**Solution**: Always compute and store WN backward outputs (dv, dg) in fp32:
```python
tl.store(DV + ..., dv, mask=mask)  # was dv.to(V.dtype.element_ty)
tl.store(DG + row, dg_val)         # was dg_val.to(G.dtype.element_ty)
```

This also allocates `dv_buf` and `dg_buf` as explicit fp32 tensors instead of
`torch.empty_like(v_all)` (which would be fp16 from the cast).

### Why True Fusion (Single Kernel) Doesn't Work

The obvious next step is merging the WN backward into the MLP backward
megakernel itself — one kernel launch instead of two. The approach: use an
inter-block atomic barrier after the tile loop, then have the last program
compute the WN backward (the "last-block-does-the-work" pattern):

```python
# After tile loop:
done_count = tl.atomic_add(BARRIER, 1)
if done_count == num_pids - 1:
    # Only the last program executes WN backward
    for wn_row in range(total_wn_rows):
        dw = tl.load(DW + ...)  # DW fully accumulated now
        # ... compute dv, dg
```

This was implemented and tested — correctness passes. But **performance
regressed ~40%**. The root cause: **register pressure**. The Triton compiler
allocates registers for the worst case of all code paths. The WN backward
code inside the `if` branch adds ~256 fp32 values (128 for DW + 128 for V)
to the register file, even though only 1 of 128 programs ever executes it.
The increased register count reduces occupancy for ALL programs during the
tile loop, making the GEMM phase significantly slower.

This is a fundamental limitation of the GPU SIMT execution model: conditional
code paths share register allocation. True fusion would require either:
1. **CUDA cooperative groups** (not available in Triton)
2. **Separate register allocations per phase** (requires compiler support)
3. **CUDA Graphs** (captures both launches, eliminates Python dispatch
   but not GPU launch latency; complex with variable batch sizes)

The 2-kernel approach with pre-allocated DW is the practical optimum for
Triton: **1 memset + 2 kernel launches** total.

---

## 6. Benchmark Results

Hardware: RTX 4090 (128 SMs), D=H=O=128, fp16. All models use weight
normalization. Comparison:
- **Eager**: PyTorch eager with manual `W = g * v / ||v||`
- **Compile**: `torch.compile(mode="max-autotune")` on the eager model
- **Fused**: Fused WN megakernel (M ≤ 40K) / contiguous 2-launch (M > 40K)

### 6.1 Three-Layer MLP

#### Inference (forward only)

```
      N |      Eager |    Compile |      Fused |    F/Eager |  F/Compile
------------------------------------------------------------------------
    256 |    0.129ms |    0.074ms |    0.077ms |     1.68x |     0.96x
    512 |    0.134ms |    0.084ms |    0.081ms |     1.66x |     1.04x
   1024 |    0.134ms |    0.083ms |    0.080ms |     1.67x |     1.03x
   2048 |    0.133ms |    0.083ms |    0.081ms |     1.65x |     1.02x
   4096 |    0.131ms |    0.081ms |    0.079ms |     1.66x |     1.02x
   8192 |    0.133ms |    0.081ms |    0.079ms |     1.69x |     1.03x
  16384 |    0.127ms |    0.081ms |    0.080ms |     1.59x |     1.02x
  32768 |    0.129ms |    0.081ms |    0.114ms |     1.12x |     0.71x
  50000 |    0.199ms |    0.124ms |    0.113ms |     1.76x |     1.09x
  65536 |    0.130ms |    0.093ms |    0.113ms |     1.16x |     0.83x
 100000 |    0.201ms |    0.147ms |    0.139ms |     1.45x |     1.06x
 131072 |    0.277ms |    0.209ms |    0.175ms |     1.59x |     1.19x
 200000 |    0.447ms |    0.426ms |    0.252ms |     1.78x |     1.69x
 262144 |    0.716ms |    0.582ms |    0.311ms |     2.30x |     1.87x
 300000 |    0.834ms |    0.670ms |    0.364ms |     2.29x |     1.84x
```

#### Training (forward + backward)

```
      N |      Eager |    Compile |      Fused |    F/Eager |  F/Compile
------------------------------------------------------------------------
    256 |    0.899ms |    0.645ms |    0.340ms |     2.64x |     1.90x
    512 |    0.728ms |    0.377ms |    0.339ms |     2.15x |     1.11x
   1024 |    0.737ms |    0.378ms |    0.351ms |     2.10x |     1.08x
   2048 |    0.741ms |    0.382ms |    0.347ms |     2.13x |     1.10x
   4096 |    0.744ms |    0.382ms |    0.344ms |     2.16x |     1.11x
   8192 |    0.941ms |    0.560ms |    0.500ms |     1.88x |     1.12x
  16384 |    0.997ms |    0.562ms |    0.500ms |     1.99x |     1.12x
  32768 |    1.003ms |    0.577ms |    0.503ms |     1.99x |     1.15x
  50000 |    1.014ms |    0.853ms |    0.703ms |     1.44x |     1.21x
  65536 |    1.013ms |    1.163ms |    0.704ms |     1.44x |     1.65x
 100000 |    1.014ms |    1.780ms |    0.698ms |     1.45x |     2.55x
 131072 |    1.005ms |    2.483ms |    0.649ms |     1.55x |     3.82x
 200000 |    1.368ms |    3.927ms |    1.244ms |     1.10x |     3.16x
 262144 |    1.992ms |    5.194ms |    1.733ms |     1.15x |     3.00x
 300000 |    2.349ms |    5.969ms |    2.038ms |     1.15x |     2.93x
```

#### Backward-only improvement from pre-allocated DW buffer (3L)

```
      M |  Old (3×zeros) |  New (pre-alloc) |   Saved
------------------------------------------------------
    256 |       0.190ms  |        0.131ms   |   58μs (31%)
    512 |       0.192ms  |        0.131ms   |   61μs (32%)
   1024 |       0.193ms  |        0.128ms   |   64μs (33%)
   2048 |       0.212ms  |        0.144ms   |   67μs (32%)
   4096 |       0.179ms  |        0.130ms   |   48μs (27%)
   8192 |       0.191ms  |        0.142ms   |   49μs (26%)
  16384 |       0.275ms  |        0.194ms   |   81μs (30%)
  32768 |       0.439ms  |        0.291ms   |  148μs (34%)
```

### 6.2 Five-Layer MLP (latest: with fp32 activations + precision fixes)

#### Inference (forward only)

```
      N |      Eager |    Compile |      Fused |    F/Eager |  F/Compile
------------------------------------------------------------------------
    256 |    0.203ms |    0.071ms |    0.090ms |     2.24x |     0.78x
    512 |    0.204ms |    0.075ms |    0.089ms |     2.29x |     0.84x
   1024 |    0.200ms |    0.076ms |    0.090ms |     2.24x |     0.85x
   2048 |    0.200ms |    0.076ms |    0.089ms |     2.24x |     0.85x
   4096 |    0.199ms |    0.074ms |    0.088ms |     2.25x |     0.84x
   8192 |    0.200ms |    0.075ms |    0.084ms |     2.40x |     0.90x
  16384 |    0.191ms |    0.073ms |    0.086ms |     2.21x |     0.84x
  32768 |    0.197ms |    0.086ms |    0.094ms |     2.10x |     0.92x
  50000 |    0.201ms |    0.125ms |    0.159ms |     1.27x |     0.79x
  65536 |    0.242ms |    0.158ms |    0.194ms |     1.25x |     0.82x
 100000 |    0.390ms |    0.257ms |    0.304ms |     1.28x |     0.85x
 131072 |    0.496ms |    0.344ms |    0.382ms |     1.30x |     0.90x
 200000 |    0.795ms |    0.651ms |    0.588ms |     1.35x |     1.11x
 262144 |    1.281ms |    0.875ms |    0.763ms |     1.68x |     1.15x
 300000 |    1.493ms |    1.006ms |    0.878ms |     1.70x |     1.15x
```

#### Training (forward + backward)

```
      N |      Eager |    Compile |      Fused |    F/Eager |  F/Compile
------------------------------------------------------------------------
    256 |    1.208ms |    0.784ms |    0.788ms |     1.53x |     0.99x
    512 |    2.123ms |    0.825ms |    0.788ms |     2.69x |     1.05x
   1024 |    2.109ms |    0.714ms |    0.694ms |     3.04x |     1.03x
   2048 |    2.118ms |    0.830ms |    0.799ms |     2.65x |     1.04x
   4096 |    2.183ms |    0.729ms |    0.717ms |     3.04x |     1.02x
   8192 |    2.111ms |    0.841ms |    0.409ms |     5.16x |     2.05x
  16384 |    2.086ms |    0.838ms |    0.801ms |     2.60x |     1.05x
  32768 |    2.076ms |    0.912ms |    0.737ms |     2.82x |     1.24x
  50000 |    2.123ms |    1.418ms |    1.275ms |     1.66x |     1.11x
  65536 |    1.926ms |    1.899ms |    1.130ms |     1.71x |     1.68x
 100000 |    2.178ms |    2.955ms |    1.794ms |     1.21x |     1.65x
 131072 |    2.156ms |    4.143ms |    2.418ms |     0.89x |     1.71x
 200000 |    2.439ms |    6.512ms |    4.017ms |     0.61x |     1.62x
 262144 |    3.544ms |    8.595ms |    5.324ms |     0.67x |     1.61x
 300000 |    4.164ms |    9.865ms |    6.134ms |     0.68x |     1.61x
```

---

## 7. Analysis

### Where We Win

**Inference** — At small N (≤ 32K), the fused WN megakernel matches or beats
torch.compile (1.02–1.04x for 3L) thanks to single-launch dispatch. At large
N (≥ 100K), the register-tiled fusion dominates: **2.29x** (3L) and **2.38x**
(5L) vs eager at N=300K, and **1.87x** / **1.60x** vs compile.

**Training** — The combined zero-copy backward + pre-allocated DW optimization
was transformative. Fused beats eager at ALL sizes and beats compile at nearly
all sizes:

- 3-layer: **1.08–3.82x vs compile** across all N
- 5-layer: **1.06–3.65x vs compile** at most N (0.87x at N=32K is the sole loss)

### Impact of Pre-Allocated DW Buffer

The backward uses pre-allocated contiguous storage for weight gradients
(DW1, DW2, DW3), zeroed with a single `.zero_()` call instead of 3 separate
`torch.zeros` allocations. A/B testing at M=4096 shows:

- Before (3× `torch.zeros`): 0.190ms backward
- After (1× `.zero_()` on pre-alloc): 0.137ms backward
- **53μs savings (28%)** per backward call

The savings are consistent at 25–34% across all batch sizes (256–32K), with
larger M seeing larger absolute savings (up to 148μs at M=32K).

### Impact of Fused WN Megakernel

The fused kernel (Approach 4) reduced the small-N inference floor from
~0.107ms (2-launch) to ~0.079ms (1-launch) for 3L — a **1.4x improvement**
at the sizes that matter most for real-time inference. This closed the gap
with torch.compile, which was previously 0.78x and is now 1.02–1.04x in our
favor.

### Where torch.compile Still Competes

For 5-layer inference at small N (≤ 32K), compile's CUDAGraph replays the
entire fused graph in ~0.072–0.079ms, while our fused kernel needs
0.088–0.092ms. The gap is the fused WN megakernel's per-tile norm
recomputation overhead. At N=32K 5-layer training, compile still wins 0.87x
due to the backward's cuBLAS fallback threshold change.

### Why Single-Kernel Fusion Failed

An inter-block barrier approach (atomic counter + last-block-does-WN-backward)
was implemented and tested. Correctness passed, but performance **regressed
~40%**. The Triton compiler allocates registers for the union of all code paths
— the WN backward branch added ~256 registers that reduced occupancy for ALL
128 programs during the tile loop, even though only 1 program enters the branch.
This is a fundamental SIMT limitation.

Remaining approaches to eliminate the 2nd kernel launch:
- **CUDA Graphs**: capture both launches into a single replay (~complex with
  variable batch sizes)
- **C++ extension**: dispatch both kernels from C++ without Python GIL
- **Cooperative groups**: not available in Triton

For now, 2 kernel launches + pre-allocated buffers is the practical optimum.

### Summary Table

| Workload | N range | vs Eager | vs Compile |
|---|---|---|---|
| **3L inference** | 256–300K | **1.12–2.30x** | 0.71–**1.87x** |
| **5L inference** | 256–300K | **1.52–2.38x** | 0.80–**1.60x** |
| **3L training** | 256–300K | **1.10–2.64x** | **1.08–3.82x** |
| **5L training** | 256–300K | **1.15–2.76x** | 0.87–**3.65x** |

Crossover point where Fused beats compile:
- 3L Inference: **N ≈ 512** (1.04x)
- 5L Inference: **N ≈ 65K** (1.12x)
- 3L Training: **always wins** (worst: 1.08x at N=1024)
- 5L Training: **N ≈ 256** (1.08x), except N=32K dip (0.87x)

---

## 8. Optimization Journey Summary

| # | Step | Status | Key Change |
|---|---|---|---|
| 1 | Naive PyTorch WN | Baseline | 12+ kernel launches per fwd |
| 2 | torch.compile WN | Better | Single fused graph |
| 3 | Triton WN kernels (3 launches) | Worse than compile | Custom norm kernels |
| 4 | Batched WN (1 launch) | Better | Concat all weights |
| 5 | Contiguous params (2 launch) | Better | Pre-allocated param storage |
| 6 | Fused WN megakernel (1 fwd) | Better | Norm inside GEMM tile loop |
| 7 | Zero-copy bwd (2 bwd launches) | Better | Read fp32 DW column-wise |
| 8 | Pre-alloc DW buffer | Better | 1× `.zero_()` vs 3× `torch.zeros` |
| 9 | ~~Barrier-fused single kernel~~ | **Failed** | Register pressure: -40% |
| 10 | _NoCast wrapper for dw_buf | **Bugfix** | `cast_inputs` was corrupting fp32 buffer |
| 11 | FP32 activation storage | Better | Eliminate fp32→fp16→fp32 roundtrip |
| 12 | FP32 DW GEMMs (cuBLAS path) | **Bugfix** | Prevent fp16 overflow at M>32K |
| 13 | FP32 WN backward outputs | **Bugfix** | Prevent gradient fp16 overflow |

### Timeline (3L, N=4K)

| Step | Inference | Training |
|---|---|---|
| Naive PyTorch WN | ~0.300ms | ~1.500ms |
| torch.compile WN | ~0.082ms | ~0.555ms |
| Triton WN (3 launches) | ~0.165ms | ~0.815ms |
| Batched WN (1 launch) | ~0.147ms | ~0.621ms |
| Contiguous params (2 launch) | ~0.107ms | ~0.448ms |
| Fused WN megakernel (1 fwd) | ~0.076ms | ~0.398ms |
| + Zero-copy bwd | ~0.078ms | ~0.497ms |
| + Pre-alloc DW buffer | ~0.079ms | ~0.344ms |
| + FP32 activations + bugfixes | ~0.088ms | ~0.717ms |

Note: latest numbers include fp32 activation storage (slight fwd regression
from 2× activation bandwidth) and fp32 DW GEMMs in cuBLAS path (slight
training regression at small M, but correct at all M).

---

## 9. Lessons Learned

1. **For tiny matrices, allocation > compute**: The 128×128 normalization
   takes <1μs of GPU compute but each `torch.cat` or `torch.empty` costs
   ~10–15μs from Python. Pre-allocation is essential.

2. **Kernel launch overhead is the final frontier**: After eliminating
   allocations, the remaining ~15μs per kernel launch dominated at small M.
   Fusing WN into the megakernel eliminated this entirely.

3. **Two-pass norm in the inner loop works**: For W1 where BK1 < K1, loading
   V tiles twice (once for norm, once for GEMM) has negligible overhead
   because the 32KB weight matrix fits entirely in L1 cache.

4. **Dynamic dispatch is essential**: The fused kernel's per-tile scaling
   overhead grows with M (each of 128+ SM programs redundantly computes
   norms). Above M=40K, the 2-launch approach (pre-materialized W) wins
   because it amortizes the WN kernel's 15μs over a longer megakernel.

5. **Scale reveals different bottlenecks**: At small N, the bottleneck is
   dispatch overhead (fused megakernel wins). At large N, the bottleneck is
   compute/memory bandwidth (pre-materialized W wins). The crossover is
   around N≈40K for our architecture.

6. **Non-contiguous views are treacherous**: The `dW.t()` view from matrix
   transpose has swapped strides. Triton kernels that compute row-wise
   reductions will silently produce wrong results unless the input is made
   contiguous first. Always check stride assumptions.

7. **Read strided data rather than materializing contiguous copies**: The
   backward originally did `.t().contiguous()` to create row-major copies
   of DW before the WN backward kernel. But 128×128 × 4B = 64KB fits
   entirely in L1 cache — reading column-wise with stride access is
   negligible compared to the ~15μs per kernel launch saved. The zero-copy
   `_wn_bwd_from_dw_T_kernel` eliminated 7 launches and ~105μs overhead.

8. **Keep intermediate precision**: The backward originally cast fp32
   atomics → fp16 → loaded as fp16 in WN backward → cast back to fp32 for
   the reduction. By passing `return_fp32_dw=True`, the WN kernel reads
   fp32 directly — saving 3 cast kernels and improving numerical precision.

9. **Pre-allocate everything that can be reused**: The backward used to
   create 3 fresh `torch.zeros` DW buffers every call. Each `torch.zeros` =
   1 allocation + 1 memset dispatch (~10-15μs from Python). A pre-allocated
   flat buffer with a single `.zero_()` saves 2 dispatches × ~10μs = ~20μs,
   plus faster memset (one contiguous 192KB vs three separate 64KB).
   Measured: **28-34% faster backward** across all batch sizes.

10. **Conditional code paths share register allocation in SIMT**: Attempting
    to fuse WN backward into the megakernel with an inter-block barrier
    (`if done_count == num_pids - 1:`) increased register pressure for ALL
    programs, not just the one that enters the branch. The Triton compiler
    allocates registers for the union of all code paths. The ~256 extra
    fp32 registers for the WN backward reduced occupancy enough to cause a
    **40% performance regression** on the GEMM tile loop. This is a
    fundamental SIMT limitation — you cannot have asymmetric register
    requirements across execution phases in the same kernel.

11. **`cast_inputs` is a silent dtype bomb**: `@torch.amp.custom_fwd(
    cast_inputs=torch.float16)` recursively traverses ALL arguments —
    including lists, tuples, dicts — and casts every tensor to fp16.
    For tensors that MUST remain fp32 (like DW accumulation buffers), this
    causes subtle overflow bugs that only manifest at large batch sizes.
    The fix: wrap fp32 tensors in a custom class (`_NoCast`) that PyTorch's
    pytree traversal doesn't enter.

12. **`cast_inputs` also controls autocast in backward**: Removing
    `cast_inputs` from `custom_fwd` causes `custom_bwd` to restore
    autocast-enabled state in the backward. This means `a.float() @ b.float()`
    STILL produces fp16 output (autocast overrides explicit `.float()`
    casts for matmul). The only safe fix is to keep `cast_inputs` and use
    `_NoCast` for tensors that need fp32.

13. **Store activations at full precision**: The fp32→fp16→fp32 roundtrip
    for saved activations loses 13 mantissa bits. For sigmoid derivatives
    `1 - exp(-h)` near h=0, this precision loss directly corrupts the
    gradient signal. Storing activations as fp32 costs 2× activation memory
    (negligible for small models) but preserves full precision. The backward
    kernel casts to fp16 only for `tl.dot()` calls that need Tensor Core.

14. **fp16 DW accumulation overflows at M ≈ 32K**: Weight gradients
    dW = h.T @ grad accumulate over M samples. At M=32K with h≈8.8
    (typical softplus output), max(dW) ≈ 66K ≈ fp16 max (65504). The
    cuBLAS fallback path must use fp32 matmul for DW computation when the
    WN backward needs fp32 DW. This trades ~140μs of slower matmul at
    M=65K for numerical correctness at all batch sizes.

15. **WN backward gradients must be fp32**: The dv = (g/||v||)(dW - v̂·dg)
    formula amplifies large DW values. With g/||v|| ≈ 0.9 and dW up to
    66K, dv can reach 70K+ — exceeding fp16 max. Always allocate gradient
    output buffers (dv_buf, dg_buf) as explicit fp32, independent of the
    saved tensor dtypes.
