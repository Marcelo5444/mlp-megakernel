# Split-K Atomic Backward: Solving the GPU Weight Gradient Bottleneck

*How a simple partitioning trick can eliminate both atomic contention and Python
dispatch overhead in fused Triton MLP backward passes.*

---

## The Problem: A Lose-Lose Choice

When you fuse a multi-layer MLP backward pass into a single GPU kernel, you face
a dilemma with weight gradients. Every thread tile computes a small partial
contribution to the weight gradient matrix, and those contributions must be
accumulated into a single result. You have two options, and both are bad.

### Option A: Atomic Accumulation (Current Fused Kernel)

Each tile computes its local `dW_partial = input_tile.T @ dz_tile` and atomically
adds it to the shared output matrix:

```python
# Inside _fused_mlp_bwd_full_kernel (one of ~2048 tiles):
dw3_partial = tl.dot(tl.trans(h2_raw), g)
tl.atomic_add(DW3 + offsets, dw3_partial)   # all tiles fight for the same matrix
```

The problem: the weight matrix is small (128×128 = 16,384 fp32 entries), and there
are MANY tiles. For a batch size M=32768 with BM=64, there are 512 row-tiles.
All 512 compete to atomically update the same 16K entries.

On an RTX 4090, `tl.atomic_add` on fp32 global memory serializes conflicting
writes. When 512 tiles simultaneously try to update overlapping addresses, the
atomic unit becomes the bottleneck — stalling warps while they wait for their
turn.

**Measured impact:**

| Batch Size (M) | Atomic Kernel | cuBLAS Fallback | Atomic Slowdown |
|---|---|---|---|
| 4,096 | 0.135 ms | 0.121 ms | 11% slower |
| 16,384 | 0.215 ms | 0.112 ms | 92% slower |
| 32,768 | 0.385 ms | 0.116 ms | **232% slower** |

At M=32K, atomic contention makes the fused kernel **3.3× slower** than cuBLAS for
the weight gradient computation alone. The kernel spends more time waiting on
atomics than it does on actual matrix math.

### Option B: cuBLAS Fallback (Current Large-M Path)

Skip the fused kernel for large batches. Instead, launch separate operations:

```python
dh2 = grad_output @ w3.t()    # cuBLAS GEMM   (~20 μs)
dw3 = h2.t() @ grad_output    # cuBLAS GEMM   (~20 μs)
dz2 = _sigmoid_h_mul(h2, dh2) # Triton kernel (~10 μs)
dh1 = dz2 @ w2.t()            # cuBLAS GEMM   (~20 μs)
dw2 = h1.t() @ dz2            # cuBLAS GEMM   (~20 μs)
dz1 = _sigmoid_h_mul(h1, dh1) # Triton kernel (~10 μs)
dx  = dz1 @ w1.t()            # cuBLAS GEMM   (~20 μs)
dw1 = x.t() @ dz1             # cuBLAS GEMM   (~20 μs)
```

That's **8 separate kernel launches**. Each one is individually fast — cuBLAS is
highly optimized for 128×128 GEMMs. But each launch must pass through the Python
interpreter, the PyTorch dispatcher, and the CUDA driver. On the RTX 4090, each
Python→GPU dispatch costs approximately **15 μs**.

**8 launches × 15 μs = 120 μs of pure overhead.**

For small matrices where the GPU compute is only 100-200 μs total, this overhead
is catastrophic. It nearly doubles the wall time.

**Full training measurements (forward + backward, through `autograd.Function`):**

| Batch Size | PyTorch Eager | Fused (Atomic) | Fused (Fallback) |
|---|---|---|---|
| 512 | 0.395 ms | 0.412 ms | 0.518 ms |
| 4,096 | 0.404 ms | 0.422 ms | 0.536 ms |

The atomic kernel — despite being **slower in raw GPU compute** — wins in full
training because its 1 kernel launch has 120 μs less Python overhead than the 8
separate launches.

### The Dilemma Visualized

```
                      Option A: Atomic               Option B: Fallback
                     ┌──────────────────┐           ┌──────────────────┐
 Python overhead:    │  15 μs (1 launch)│           │ 120 μs (8 launch)│
                     ├──────────────────┤           ├──────────────────┤
                     │                  │           │                  │
 GPU compute (dx):   │  ~80 μs          │           │  ~80 μs          │
                     │                  │           │                  │
                     ├──────────────────┤           ├──────────────────┤
 GPU compute (dW):   │ ~300 μs (contend)│           │  ~60 μs (cuBLAS) │
                     │  ████████████████│           │                  │
                     │  █ BLOCKED ON  ██│           │                  │
                     │  █  ATOMICS   ██ │           │                  │
                     └──────────────────┘           └──────────────────┘
                       Total: ~395 μs                 Total: ~260 μs
                       +15 μs = 410 μs                +120 μs = 380 μs
```

Neither option is satisfying. We want the low launch overhead of the atomic path
AND the low contention of separate cuBLAS calls. Can we have both?

---

## The Idea: Partition the Atomic Target

The core insight is simple: **atomic contention is proportional to how many tiles
write to the same memory address**. If 512 tiles all write to `DW3[i][j]`, that's
512-way contention. But if we give them 8 different copies of `DW3` and each tile
writes to one of the 8 copies, contention drops to 512/8 = 64-way.

This is the **split-K** pattern, adapted for atomic accumulation.

### The Current Atomic Approach

All tiles accumulate into ONE copy of each weight gradient:

```
Tile 0 ─► tl.atomic_add(DW3[i,j], partial_0)  ──┐
Tile 1 ─► tl.atomic_add(DW3[i,j], partial_1)  ──┤
Tile 2 ─► tl.atomic_add(DW3[i,j], partial_2)  ──┤  512 writers, 1 target
Tile 3 ─► tl.atomic_add(DW3[i,j], partial_3)  ──┤  = massive contention
...                                               │
Tile 511 ─► tl.atomic_add(DW3[i,j], partial_511)─┘

Result: DW3 = Σ(all partials)   ← correct, but SLOW
```

### The Split-K Approach

Allocate `SPLIT_K=8` copies of each weight gradient. Route tiles to their
assigned partition using modular arithmetic:

```
Tile 0   ─► tl.atomic_add(DW3_parts[0][i,j], partial)  ──┐  64 writers each
Tile 8   ─► tl.atomic_add(DW3_parts[0][i,j], partial)  ──┤
Tile 16  ─► tl.atomic_add(DW3_parts[0][i,j], partial)  ──┘

Tile 1   ─► tl.atomic_add(DW3_parts[1][i,j], partial)  ──┐  64 writers each
Tile 9   ─► tl.atomic_add(DW3_parts[1][i,j], partial)  ──┤
Tile 17  ─► tl.atomic_add(DW3_parts[1][i,j], partial)  ──┘

...

Tile 7   ─► tl.atomic_add(DW3_parts[7][i,j], partial)  ──┐  64 writers each
Tile 15  ─► tl.atomic_add(DW3_parts[7][i,j], partial)  ──┤
Tile 23  ─► tl.atomic_add(DW3_parts[7][i,j], partial)  ──┘

═══════════════════════════════════════════════════════════
Then: DW3 = DW3_parts[0] + DW3_parts[1] + ... + DW3_parts[7]   ← trivial reduce
```

512 tiles split into 8 groups of 64. Each group writes to a DIFFERENT memory
region. Contention drops 8×.

---

## How It Works in Detail

### Step 1: Allocate Partitioned Buffers

Instead of one `DW3 = zeros(128, 128)`, allocate 8 copies:

```python
SPLIT_K = 8

DW3_parts = torch.zeros((SPLIT_K, N2, N3), device=device, dtype=torch.float32)
DW2_parts = torch.zeros((SPLIT_K, N1, N2), device=device, dtype=torch.float32)
DW1_parts = torch.zeros((SPLIT_K, K1, N1), device=device, dtype=torch.float32)
```

Memory cost: for 128-wide 3-layer MLP, each weight matrix is 128×128 × 4 bytes =
64 KB. Three matrices × 8 copies = 3 × 64 KB × 8 = **1.5 MB**. Negligible compared
to the batch data (M × 128 × 2 bytes = 8 MB for M=32768 in fp16).

### Step 2: Route Tiles to Partitions

Inside the backward kernel, each tile computes which partition it belongs to:

```python
@triton.jit
def _fused_mlp_bwd_splitk_kernel(
    GRAD, W3, H2, W2, H1, W1, X,
    DX,
    DW3_PARTS, DW2_PARTS, DW1_PARTS,    # new: partitioned buffers
    SPLIT_K: tl.constexpr,               # new: number of partitions
    M, N3: tl.constexpr, N2: tl.constexpr, N1: tl.constexpr, K1: tl.constexpr,
    # ... strides ...
    BM: tl.constexpr, BOUT: tl.constexpr,
    BK1: tl.constexpr, BK2: tl.constexpr, BK3: tl.constexpr,
    BKx: tl.constexpr,
    GROUP_M: tl.constexpr,
    INPUT_DTYPE: tl.constexpr = tl.float16,
):
    raw_pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_m = tl.cdiv(M, BM)

    for tile_id in range(raw_pid, num_tiles, num_pids):
        # ... tile indexing (same as before) ...

        # ─── Partition assignment ───
        part_idx = tile_id % SPLIT_K    # which of the 8 partitions

        # ─── Data gradient chain (UNCHANGED) ───
        # dh2 = grad @ W3.T, dz2 = sigmoid(h2) * dh2, ...
        # dx_acc accumulation, etc.
        # ... (identical to current kernel) ...

        # ─── Weight gradients: write to PARTITIONED buffer ───
        dw3_partial = tl.dot(tl.trans(h2_raw), g, out_dtype=tl.float32)
        tl.atomic_add(
            DW3_PARTS                     # base of (SPLIT_K, N2, N3) tensor
            + part_idx * (N2 * N3)        # select partition
            + offs_k2[:, None] * N3       # row within partition
            + offs_k1[None, :],           # column within partition
            dw3_partial,
        )
        # Same pattern for DW2_PARTS and DW1_PARTS
```

The critical line is `part_idx = tile_id % SPLIT_K`. Consecutive tile IDs
(0, 1, 2, ..., 7) map to different partitions. Tiles 0, 8, 16, 24, ... all write
to partition 0. Tiles 1, 9, 17, 25, ... write to partition 1. And so on.

Since tiles executing simultaneously tend to have consecutive IDs (from the
persistent kernel's round-robin schedule), adjacent tiles write to DIFFERENT
partitions — minimizing the chance of simultaneous atomic conflict.

### Step 3: Reduce Partitions

After the fused kernel completes, sum the 8 partitions into the final weight
gradient. This is a trivial element-wise reduction:

```python
@triton.jit
def _reduce_splitk_kernel(PARTS, OUT, n_elements, SPLIT_K: tl.constexpr,
                          BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(SPLIT_K):
        acc += tl.load(PARTS + k * n_elements + offs, mask=mask, other=0.0)

    tl.store(OUT + offs, acc, mask=mask)
```

For a 128×128 matrix with SPLIT_K=8, that's 16,384 elements summed 8 times.
At ~1 TB/s memory bandwidth, this takes **~1 μs**. Three matrices (DW1, DW2, DW3)
= **~3 μs total** for the reduction step.

### The Full Backward Call

```python
def _fused_mlp_bwd_splitk(grad_output, x, w1, w2, w3, h1, h2):
    SPLIT_K = 8
    M, N3, N2, N1, K1 = grad_output.shape[0], w3.shape[1], w2.shape[1], w1.shape[1], w1.shape[0]

    DX = torch.empty((M, K1), device=x.device, dtype=x.dtype)
    DW3_parts = torch.zeros((SPLIT_K, N2, N3), device=x.device, dtype=torch.float32)
    DW2_parts = torch.zeros((SPLIT_K, N1, N2), device=x.device, dtype=torch.float32)
    DW1_parts = torch.zeros((SPLIT_K, K1, N1), device=x.device, dtype=torch.float32)

    # Launch 1: Fused backward with partitioned atomics
    _fused_mlp_bwd_splitk_kernel[grid](
        grad_output, w3, h2, w2, h1, w1, x,
        DX, DW3_parts, DW2_parts, DW1_parts,
        SPLIT_K=SPLIT_K, M=M, ...
    )

    # Launch 2: Reduce partitions → final weight gradients
    DW3 = _reduce(DW3_parts)   # sum over SPLIT_K dimension
    DW2 = _reduce(DW2_parts)
    DW1 = _reduce(DW1_parts)

    return DX, DW1.to(x.dtype), DW2.to(x.dtype), DW3.to(x.dtype)
```

**Total kernel launches: 2** (fused backward + reduce). Compare to the current
system's 1 launch (atomic, high contention) or 8 launches (fallback, high overhead).

---

## Why This Works: The Math of Contention

### Modeling Atomic Throughput

On NVIDIA Ampere/Ada GPUs, `atomicAdd` on fp32 global memory has a throughput of
approximately **one atomic per 16-32 cycles per L2 cache partition**. The L2 cache
has 64 partitions on the RTX 4090.

For a 128×128 matrix (16,384 entries), the entries are distributed across ~64 L2
partitions, giving ~256 entries per partition.

**Without split-K** (M=32768, BM=64 → 512 tiles):
- Each entry receives 512 atomic writes
- Each L2 partition handles ~256 × 512 = 131K atomics
- At 1 atomic per 16 cycles per partition at 2.5 GHz: 131K × 16 / 2.5G ≈ 840 μs
- But atomics can pipeline, so realistic is ~300-400 μs (matches our measured 385 μs)

**With split-K = 8** (same scenario):
- Each entry in each partition receives 512/8 = 64 atomic writes
- Each L2 partition handles ~256 × 64 = 16K atomics per partition
- 16K × 16 / 2.5G ≈ 100 μs
- With pipelining: ~40-60 μs

**Estimated improvement: 385 μs → ~50 μs for the atomic portion alone.**

### The Full Picture

| Component | Current Atomic | Split-K (SPLIT_K=8) | Fallback (8 launches) |
|---|---|---|---|
| Python dispatch | ~15 μs (1 launch) | ~30 μs (2 launches) | ~120 μs (8 launches) |
| Data gradient (dx) | ~80 μs | ~80 μs | ~80 μs |
| Weight gradients (dW) | ~300 μs (contention) | ~50 μs (partitioned) | ~60 μs (cuBLAS) |
| Reduce step | — | ~3 μs | — |
| **Total** | **~395 μs** | **~163 μs** | **~260 μs** |

Split-K is faster than BOTH alternatives at M=32768:
- **2.4× faster** than current atomic (395 → 163 μs)
- **1.6× faster** than cuBLAS fallback (260 → 163 μs)

The key is that split-K achieves near-cuBLAS compute efficiency (~50 μs vs ~60 μs)
while maintaining the low launch overhead of the fused path (~30 μs vs ~120 μs).

---

## Choosing SPLIT_K: The Tradeoff Space

SPLIT_K is not free. There are three costs:

### 1. Memory

Each weight matrix gets SPLIT_K copies. For 128-wide 3-layer MLP:

| SPLIT_K | Extra Memory | % of M=4096 batch data |
|---|---|---|
| 4 | 0.75 MB | 75% |
| 8 | 1.50 MB | 150% |
| 16 | 3.00 MB | 300% |
| 32 | 6.00 MB | 600% |

For practical batch sizes (M ≥ 1024), even SPLIT_K=16 is negligible.

### 2. Reduction Kernel Overhead

The reduce kernel launch adds ~15 μs (Python dispatch) + ~3 μs (compute). This
is a fixed cost independent of M. For very small M (≤ 512), this overhead may
exceed the contention savings.

### 3. Diminishing Returns

| SPLIT_K | Contention per entry (M=32K) | Estimated dW time | Reduce cost |
|---|---|---|---|
| 1 (current) | 512-way | ~300 μs | 0 μs |
| 4 | 128-way | ~100 μs | ~3 μs |
| 8 | 64-way | ~50 μs | ~3 μs |
| 16 | 32-way | ~30 μs | ~4 μs |
| 32 | 16-way | ~25 μs | ~6 μs |

Beyond SPLIT_K=8, the contention is already low enough that further splitting
barely helps. The sweet spot is **SPLIT_K=8**: it's the knee of the curve where
contention drops ~6× with minimal memory and reduce overhead.

### Dynamic SPLIT_K Heuristic

The optimal SPLIT_K depends on the number of tiles (which depends on M):

```python
def _pick_split_k(M, BM):
    num_m_tiles = (M + BM - 1) // BM
    if num_m_tiles <= 64:
        return 1    # contention is already low, no split needed
    elif num_m_tiles <= 256:
        return 4
    elif num_m_tiles <= 1024:
        return 8
    else:
        return 16
```

This mirrors tiny-cuda-nn's approach: `split_k_factor = batch_size / min(4096, batch_size)`.
But instead of launching separate CUTLASS kernels (which requires C++ stream
management), we partition within the same fused kernel.

---

## Why This Eliminates the Hybrid Dispatch

With split-K atomics, there's no longer a need for two separate code paths.
The current system switches between strategies at a threshold:

```python
# Current: hard threshold between two imperfect options
_ATOMIC_M_THRESHOLD = 32768

if M > _ATOMIC_M_THRESHOLD:
    return _fallback_backward(...)    # 8 launches, no contention
else:
    return _fused_atomic_backward(...)  # 1 launch, high contention
```

With split-K, a SINGLE code path handles all batch sizes:

```python
# Split-K: one path, dynamically tuned
SPLIT_K = _pick_split_k(M, BM)
return _fused_splitk_backward(...)    # 2 launches, low contention
```

- **Small M** (≤ 4096): `SPLIT_K=1` (no split, same as current atomic — contention
  is already low with few tiles)
- **Medium M** (4K-32K): `SPLIT_K=4-8` (moderate split, eliminates contention)
- **Large M** (> 32K): `SPLIT_K=8-16` (full split, matches cuBLAS speed with
  much less Python overhead)

No more threshold tuning. No more maintaining two backward paths. One kernel,
one heuristic, all batch sizes.

---

## What Changes in the Kernel

The kernel modifications are minimal. Here's a diff-style view of the critical
sections:

### Kernel Arguments

```python
# Before:
DW3, DW2, DW1,                          # shape: (N2, N3), (N1, N2), (K1, N1)

# After:
DW3_PARTS, DW2_PARTS, DW1_PARTS,        # shape: (SPLIT_K, N2, N3), etc.
SPLIT_K: tl.constexpr,                   # new parameter
```

### Partition Assignment

```python
# New: one line at the top of the tile loop
part_idx = tile_id % SPLIT_K
```

### Atomic Writes

```python
# Before (writing to single matrix):
tl.atomic_add(
    DW3 + offs_k2[:, None] * stride_dw3k + offs_k1[None, :] * stride_dw3n,
    dw3_partial,
)

# After (writing to partitioned buffer):
tl.atomic_add(
    DW3_PARTS
    + part_idx * (N2 * N3)                # offset to this tile's partition
    + offs_k2[:, None] * N3               # row within partition
    + offs_k1[None, :],                   # column within partition
    dw3_partial,
)
```

Everything else — the data gradient chain, the tile indexing, the heuristic — stays
identical. The kernel does the same work; it just spreads the atomic pressure across
more memory.

---

## Comparison with tiny-cuda-nn's Approach

tiny-cuda-nn solves the same problem differently. It launches weight gradient GEMMs
as **separate CUTLASS split-K kernels on separate C++ CUDA streams**:

```cpp
// tiny-cuda-nn backward_impl() — C++ code
for (uint32_t i = 0; i < n_hidden_layers; ++i) {
    multi_streams.emplace_back(stream, 2);
    fc_multiply_split_k(
        multi_streams.back().get(1),    // separate CUDA stream
        backward_tmp, forward_hidden.transposed(),
        gradient_matrix, split_k_factor, beta
    );
}
```

| Aspect | tiny-cuda-nn | Our Split-K Atomic |
|---|---|---|
| Weight grad strategy | Separate CUTLASS GEMM | Fused inside backward kernel |
| Parallelism mechanism | Separate CUDA streams (C++) | Partitioned atomic buffers |
| Stream overhead | ~0 μs (C++ dispatch) | N/A (no streams needed) |
| Launch overhead | 0 extra (concurrent streams) | +1 launch (reduce kernel) |
| Memory cost | Split-K workspace | SPLIT_K × weight matrix copies |
| Overlap with dx chain | Full overlap (separate stream) | Implicit (same kernel) |

tiny-cuda-nn's approach is architecturally cleaner — it uses the GPU's native
stream parallelism. But it requires C++ orchestration. Our approach achieves a
similar result within the Python/Triton ecosystem by converting the inter-stream
parallelism problem into an intra-kernel memory partitioning problem.

---

## The Performance Landscape After Split-K

Projected performance comparison at various batch sizes:

| M | Eager PyTorch | Current Fused | Split-K Fused | Speedup vs Eager |
|---|---|---|---|---|
| 512 | 0.395 ms | 0.412 ms | ~0.395 ms | 1.0× |
| 4,096 | 0.404 ms | 0.422 ms | ~0.350 ms | 1.15× |
| 16,384 | 0.470 ms | 0.480 ms | ~0.300 ms | 1.57× |
| 32,768 | 0.530 ms | 0.580 ms | ~0.250 ms | 2.12× |
| 65,536 | 0.650 ms | fallback | ~0.400 ms | 1.63× |

The key improvement is at **medium-to-large M** (4K-64K), where the current system
either suffers from atomic contention or falls back to high-overhead cuBLAS.
Split-K eliminates this dead zone.

---

## Summary

The split-K atomic backward is a targeted fix for a specific bottleneck:

1. **Problem**: Atomic contention for weight gradients scales quadratically with
   batch size, but the only alternative (separate cuBLAS launches) adds too much
   Python dispatch overhead.

2. **Solution**: Allocate SPLIT_K copies of each weight gradient matrix. Route
   each tile to a specific partition. Reduce after the kernel finishes.

3. **Cost**: 1.5 MB extra memory (SPLIT_K=8), 1 extra kernel launch (~18 μs).

4. **Benefit**: Atomic contention drops ~8×, making the fused kernel competitive
   with cuBLAS at all batch sizes while maintaining the low dispatch overhead
   of a single fused launch.

5. **Code change**: ~20 lines modified in the backward kernel, ~15 lines for the
   reduce kernel. No changes to the forward pass, the autograd function, or the
   module interface.

The optimization is simple, low-risk, and addresses the single largest remaining
bottleneck in the training path. It bridges the gap between our current system and
tiny-cuda-nn's C++ multi-stream approach — without leaving Python.
