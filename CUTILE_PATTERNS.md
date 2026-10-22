# Verified cuTile Patterns (GB10, sm_121, cuTile 1.7, CUDA 13.4)

These patterns are TESTED and CORRECT in this repo. Copy them exactly.

## Launch syntax (NOT Triton-style)

```python
import cuda.tile as ct
stream = torch.cuda.current_stream()
ct.launch(stream, grid, kernel, (arg1, arg2, ...))   # CORRECT
# kernel[grid](args)  # WRONG — TypeError: 'kernel' object is not subscriptable
```

Grid is a tuple: `grid = (min(num_sms, num_tiles),)`

## A @ W^T pattern (verified, diff=0.0)

For `OUT[m, n] = sum_k A[m, k] * W[n, k]` (e.g. dh1 = dout @ W3^T, W3 is (N2, N3)):
```python
# W is stored as (N_rows, K_cols) = (N2, N3); we need W^T[k, n] = W[n, k]
# Load W at tile (n_tile, k_tile) with shape (TN, TK), then transpose -> (TK, TN)
for k in range(ct.cdiv(K_cols, TK)):
    a_t = ct.load(A, (pm, k), shape=(TM, TK), latency=4)
    w_t = ct.load(W, (pn, k), shape=(TN, TK), latency=4)  # NOTE: (n, k) NOT (k, n)
    w_t = ct.transpose(w_t)                                # (TK, TN)
    acc = ct.mma(a_t, w_t, acc)                            # (TM, TN)
```
Getting the load indices backwards produces wrong results (cos ~0.1), NOT a compile error.

## MMA dtype rule

`ct.mma(x, y, acc)`: x and y must have the SAME dtype (both fp16 for tensor cores).
acc can be fp32. Cast operands with `ct.astype(tile, ct.float16)` before mma.

## atomic_add for gradient accumulation (verified)

```python
# Accumulate a (TR, TC) fp32 tile into a global 2D fp32 tensor at tile (pr, pc)
rows = ct.arange(TR, dtype=ct.int32) + pr * TR
cols = ct.arange(TC, dtype=ct.int32) + pc * TC
row_idx = ct.reshape(rows, (TR, 1))
col_idx = ct.reshape(cols, (1, TC))
ct.atomic_add(DW, (row_idx, col_idx), acc)   # acc must be fp32, DW must be fp32
```
atomic_add does NOT work on fp16 tensors — use fp32 accumulator tensors and cast
on the host side afterward.

## Softplus + derivative (verified)

```python
def softplus_tile(x):
    return ct.where(x > 20.0, x, ct.log(ct.exp(x) + 1.0))

def softplus_grad_tile(x):   # sigmoid
    ex = ct.exp(x)
    return ct.where(x > 20.0, 1.0, ex / (1.0 + ex))
```

## Persistent kernel skeleton

```python
@ct.kernel
def my_kernel(..., TM: ct.Constant[int], ...):
    raw_pid = ct.bid(0)
    num_pids = ct.num_blocks(0)
    num_m = ct.cdiv(M, TM)
    for pm in range(raw_pid, num_m, num_pids):
        ...  # process tile pm
```

## Constants

Every non-tensor scalar in the kernel signature needs `ct.Constant[int]` annotation.
Tile dims must be powers of 2.

## Docker (GB10)

```bash
docker run --rm --gpus all -v /home/marcelo/mlp-megakernel:/work -w /work \
    cuda134-cutile17-cutedsl:latest python3 <script>.py
```
In scripts, `sys.path.insert(0, "/work")` for imports from the repo.

## Leaf tensors for autograd tests

`torch.randn(...) * 0.02` creates a NON-leaf tensor — `.requires_grad_(True)`
must be called AFTER the multiply, and grads only populate on leaf tensors:
```python
x = torch.randn(M, D, dtype=torch.float16, device='cuda') * 0.02
x.requires_grad_(True)   # leaf now
```

## Correctness gate for backward

vs PyTorch eager autograd, cosine similarity >= 0.99 on all 4 grads (dx, dW1, dW2, dW3).
Note: fp16 intermediate storage amplifies absolute error in dW2 (~5-8 abs diff at
M=256 is normal with cos ~0.997). Use cosine, not allclose.

## Benchmark

```python
from triton.testing import do_bench
t_us = do_bench(fn, warmup=10, rep=50) * 1000   # L2 flushed between reps
```
