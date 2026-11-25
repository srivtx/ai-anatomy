# %% [markdown]
# # 13 · Your first Triton kernels
# Write a fused softmax and a tiled matrix multiply in Triton, check them against PyTorch, and place them on a
# roofline: is each kernel limited by memory bandwidth or by arithmetic? Chapter: **Kernels**.
#
# **Hardware:** a Colab T4 (Runtime → Change runtime type → T4 GPU). Without a GPU, Triton's interpreter runs
# the same kernels on the CPU for correctness only; timings are skipped.
#
# **You're done when** your fused softmax beats the unfused PyTorch version and both kernels are on your roofline.

# %%
import os
import torch

GPU = torch.cuda.is_available()

if not GPU:
    os.environ["TRITON_INTERPRET"] = "1"  # must be set before importing triton

import triton
import triton.language as tl
import matplotlib.pyplot as plt

dev = "cuda" if GPU else "cpu"
print("GPU:", torch.cuda.get_device_name() if GPU else "none (interpreter mode: correctness only)")

# %% [markdown]
# ## Fused softmax: one read, one write
# PyTorch written naively (max, subtract, exp, sum, divide) launches several kernels, each reading and writing
# the whole matrix from GPU memory. Fused, each program loads one row into registers, does everything there, and
# writes once. Softmax does ~5 flops per element, so it's memory-bound: fewer bytes moved = faster.


# %%
@triton.jit
def softmax_kernel(out_ptr, in_ptr, stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)  # numerical stability
    e = tl.exp(x)
    y = e / tl.sum(e, axis=0)
    tl.store(out_ptr + row * stride + cols, y, mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    softmax_kernel[(n_rows,)](out, x, x.stride(0), n_cols, BLOCK=triton.next_power_of_2(n_cols))
    return out


def naive_softmax(x):
    m = x.max(1, keepdim=True).values
    e = torch.exp(x - m)
    return e / e.sum(1, keepdim=True)


x = torch.randn(256 if GPU else 16, 1000, device=dev)
print("softmax max error vs torch:", (softmax(x) - torch.softmax(x, 1)).abs().max().item())

# %% [markdown]
# ## Tiled matmul
# Each program computes a BLOCK_M × BLOCK_N tile of C, looping over K in BLOCK_K steps. Loaded tiles are reused
# BLOCK_N (or BLOCK_M) times from fast on-chip memory, raising arithmetic intensity. `tl.dot` uses tensor cores.


# %%
@triton.jit
def matmul_kernel(
    a,
    b,
    c,
    M,
    N,
    K,
    sam,
    sak,
    sbk,
    sbn,
    scm,
    scn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        A = tl.load(
            a + rm[:, None] * sam + (k + rk)[None, :] * sak,
            mask=(rm[:, None] < M) & ((k + rk)[None, :] < K),
            other=0.0,
        )
        B = tl.load(
            b + (k + rk)[:, None] * sbk + rn[None, :] * sbn,
            mask=((k + rk)[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(A, B)
    tl.store(
        c + rm[:, None] * scm + rn[None, :] * scn,
        acc.to(c.dtype.element_ty),
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


def matmul(A, B, BM=64, BN=64, BK=32):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, device=A.device, dtype=A.dtype)
    matmul_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        A, B, C, M, N, K, *A.stride(), *B.stride(), *C.stride(), BM=BM, BN=BN, BK=BK
    )
    return C


n = 1024 if GPU else 64
dt = torch.float16 if GPU else torch.float32
A, B = torch.randn(n, n, device=dev, dtype=dt), torch.randn(n, n, device=dev, dtype=dt)
print(
    "matmul max error vs torch:",
    (matmul(A, B).float() - (A.float() @ B.float())).abs().max().item(),
)

# %% [markdown]
# ## Benchmark and roofline (GPU only)
# Achieved performance = min(peak FLOP/s, bandwidth × arithmetic intensity). T4: ~65 TFLOP/s fp16 with tensor
# cores, ~320 GB/s memory bandwidth, so the ridge point is ~200 flops per byte.

# %%
if GPU:
    from triton.testing import do_bench

    PEAK, BW = 65e12, 320e9
    pts = []
    for cols in [512, 2048, 8192]:
        x = torch.randn(4096, cols, device=dev)
        byts = 2 * x.numel() * 4
        for name, f in [
            ("torch.softmax", lambda: torch.softmax(x, 1)),
            ("naive (unfused)", lambda: naive_softmax(x)),
            ("triton fused", lambda: softmax(x)),
        ]:
            ms = do_bench(f)
            print(f"softmax {cols:5d} cols  {name:16s} {ms:7.3f} ms  {byts / ms / 1e6:6.0f} GB/s")
            if name == "triton fused":
                pts.append(("softmax", 5 * x.numel() / byts, 5 * x.numel() / ms * 1e3))
    for n in [256, 1024, 4096]:
        A, B = torch.randn(n, n, device=dev, dtype=torch.float16), torch.randn(
            n, n, device=dev, dtype=torch.float16
        )
        for name, f in [("torch.matmul", lambda: A @ B), ("triton tiled", lambda: matmul(A, B))]:
            ms = do_bench(f)
            fl = 2 * n**3 / ms * 1e3
            print(f"matmul {n:5d}  {name:14s} {ms:7.3f} ms  {fl / 1e12:5.1f} TFLOP/s")
            if name == "triton tiled":
                pts.append((f"matmul {n}", 2 * n**3 / (3 * n * n * 2), fl))
    ai = torch.logspace(-1, 4, 100)
    plt.loglog(ai, torch.minimum(torch.full_like(ai, PEAK), BW * ai), "k-", label="roofline (T4)")
    for name, a_, f in pts:
        plt.loglog(a_, f, "o")
        plt.annotate(name, (a_, f), fontsize=8)
    plt.xlabel("arithmetic intensity (flops / byte)")
    plt.ylabel("achieved flop/s")
    plt.legend()
    plt.show()
else:
    print(
        "No GPU: kernels verified for correctness in the interpreter; switch to a T4 for timings."
    )

# %% [markdown]
# **What to expect on a T4.** The fused softmax runs at a large fraction of the 320 GB/s bandwidth (often
# 200–280 GB/s) and beats the naive five-kernel version by ~2–4×; it's comparable to `torch.softmax`, which is
# also fused. Softmax sits far left on the roofline: memory-bound. The tiled matmul at 4096 reaches a good share
# of `torch.matmul` without autotuning (try BM=128, BN=128, BK=32, `num_warps=8`) and sits near the compute roof.
#
# **Stretch:** a fused attention forward pass: load a tile of Q, loop over K/V tiles keeping a running max and
# sum (online softmax), never materialising the T×T attention matrix. That's FlashAttention.
