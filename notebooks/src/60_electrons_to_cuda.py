# %% [markdown]
# # 60 · From electrons to CUDA, measured
# The "From electrons to CUDA" level makes claims at every layer of the stack. This notebook checks them with
# numbers you compute or measure yourself, bottom to top.
#
# 1. **Physics.** Free electrons in silicon against temperature, and the 60 mV-per-decade limit from the Boltzmann factor.
# 2. **Logic.** An adder and a multiplier built only from NAND gates. Does the multiplier really grow as bits²?
# 3. **Latency hiding.** Simulate a GPU scheduler with many warps and check Little's law.
# 4. **A real GPU** (Colab T4). Your first CUDA kernel, HBM against PCIe bandwidth, launch overhead, and bandwidth
#    against the amount of work in flight.
# 5. **The whole trip.** Electrons per bit, distance from the Landauer limit, and what one token costs.
#
# **Hardware:** parts 1–3 and 5 run on any CPU in seconds. Part 4 needs a GPU (Runtime → Change runtime type → T4) and
# is skipped without one. **You're done when** every check prints ✅.

# %%
import os, time, math, numpy as np
QUICK = os.environ.get("QUICK") == "1"
k = 8.617e-5                                    # Boltzmann's constant in eV per kelvin

# %% [markdown]
# ## Part 1: electrons and the 60 mV rule
# Pure silicon's free electrons are those heat lifts across a 1.12 eV gap. Their density is
# $n_i = \sqrt{N_c N_v}\,e^{-E_g/2kT}$. Then the transistor: below threshold, current is electrons hopping a barrier,
# so it scales as $e^{qV/nkT}$, and 10× more current needs at least $kT\ln 10$ of gate voltage.

# %%
gap = lambda T: 1.17 - 4.73e-4 * T**2 / (T + 636)                       # Varshni fit for silicon
n_i = lambda T: math.sqrt(2.86e19 * 3.10e19 * (T / 300)**3) * math.exp(-gap(T) / (2 * k * T))
for T in [250, 300, 350, 400, 500]:
    print(f"{T} K ({T-273:>4} °C): {n_i(T):.2e} free electrons per cm³")
ratio = n_i(400) / n_i(300)
print(f"300 K → 400 K multiplies them by {ratio:.0f}")

def drain_current(Vg, Vt=0.3, n=1.3, T=300, I0=1.8e-5):
    x = (Vg - Vt) / (2 * n * k * T)
    return I0 * np.log1p(np.exp(x))**2

V = np.linspace(0, 0.2, 201)                                              # well below threshold
slope = np.polyfit(V, np.log10(drain_current(V)), 1)[0]                   # decades per volt
swing = 1000 / slope
print(f"measured swing: {swing:.1f} mV per decade; Boltzmann limit × n: {1.3 * k * 300 * math.log(10) * 1000:.1f}")
ok1 = 1e9 < n_i(300) < 2e10 and 200 < ratio < 1000 and abs(swing - 1.3 * 59.5) < 3
print("✅ electrons are exponential in temperature, and so is a transistor below threshold" if ok1 else "❌ check part 1")

# %% [markdown]
# ## Part 2: arithmetic from one gate
# Everything below is built from `NAND` alone, and every call is counted. Then multipliers of growing width:
# if the chapter is right, the gate count grows as about n².

# %%
calls = 0
def NAND(a, b):
    global calls; calls += 1
    return 1 - (a & b)
def NOT(a): return NAND(a, a)
def AND(a, b): return NOT(NAND(a, b))
def OR(a, b): return NAND(NOT(a), NOT(b))
def XOR(a, b):
    n = NAND(a, b); return NAND(NAND(a, n), NAND(b, n))
def full_adder(a, b, c):
    s1 = XOR(a, b); return XOR(s1, c), OR(AND(a, b), AND(s1, c))
def add(x, y, n):
    out, c = 0, 0
    for i in range(n):
        s, c = full_adder((x >> i) & 1, (y >> i) & 1, c); out |= s << i
    return out | (c << n)
def multiply(x, y, n):
    total = 0
    for i in range(n):
        bit = (y >> i) & 1
        row = 0
        for j in range(n): row |= AND((x >> j) & 1, bit) << (i + j)
        total = add(total, row, 2 * n) & ((1 << 2 * n) - 1)
    return total

rng = np.random.default_rng(0)
pairs = [(int(a), int(b)) for a, b in rng.integers(0, 256, (200 if QUICK else 2000, 2))]
assert all(add(a, b, 8) == a + b for a, b in pairs), "the NAND adder is wrong"
assert all(multiply(a, b, 8) == a * b for a, b in pairs[:200]), "the NAND multiplier is wrong"
print(f"8-bit add and multiply agree with Python on {len(pairs)} random pairs")

widths, gates = [2, 4, 8, 12, 16] if QUICK else [2, 4, 8, 12, 16, 24, 32], []
for n in widths:
    calls = 0; multiply((1 << n) - 1, (1 << n) - 1, n); gates.append(calls)
    print(f"{n:>2}-bit multiplier: {calls:>6} NAND gates")
power = np.polyfit(np.log(widths[2:]), np.log(gates[2:]), 1)[0]
print(f"gates grow as bits^{power:.2f}")
ok2 = 1.8 < power < 2.2
print("✅ a multiplier grows with the square of its bits (so fp8 is far cheaper than fp32)" if ok2 else "❌ check part 2")

# %% [markdown]
# Note: this textbook multiplier adds each of its n rows with a 2n-bit ripple adder, so it uses about n × 2n full
# adders. Real multipliers use adder trees: the same n² growth in gates, but far fewer gates on the slowest path.
#
# ## Part 3: hiding latency with warps
# One scheduler issues one instruction per cycle from the first warp that is ready. Each warp computes for C
# cycles, then waits L cycles for memory. Little's law predicts the scheduler is busy min(1, W·C / (C + L)) of
# the time with W warps.

# %%
def simulate(W, C, L, T=20000):
    ready, left, busy = [0] * W, [C] * W, 0
    for t in range(T):
        for w in range(W):
            if ready[w] <= t:
                busy += 1; left[w] -= 1
                if left[w] == 0: left[w], ready[w] = C, t + 1 + L
                break
    return busy / T

C, L = 20, 400
print(f"{'warps':>5} {'simulated':>10} {'Little':>8}")
errs = []
for W in [1, 2, 4, 8, 16, 21, 32, 64]:
    sim, pred = simulate(W, C, L, 6000 if QUICK else 20000), min(1, W * C / (C + L))
    errs.append(abs(sim - pred)); print(f"{W:>5} {sim:>10.3f} {pred:>8.3f}")
need = math.ceil((C + L) / C)
print(f"warps needed to stay busy: {need}")
h100_inflight = 3.35e12 * 350e-9
print(f"H100: 3.35 TB/s × 350 ns = {h100_inflight / 1e6:.2f} MB must be in flight, {h100_inflight / 132 / 1024:.1f} KB per SM")
ok3 = max(errs) < 0.05
print("✅ more warps hide more latency, exactly as Little's law says" if ok3 else "❌ check part 3")

# %% [markdown]
# ## Part 4: a real GPU
# Compile a CUDA kernel from Python with `load_inline` (about a minute the first time), check it, then measure:
# * **bandwidth** of on-GPU copies (HBM/GDDR) against copies from the CPU over PCIe;
# * **launch overhead**: the cost of a kernel that does almost nothing;
# * **bandwidth against blocks launched**: too little work in flight and memory sits idle (Little's law again).

# %%
try:
    import torch
    GPU = torch.cuda.is_available()
except ImportError:
    GPU = False
if not GPU:
    print("⏭ No GPU here, so part 4 is skipped (not passed). On Colab: Runtime → Change runtime type → T4 GPU.")
    ok4 = True
else:
    from torch.utils.cpp_extension import load_inline
    src = r"""
    __global__ void add_k(const float* a, const float* b, float* c, long n) {
        long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
        for (; i < n; i += (long)gridDim.x * blockDim.x) c[i] = a[i] + b[i];   // grid-stride loop
    }
    torch::Tensor add(torch::Tensor a, torch::Tensor b, int64_t blocks) {
        auto c = torch::empty_like(a);
        add_k<<<blocks, 256>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), a.numel());
        return c;
    }"""
    ext = load_inline("add60", cpp_sources="torch::Tensor add(torch::Tensor a, torch::Tensor b, int64_t blocks);",
                      cuda_sources=src, functions=["add"], verbose=False)
    n = 1 << 25
    a = torch.randn(n, device="cuda"); b = torch.randn_like(a)
    assert torch.allclose(ext.add(a, b, (n + 255) // 256), a + b), "kernel is wrong"
    print("your kernel matches torch")

    def timed(fn, reps=20):
        fn(); torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(reps): fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / reps

    t_gpu = timed(lambda: ext.add(a, b, (n + 255) // 256))
    hbm = 3 * n * 4 / t_gpu / 1e9
    host = torch.randn(n, pin_memory=True)
    t_pcie = timed(lambda: host.to("cuda", non_blocking=True), 10)
    pcie = n * 4 / t_pcie / 1e9
    tiny = torch.randn(32, device="cuda")
    t_launch = timed(lambda: ext.add(tiny, tiny, 1), 1000)
    print(f"on-GPU memory: {hbm:.0f} GB/s · PCIe: {pcie:.1f} GB/s · ratio {hbm / pcie:.0f}× · launch: {t_launch * 1e6:.1f} µs")

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"{'blocks':>7} {'GB/s':>7}")
    bw = {}
    for blocks in [1, sms // 4, sms, 4 * sms, 16 * sms, 64 * sms]:
        blocks = max(1, blocks); bw[blocks] = 3 * n * 4 / timed(lambda: ext.add(a, b, blocks), 5) / 1e9
        print(f"{blocks:>7} {bw[blocks]:>7.0f}")
    ok4 = hbm > 5 * pcie and bw[max(bw)] > 4 * bw[1]
if GPU: print("✅ the GPU's own memory is far faster than the trip to it, and needs lots of work in flight" if ok4 else "❌ check part 4")

# %% [markdown]
# ## Part 5: the whole trip in numbers
# The capstone chapter follows one multiply-add from Python to electrons. Recompute its key numbers from the
# H100's published specs plus Q = CV and E = CV², then the cost of one token for a 70B model as the batch grows.

# %%
SMS, TC_PER_SM, FMA_PER_CLK, CLK, WATTS = 132, 4, 512, 1.83e9, 700
fma_per_s = SMS * TC_PER_SM * FMA_PER_CLK * CLK
pj_per_fma = WATTS / fma_per_s * 1e12
print(f"peak: {2 * fma_per_s / 1e12:.0f} TFLOP/s, {pj_per_fma:.2f} pJ per multiply-add chip-wide")

C, V, q, kT = 0.2e-15, 0.75, 1.602e-19, 1.380649e-23 * 300
electrons, toggle, landauer = C * V / q, C * V * V, kT * math.log(2)
print(f"one bit: {electrons:.0f} electrons, {toggle * 1e18:.0f} aJ, {toggle / landauer:,.0f}x the Landauer limit")

def joules_per_token(params, batch, hbm_pj_per_bit=4, idle_w=150):
    gpus = math.ceil(2 * params / 70e9)
    step = max(2 * params / (gpus * 3.35e12), batch * 2 * params / (gpus * 989e12))
    math_j = params * pj_per_fma * 1e-12
    move_j = 2 * params * 8 * hbm_pj_per_bit * 1e-12 / batch
    idle_j = gpus * idle_w * step / batch
    return math_j + move_j + idle_j, math_j, move_j, step

print(f"{'batch':>5} {'J/token':>8} {'moving':>7} {'tokens/s':>9}")
costs = {}
for b in [1, 4, 16, 64, 256]:
    j, m, mv, step = joules_per_token(70e9, b); costs[b] = j
    print(f"{b:>5} {j:>8.3f} {mv / j:>7.0%} {b / step:>9,.0f}")
ok5 = 1.3 < pj_per_fma < 1.5 and 800 < electrons < 1100 and 1e4 < toggle / landauer < 1e5 and costs[1] > 20 * costs[256]
print("✅ a bit is ~1,000 electrons, and batching cuts the energy per token by >20x" if ok5 else "❌ check part 5")

# %% [markdown]
# ## What you measured
# | Layer | Claim | Your check |
# |---|---|---|
# | Electrons | free carriers grow exponentially with temperature | part 1 |
# | Transistor | ≥ 60 mV of gate per 10× current (× n for a real gate) | part 1 |
# | Logic | multipliers grow as bits² | part 2 |
# | GPU design | enough warps hide memory latency (Little's law) | part 3 |
# | CUDA | keep data on the GPU; launch enough blocks | part 4 |
# | The whole stack | a bit is ~1,000 electrons; one token's energy is mostly moving weights | part 5 |
#
# ## Stretch goals
# - Part 2: build a carry-lookahead adder from NANDs and compare its longest gate chain with the ripple adder's.
# - Part 3: give each warp two independent loads in flight (instruction-level parallelism). How many warps do you need now?
# - Part 4: time a 4096×4096 matmul in fp32 and fp16 on the T4. How close does fp16 get to the tensor cores' 65 TFLOP/s?
