# Apple Silicon

Marguerite runs on Apple Silicon out of the box. Two optional dependencies
substantially accelerate it on M-series Macs:

- [`AppleAccelerate.jl`](https://github.com/JuliaLinearAlgebra/AppleAccelerate.jl)
  forwards BLAS and LAPACK through Apple's Accelerate framework, including
  the AMX matrix coprocessor. It is loaded as a package extension — install
  `AppleAccelerate` in your environment and `using AppleAccelerate` from
  client code; Marguerite picks it up automatically.
- [`Metal.jl`](https://github.com/JuliaGPU/Metal.jl) provides GPU programming
  on Apple Silicon. Pass `MtlArray` inputs to `batch_solve` to dispatch the
  Metal-broadcast oracle paths.

```julia
using Marguerite
using AppleAccelerate     # Apple Silicon BLAS/LAPACK uplift
using Metal               # GPU; only if you want batched_solve on the device
```

You can verify forwarding after loading:

```julia
using LinearAlgebra
LinearAlgebra.BLAS.get_config()
# expects: LBTConfig([ILP64] ..., [LP64] Accelerate, [ILP64] Accelerate)
```

## Hardware notes (M5 Pro)

Numbers in this page are from an Apple M5 Pro:

* 12 P + 4 E CPU cores (18 reported logical threads)
* 16- or 20-core integrated GPU (`AGXG17SDevice`)
* 307 GB/s unified-memory bandwidth
* AMX matrix coprocessor accessed implicitly through Accelerate
* `Float32` is the GPU's native precision; `Float64` is **not supported by
  Metal** at all — `batch_solve` on `MtlMatrix{Float64}` errors at run time

## Backend decision tree

Crossover points are M5-Pro-specific but the qualitative picture should
generalize across M-series Macs. For NVIDIA / AMD GPUs the numbers will
differ; the broadcast-based GPU path is cross-vendor.

| Problem | Recommendation |
|---------|----------------|
| Small (`n × B < 10⁴`)              | batched CPU. Serial is much slower; GPU overhead dominates. |
| Moderate (`10⁴ ≤ n × B < 10⁶`), F64 | batched CPU + AppleAccelerate. |
| Moderate / large F32, `n × B ≥ 10⁶` | batched Metal. ~2–5× over CPU+Accelerate. |
| Spectraplex (any size)              | CPU. AppleAccelerate ≈ no help here (eigen is the bottleneck and Apple's `dsyevd`/`ssyevd` is not measurably faster than OpenBLAS at n ≤ 500). GPU eigen is future work. |
| Bilevel (`batch_bilevel_solve`)     | Today, `bilevel_solve` in a serial loop is **faster** than `batch_bilevel_solve` because the batched pullback is structurally inefficient — see Experiment 1 below. |

## Benchmarks

All numbers below are from `examples/bench_*.jl` scripts on M5 Pro, Julia 1.12.
Each cell is the median of 3 timed runs after a warmup. Convergence is at a
fixed iteration count (`tol=0`) so the comparison reflects per-iteration cost.

### Experiment 2 — batched broadcast oracles

See [Batched Solving](batching.md#benchmarks) for the full table.
Key result, n=10000, B=64, F32, ScalarBox:

| condition | wall (s) | speedup over serial OpenBLAS |
|-----------|---------:|------:|
| serial CPU (OpenBLAS)       | 281.8 | 1.0× |
| serial CPU (AppleAccelerate)| 115.2 | 2.4× |
| batched CPU (OpenBLAS)      |  20.2 | 14×  |
| batched CPU (AppleAccelerate)|  9.3 | 30×  |
| batched Metal (`MtlMatrix`)  |  4.2 | 67×  |

Headlines:
* `AppleAccelerate` gives **2.2–2.4×** on `mul!`-dominated paths (real AMX win).
* Metal beats CPU+Accelerate by **2.2× at n=10000** for `B=64`. Crossover is
  around `n × B ≈ 10⁶`.
* `Float32` is faster than `Float64` on every CPU path; on the GPU path
  `Float64` is not available at all.

### Experiment 1 — batched bilevel

**Today: serial loop is ~80–110× faster than `batch_bilevel_solve`. Use
`bilevel_solve` in a `for` loop.** The batched API works correctly; the
implementation needs a batched-pullback refactor before it pays off.

`examples/bench_batched_bilevel.jl` runs a Tikhonov-regularized QP on the
simplex with a closed-form interior bilevel gradient. Forward solve and
pullback are timed separately.

n=1000, B=64, 500 inner FW iters:

| T   | condition | forward (s) | pullback (s) |
|-----|-----------|------------:|-------------:|
| F32 | serial    |        0.07 |         0.30 |
| F32 | batched   |        0.05 |    **24.10** |
| F64 | serial    |        0.05 |         0.37 |
| F64 | batched   |        0.17 |    **39.94** |

Forward times are comparable; the batched pullback is 80-110× slower.
AppleAccelerate does not change this ratio — the bottleneck is Julia
closure overhead, not BLAS.

Root cause: the rrule pullback wraps each problem's scalar objective via
a closure that pads `x` into a zero `(n, B)` matrix and calls the user's
batched `f` / `grad_batch` on the full matrix to extract column `b`. Each
`DI.gradient` call thus runs `B ×` more work than a scalar `bilevel_solve`.
Buffer caching reclaims ~17% of allocations; the structural cost remains.

GPU bilevel on Metal F32 hits scalar-indexing errors deeper in the
per-problem KKT-adjoint path and is currently blocked.

Per-sample pullback time is computed by subtraction: `bilevel_solve(...)`
total minus a separate `solve(...)` forward run. ±5% measurement noise; the
80-110× ratio is robust to it.

### Experiment 3 — Spectraplex with AppleAccelerate

`examples/bench_spectraplex_apple.jl` runs a trace-regression problem on the
PSD cone:

```math
f(X) = \tfrac{1}{2}\|A X - B\|_F^2,\quad X \in \text{Spectraplex}(n, r=1).
```

Per FW iteration the solver evaluates the gradient (two matmuls,
`A^\top(A X - B)`) and runs the Spectraplex LMO (symmetrize + eigendecomposition
to extract the smallest eigenvector). Sweep `n ∈ {50, 100, 200}`,
`T ∈ {Float32, Float64}`, 200 fixed iterations.

Per-iteration breakdown, n=200:

| T   | condition       | grad (ms) | eigen (ms) | full iter (ms) |
|-----|-----------------|----------:|-----------:|---------------:|
| F32 | default         |     80.0  |    1867    |    2552 |
| F32 | AppleAccelerate |   **32.1** (2.49×) | 1862 (1.00×) | 2283 (1.12×) |
| F64 | default         |    133.8  |    2073    |    2419 |
| F64 | AppleAccelerate |   **92.8** (1.44×) | 1688 (1.23×) | 1949 (1.24×) |

Headlines:
* `AppleAccelerate` gives **2–3× on the matmul-heavy gradient** (consistent
  with Experiment 2 and the general AMX-via-Accelerate story).
* It does **not** measurably accelerate symmetric eigendecomposition at
  `n ≤ 500` — Apple's `dsyevd`/`ssyevd` ends up roughly equal to OpenBLAS,
  sometimes 5–10% slower in noise.
* Eigen dominates the FW iteration (~75–80% of per-iter cost), so the
  matmul speedup is diluted: total Spectraplex solve is only **1.0–1.24×**
  faster with AppleAccelerate.

The real lever for Spectraplex acceleration is moving the eigendecomposition
to the GPU (Metal MPS, cuSOLVER, or a cross-vendor Lanczos via
`KernelAbstractions.jl`). That work is outside the current branch and is
captured in `issues/future_gpu_optimizations.md` → "spectraplex GPU
eigendecomposition".

## Limitations & future work

What works on Apple Silicon today:
* CPU oracle paths (all oracles), with AppleAccelerate giving 1.3–2.4× on
  matmul-heavy workloads.
* `batch_solve` on `MtlMatrix{Float32}` for `ScalarBox`, `Box`, `ProbSimplex`,
  `Simplex`. ~50× over serial CPU at large `n × B`.

What does **not** work on Apple Silicon today:
* `batch_solve(::MtlMatrix{Float64})` — Metal hardware limitation.
* Knapsack, MaskedKnapsack, Spectraplex, FunctionOracle on GPU — CPU-only
  by design (sparse vertex protocol / eigendecomposition).
* `AdaptiveStepSize` on GPU arrays — CPU-only.
* `ForwardDiff` auto-gradient on GPU arrays — provide `grad_batch` manually.
* `batch_bilevel_solve` on real device arrays — additional scalar-indexing
  fixes needed in the pullback path.

Tracked future work lives in `issues/` at the repo root:

* `issues/future_gpu_optimizations.md` — spectraplex GPU eigen, GPU adaptive
  step, knapsack GPU sort, fused gap reduction, batched-pullback refactor,
  cross-vendor parity tracking.
* `issues/mixed_precision.md` — F32 / F64 / bfloat16 audit, tolerance
  scaling for differentiation, F32-inner / F64-outer hybrid.

## Reproducing the numbers

Each experiment lives as a single script under `examples/`:

* `bench_batched_oracles.jl` — Experiment 2
* `bench_batched_bilevel.jl` — Experiment 1
* `bench_spectraplex_apple.jl` — Experiment 3

The `examples/` directory has its own `Project.toml` so dependencies stay
isolated. To run everything end-to-end on Apple Silicon (assuming Marguerite
is checked out at `Marguerite.jl`):

```bash
cd Marguerite.jl
# Marguerite must be dev'd into examples/ once:
julia --project=examples -e 'using Pkg; Pkg.develop(path=".")'

# Run each experiment twice (default BLAS, then AppleAccelerate):
julia --project=examples examples/bench_batched_oracles.jl --grid=small
BENCH_USE_ACCELERATE=1 julia --project=examples examples/bench_batched_oracles.jl --grid=small

julia --project=examples examples/bench_batched_bilevel.jl --grid=small
BENCH_USE_ACCELERATE=1 julia --project=examples examples/bench_batched_bilevel.jl --grid=small

julia --project=examples examples/bench_spectraplex_apple.jl --grid=small
BENCH_USE_ACCELERATE=1 julia --project=examples examples/bench_spectraplex_apple.jl --grid=small
```

Each script writes a JSONL results file under `examples/results/` (gitignored)
with a hardware fingerprint header (`Sys.cpu_info()`, BLAS config, GPU device
descriptor, Julia version, timestamp) followed by one record per measurement.
