# GPU Backends

Marguerite's batched solver dispatches GPU paths via package extensions.
The supported oracles (`ScalarBox`, `Box`, `ProbSimplex`, `Simplex`) work on
any `AbstractMatrix` whose backend implements GPUArrays-style broadcast,
which means CUDA, AMDGPU, and Metal are all first-class. CPU-side, Apple
Silicon users can also opt into `AppleAccelerate.jl` for an AMX-backed
BLAS / LAPACK uplift.

This page covers backend setup, supported / unsupported features per
backend, the BLAS option for Apple Silicon, and a benchmark snapshot.

## Backends

Pass a device matrix to `batch_solve` and the GPU broadcast path is taken
automatically:

```julia
using Marguerite
using CUDA  # or Metal, or AMDGPU

X0 = CuArray(fill(1.0f0 / n, n, B))
X, result = batch_solve(f_batch, lmo, X0; grad_batch=grad_batch!)
```

### CUDA (NVIDIA)

```julia
using CUDA
```

Supported: `ScalarBox`, `Box`, `ProbSimplex`, `Simplex` on `CuArray{T}` for
both `Float32` and `Float64`. Sparse-vertex oracles
(`Knapsack`, `MaskedKnapsack`) and `Spectraplex` remain CPU-only by design
— they rely on the sparse-vertex protocol or dense eigendecomposition,
neither of which has a GPU broadcast implementation today. `AdaptiveStepSize`
on a `CuArray` is CPU-only. ForwardDiff auto-gradient on a `CuArray` is
unsupported; provide `grad_batch` manually.

### AMDGPU (AMD)

```julia
using AMDGPU
```

Same support matrix as CUDA: broadcast-friendly oracles (`ScalarBox`, `Box`,
`ProbSimplex`, `Simplex`) on `ROCArray{T}` for `Float32` / `Float64`.
Sparse-vertex and Spectraplex CPU-only, AdaptiveStepSize CPU-only,
ForwardDiff auto-grad CPU-only — same as CUDA.

### Metal (Apple Silicon)

```julia
using Metal
```

Supported oracles match the others. Metal hardware does **not** support
`Float64` — `batch_solve(::MtlMatrix{Float64})` raises at run time. Use
`Float32`. Beyond that, the same restrictions as CUDA / AMDGPU apply
(sparse-vertex CPU-only, AdaptiveStepSize CPU-only, no
`ForwardDiff`-on-device).

### Verifying a backend loaded

```julia
using Marguerite, CUDA  # or Metal / AMDGPU
X = CuArray(rand(Float32, 10, 4))
@assert Marguerite._array_style(X) === Marguerite._GPUStyle()
```

## Apple Silicon CPU acceleration

Independent of any GPU backend, Apple Silicon users can route BLAS / LAPACK
through Apple's Accelerate framework via
[`AppleAccelerate.jl`](https://github.com/JuliaLinearAlgebra/AppleAccelerate.jl):

```julia
using Marguerite
using AppleAccelerate
```

Loading `AppleAccelerate` redirects `mul!`, `eigen!`, etc. through Accelerate
(which uses the AMX matrix coprocessor on M-series chips). For Marguerite,
the matmul-heavy paths (gradient evaluation in batched solves and Spectraplex
trace regression) see a 1.3–2.4× lift; the symmetric eigensolver path
sees no measurable difference at `n ≤ 500`. Verify forwarding after loading:

```julia
using LinearAlgebra
LinearAlgebra.BLAS.get_config()
# expect: LBTConfig([ILP64] ..., [LP64] Accelerate, [ILP64] Accelerate)
```

`AppleAccelerate` is a `weakdep`. Loading it is enough — Marguerite's
extension picks it up automatically and there is no API change.

## Backend decision

Numbers below are from a single test machine (M-series Mac); CUDA / AMDGPU
relative speedups will differ but the qualitative regimes generalize.

| Regime | Recommendation |
|---|---|
| Small (`n × B < 10⁴`) | batched CPU. Serial is much slower; GPU launch overhead dominates at this scale. |
| Moderate F64 (`10⁴ ≤ n × B < 10⁶`) | batched CPU. On Apple Silicon, add `using AppleAccelerate` for ~1.3–2.4× on matmul. CUDA/AMDGPU will start to win in this range — measurements pending. |
| Moderate F32 (`10⁴ ≤ n × B < 10⁶`) | batched CPU + `AppleAccelerate` (Apple Silicon). On CUDA / AMDGPU the GPU path likely wins earlier here than at F64 — measurements pending. |
| Large F32 (`n × B ≥ 10⁶`) | batched GPU. Metal: ~2–5× over CPU+Accelerate, ~50× over serial CPU. CUDA/AMDGPU expected similar or larger. |
| F64 on Metal | Not supported by the hardware — use Float32. CUDA / AMDGPU support F64 normally. |
| Spectraplex (any backend) | CPU. The eigendecomposition is the bottleneck and there's no GPU eigen path yet — tracked future work. |

## Benchmarks

All numbers are from `examples/bench_*.jl` scripts on a test M-series Mac,
Julia 1.12. Each cell is the median of 3 timed runs after warmup, at a
fixed iteration count (`tol=0`) so wall-time differences reflect per-iter
cost. CUDA and AMDGPU rows are not yet measured — they show "—" pending
benchmark runs on representative hardware.

### Batched broadcast oracles (`Box`, `ProbSimplex`)

`examples/bench_batched_oracles.jl`. Quadratic on each oracle, sweep over
`(n, B, T)`.

n=10000, B=64, F32, ScalarBox:

| condition | wall (s) | speedup vs serial OpenBLAS |
|---|---:|---:|
| serial CPU (OpenBLAS) | 281.8 | 1.0× |
| serial CPU (`AppleAccelerate`) | 115.2 | 2.4× |
| batched CPU (OpenBLAS) | 20.2 | 14× |
| batched CPU (`AppleAccelerate`) | 9.3 | 30× |
| batched Metal | 4.2 | 67× |
| batched CUDA | — | — |
| batched AMDGPU | — | — |

Highlights:

- `AppleAccelerate` gives 2.2–2.4× on `mul!`-dominated paths (real AMX win).
- Metal beats CPU+`AppleAccelerate` by ~2.2× at `n=10000`, `B=64`. Crossover
  is around `n × B ≈ 10⁶`.
- `Float32` is faster than `Float64` on every CPU path; on Metal,
  `Float64` is unavailable.

### Spectraplex with `AppleAccelerate`

`examples/bench_spectraplex_apple.jl`. Trace-regression problem on the PSD
cone:

```math
f(X) = \tfrac{1}{2}\|A X - B\|_F^2,\quad X \in \text{Spectraplex}(n, r=1).
```

Per FW iteration: gradient (two matmuls) + Spectraplex LMO (symmetrize +
eigendecomposition). Sweep `n ∈ {50, 100, 200}`, `T ∈ {Float32, Float64}`,
200 fixed iterations.

Per-iteration breakdown, n=200:

| T | condition | grad (ms) | eigen (ms) | full iter (ms) |
|---|---|---:|---:|---:|
| F32 | default | 80.0 | 1867 | 2552 |
| F32 | `AppleAccelerate` | **32.1** (2.49×) | 1862 (1.00×) | 2283 (1.12×) |
| F64 | default | 133.8 | 2073 | 2419 |
| F64 | `AppleAccelerate` | **92.8** (1.44×) | 1688 (1.23×) | 1949 (1.24×) |

Headlines:

- `AppleAccelerate` gives 2–3× on the matmul-heavy gradient.
- Symmetric eigendecomposition does not measurably accelerate at
  `n ≤ 500` — Apple's `dsyevd` / `ssyevd` ends up roughly equal to
  OpenBLAS.
- Eigen dominates ~75–80% of per-iter cost, so the gradient speedup is
  diluted: total solve is only 1.0–1.24× faster with `AppleAccelerate`.

The lever for Spectraplex acceleration is GPU eigen (Metal MPS, cuSOLVER,
or a cross-vendor Lanczos). Tracked as future work.

## Limitations

Currently CPU-only on every backend (Metal, CUDA, AMDGPU):

- `Knapsack`, `MaskedKnapsack`, `Spectraplex`, `FunctionOracle` —
  sparse-vertex protocol or eigendecomposition is not on the GPU broadcast
  path.
- `AdaptiveStepSize` on a device array — per-problem backtracking is
  CPU-side.
- `ForwardDiff` auto-gradient on a device array — provide `grad_batch`
  manually.
- `batch_bilevel_solve` on a device array — scalar-indexing in the
  per-problem KKT-adjoint path. Tracked.

Apple-specific:

- `Float64` on Metal — Apple Silicon GPU hardware does not support FP64.
  Use `Float32` for the Metal path. CUDA / AMDGPU support `Float64` normally.

Future-work items (eigen on GPU, sparse-vertex sort on GPU, batched-pullback
refactor for `batch_bilevel_solve`, etc.) are tracked as GitHub issues —
search the issue tracker.

## Reproducing the numbers

```bash
cd Marguerite.jl

# Marguerite must be dev'd into examples/ once:
julia --project=examples -e 'using Pkg; Pkg.develop(path=".")'

# Add the platform-specific deps you want to exercise (see examples/Project.toml):
julia --project=examples -e 'using Pkg; Pkg.add("Metal")'              # Apple Silicon GPU
julia --project=examples -e 'using Pkg; Pkg.add("AppleAccelerate")'    # Apple Silicon CPU BLAS
julia --project=examples -e 'using Pkg; Pkg.add("CUDA")'               # NVIDIA GPU
julia --project=examples -e 'using Pkg; Pkg.add("AMDGPU")'             # AMD GPU

# Run each experiment twice (default BLAS, then with AppleAccelerate):
julia --project=examples examples/bench_batched_oracles.jl --grid=small
BENCH_USE_ACCELERATE=1 julia --project=examples examples/bench_batched_oracles.jl --grid=small

julia --project=examples examples/bench_spectraplex_apple.jl --grid=small
BENCH_USE_ACCELERATE=1 julia --project=examples examples/bench_spectraplex_apple.jl --grid=small
```

Each script writes a JSONL file under `examples/results/` with a hardware
fingerprint header (`Sys.cpu_info()`, BLAS config, GPU device descriptor,
Julia version, timestamp) followed by one record per measurement.
