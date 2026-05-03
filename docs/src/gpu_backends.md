# GPU Backends

Marguerite dispatches CPU vs GPU code paths through
`KernelAbstractions.get_backend(X)`. CUDA, AMDGPU, and Metal each register
their own `KernelAbstractions.Backend` through their package extensions;
Marguerite picks up whichever is loaded and has no per-vendor extension
code of its own. The broadcast oracles (`ScalarBox`, `Box`, `ProbSimplex`,
`Simplex`) compile to KA kernels at first call.

On Apple Silicon, `AppleAccelerate.jl` separately routes BLAS and LAPACK
through Apple's Accelerate framework (which uses the AMX matrix
coprocessor). This is independent of GPU dispatch and stays as a Marguerite
weakdep.

This page covers setup, the support matrix per backend, the Apple Silicon
BLAS path, and a benchmark snapshot.

## Backends

Pass a device matrix to `batch_solve`. Marguerite reads its backend via
`KernelAbstractions.get_backend(X0)` and launches the matching kernel:

```julia
using Marguerite
using CUDA  # or Metal, or AMDGPU

n, B = 10, 4
H = CUDA.fill(1.0f0, n, n)
X0 = CUDA.fill(1.0f0 / n, n, B)

f_per_col(x, _, b) = 0.5f0 * dot(x, H * x)
grad_per_col!(g, x, _, b) = (mul!(g, H, x); g)
expr = BatchedExpression(f_per_col, grad_per_col!)

X, result = batch_solve(expr, ProbSimplex(1.0f0), X0)
```

### CUDA (NVIDIA)

```julia
using CUDA
```

Supported on device: `ScalarBox`, `Box`, `ProbSimplex`, `Simplex` on
`CuArray{T}` for both `Float32` and `Float64`. The sparse vertex oracles
(`Knapsack`, `MaskedKnapsack`) and `Spectraplex` stay on the CPU — the
sparse vertex protocol and the dense eigendecomposition have no device
implementation today. `AdaptiveStepSize` on a `CuArray` is rejected at the
public API level. The rrule pipeline pulls each column to CPU before the
KKT adjoint solve, so cotangents flow through CPU code regardless of the
forward solve's device.

### AMDGPU (AMD)

```julia
using AMDGPU
```

Same support as CUDA: the broadcast oracles (`ScalarBox`, `Box`,
`ProbSimplex`, `Simplex`) on `ROCArray{T}` for `Float32` / `Float64`. Sparse
vertex oracles and `Spectraplex` stay on the CPU; `AdaptiveStepSize` is
rejected.

### Metal (Apple Silicon)

```julia
using Metal
```

Supported oracles match the others. Apple Silicon GPU hardware does **not**
support `Float64` — `batch_solve(::MtlMatrix{Float64})` raises at run time.
Use `Float32`. Other restrictions match CUDA and AMDGPU.

### Verifying a backend loaded

```julia
using Marguerite, CUDA  # or Metal / AMDGPU
using KernelAbstractions
X = CuArray(rand(Float32, 10, 4))
@assert KernelAbstractions.get_backend(X) isa KernelAbstractions.Backend
@assert !(KernelAbstractions.get_backend(X) isa KernelAbstractions.CPU)
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
the matmul paths (gradient evaluation in batched solves and the Spectraplex
trace regression) get a 1.3–2.4× lift; the symmetric eigensolver path shows
no measurable difference at `n ≤ 500`. Verify forwarding after loading:

```julia
using LinearAlgebra
LinearAlgebra.BLAS.get_config()
# expect: LBTConfig([ILP64] ..., [LP64] Accelerate, [ILP64] Accelerate)
```

`AppleAccelerate` is a `weakdep`. Loading it is enough — Marguerite's
extension picks it up automatically.

## Backend decision

Numbers below are from a single test machine (M-series Mac); CUDA / AMDGPU
relative speedups will differ but the qualitative regimes generalize.

| Regime | Recommendation |
|---|---|
| Small (`n × B < 10⁴`) | batched CPU. Serial is much slower; GPU launch overhead dominates at this scale. |
| Moderate F64 (`10⁴ ≤ n × B < 10⁶`) | batched CPU. On Apple Silicon, add `using AppleAccelerate` for ~1.3–2.4× on matmul. CUDA/AMDGPU will start to win in this range; measurements pending. |
| Moderate F32 (`10⁴ ≤ n × B < 10⁶`) | batched CPU + `AppleAccelerate` (Apple Silicon). On CUDA / AMDGPU the GPU path wins earlier here than at F64; measurements pending. |
| Large F32 (`n × B ≥ 10⁶`) | batched GPU. Metal: ~2–5× over CPU+Accelerate, ~50× over serial CPU. CUDA/AMDGPU expected similar or larger. |
| F64 on Metal | Not supported by the hardware — use Float32. CUDA / AMDGPU support F64 normally. |
| Spectraplex (any backend) | CPU. The eigendecomposition is the bottleneck and there is no GPU eigen path. |

## Benchmarks

All numbers are from `examples/bench_*.jl` scripts on a test M-series Mac,
Julia 1.12. Each cell is the median of 3 timed runs after warmup, at a
fixed iteration count (`tol=0`) so wall time differences reflect per iter
cost. CUDA and AMDGPU rows show "—" pending benchmark runs on those
vendors.

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

Notes:

- `AppleAccelerate` gives 2.2–2.4× on the `mul!` paths.
- Metal beats CPU + `AppleAccelerate` by ~2.2× at `n=10000`, `B=64`.
  Crossover is around `n × B ≈ 10⁶`.
- `Float32` is faster than `Float64` on every CPU path; on Metal, `Float64`
  is unavailable.

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

Notes:

- `AppleAccelerate` gives 2–3× on the matmul gradient.
- Symmetric eigendecomposition does not measurably accelerate at `n ≤ 500`
  — Apple's `dsyevd` / `ssyevd` ends up close to OpenBLAS.
- Eigen takes ~75–80% of per-iter cost, so the gradient speedup is diluted:
  total solve is 1.0–1.24× faster with `AppleAccelerate`.

The lever for Spectraplex acceleration is GPU eigen (Metal MPS, cuSOLVER,
or a cross vendor Lanczos).

## Limitations

The following currently run on the CPU regardless of which device backend
is loaded:

- `Knapsack`, `MaskedKnapsack`, `Spectraplex`, `FunctionOracle` — the
  sparse vertex protocol and the dense eigendecomposition have no device
  broadcast implementation.
- `AdaptiveStepSize` with a device array — the per-problem backtrack runs
  on the CPU.
- `ForwardDiff` auto-gradient with a device array — provide `grad_batch`
  manually.
- `batch_bilevel_solve` with a device array — the per-problem KKT adjoint
  path scalar-indexes on the CPU.

Apple Silicon specific:

- `Float64` on Metal — Apple Silicon GPU hardware does not support FP64.
  Use `Float32`. CUDA and AMDGPU support `Float64` normally.

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
