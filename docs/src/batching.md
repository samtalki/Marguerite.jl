# Batched Solving

Marguerite provides first-class support for solving multiple independent
Frank-Wolfe problems simultaneously via `batch_solve`. All problems share
the same constraint set (oracle) and run in lockstep, with per-problem
convergence masking so that converged problems are skipped.

Batched solving supports CPU and GPU arrays, automatic differentiation
through the solution via `ChainRulesCore.rrule`, and bilevel optimization
via `batch_bilevel_solve`.

## Basic Usage

```julia
using Marguerite, LinearAlgebra

n, B = 10, 50
H = Matrix{Float64}(I, n, n)
C = randn(n, B)   # per-problem linear terms

# Batched objective: f_b(x) = 0.5 x'Hx + c_b'x
f_batch(X) = [0.5 * dot(X[:, b], H * X[:, b]) + dot(C[:, b], X[:, b]) for b in 1:B]
grad_batch!(G, X) = (G .= H * X .+ C)

X0 = fill(1.0 / n, n, B)
lmo = ProbSimplex()

X, result = batch_solve(f_batch, lmo, X0; grad_batch=grad_batch!)
```

The return value is a [`BatchSolveResult`](@ref) supporting tuple unpacking.
`result` is a [`BatchResult`](@ref) with per-problem objectives, gaps,
convergence flags, and discard counts.

## Auto-Gradient

When `grad_batch` is omitted, Marguerite computes gradients automatically
using `DifferentiationInterface` (ForwardDiff by default):

```julia
X, result = batch_solve(f_batch, lmo, X0)
```

Auto-gradient is CPU-only. For GPU arrays, provide `grad_batch` explicitly.

## GPU Support

Pass GPU matrices directly. Marguerite dispatches GPU-compatible broadcast
operations for supported oracles:

```julia
using CUDA

X0_gpu = CUDA.fill(1.0f0 / n, n, B)
X, result = batch_solve(f_batch, lmo, X0_gpu; grad_batch=grad_batch!)
```

**GPU-supported oracles**: [`ScalarBox`](@ref), [`Box`](@ref),
[`ProbSimplex`](@ref), [`Simplex`](@ref).

**CPU-only oracles**: [`Knapsack`](@ref), [`MaskedKnapsack`](@ref),
[`Spectraplex`](@ref), [`FunctionOracle`](@ref).

## Adaptive Step Size

Pass `step_rule=AdaptiveStepSize()` for per-problem backtracking line
search. Each problem maintains an independent Lipschitz estimate:

```julia
X, result = batch_solve(f_batch, lmo, X0;
                         grad_batch=grad_batch!,
                         step_rule=AdaptiveStepSize(1.0))
```

`AdaptiveStepSize` is CPU-only.

## Parametric Problems

For problems parameterized by shared ``\theta``:

```math
\min_{x_b \in C(\theta)} f_b(x_b, \theta) \quad b = 1, \ldots, B
```

```julia
f_param(X, θ) = [0.5 * dot(X[:, b], H * X[:, b]) - dot(θ, X[:, b]) for b in 1:B]
grad_param!(G, X, θ) = (G .= H * X .- θ)
θ = randn(n)

X, result = batch_solve(f_param, lmo, X0, θ; grad_batch=grad_param!)
```

If `lmo` is a [`ParametricOracle`](@ref), the constraint set is
materialized at ``\theta`` automatically.

## Implicit Differentiation

The parametric `batch_solve` has a `ChainRulesCore.rrule` for automatic
differentiation through the solution mapping ``\theta \mapsto X^*(\theta)``.

The gradient ``\partial\theta`` is the sum of per-problem KKT adjoint
contributions, since ``\theta`` is shared across all problems:

```julia
using ChainRulesCore

(X_star, result), pb = rrule(batch_solve, f_param, lmo, X0, θ;
                              grad_batch=grad_param!)
dX = randn(size(X_star))  # cotangent
_, _, _, _, dθ = pb(dX)
```

For the full Jacobian ``\partial X^* / \partial\theta``:

```julia
J, result = batch_solution_jacobian(f_param, lmo, X0, θ; grad_batch=grad_param!)
```

## Bilevel Optimization

`batch_bilevel_solve` computes hypergradients for batched bilevel problems:

```math
\min_\theta \sum_{b=1}^B L_b(x^*_b(\theta))
\quad \text{where} \quad
x^*_b(\theta) = \arg\min_{x \in C(\theta)} f_b(x, \theta)
```

```julia
outer_batch(X) = [sum(X[:, b] .^ 2) for b in 1:B]
inner_batch(X, θ) = f_param(X, θ)

X, dθ, cg_results = batch_bilevel_solve(outer_batch, inner_batch, lmo, X0, θ;
                                          grad_batch=grad_param!)
```

The `dθ` is the summed hypergradient across all problems. Per-problem
CG diagnostics are in `cg_results`.

## Pre-allocated Cache

For repeated solves (e.g., inside a training loop), pre-allocate a
[`BatchCache`](@ref) to avoid repeated allocation:

```julia
cache = BatchCache(X0)
for epoch in 1:100
    X, result = batch_solve(f_batch, lmo, X0; grad_batch=grad_batch!, cache=cache)
end
```

## Performance Tips

- **Pre-allocate `BatchCache`**: avoids ``O(nB)`` allocation per solve.
- **Provide `grad_batch`**: manual gradients avoid per-column AD overhead.
- **Use `ScalarBox` or `ProbSimplex` on GPU**: these have fully broadcast
  implementations with no scalar indexing.
- **Tune `tol`**: loose tolerance (e.g., `1e-3`) converges much faster for
  warm-started training loops.

## Benchmarks

Numbers below are from `examples/bench_batched_oracles.jl` on an Apple M5 Pro
(18 CPU threads, 16/20-core integrated GPU, 307 GB/s memory bandwidth) with
Julia 1.12. The benchmark sweeps `(n, B, T, oracle)` and times three conditions
per cell at a fixed-iteration regime (`tol=0`, `max_iters=500`) so wall-time
differences reflect per-iter cost, not convergence-rate noise.

Conditions:
* `serial_cpu`  — `solve()` called `B` times sequentially
* `batched_cpu` — `batch_solve()` on a CPU `Matrix{T}`
* `batched_gpu` — `batch_solve()` on a `MtlMatrix{T}` (Metal). `Float64` is
  skipped because Metal does not support `Float64` on the GPU.

The `accel` columns are taken from a second run with `BENCH_USE_ACCELERATE=1`,
which loads `AppleAccelerate` and forwards BLAS/LAPACK through Apple's
Accelerate framework (uses the AMX matrix coprocessor).

### ScalarBox `Box(0, 1)` — quadratic on the unit box

| n × B          | T   | serial | batched | +Accel | Metal |
|----------------|-----|-------:|--------:|-------:|------:|
| 100 × 64       | F32 |  29 ms |    11 ms |  4.5 ms |  537 ms |
| 100 × 1024     | F32 | 473 ms |   101 ms | 65.6 ms |  509 ms |
| 1000 × 256     | F32 | 16.85 s |  1.21 s | 0.35 s | 0.55 s |
| 10000 × 64     | F32 | 281.8 s |  20.2 s |  9.3 s |  4.2 s |
| 10000 × 64     | F64 | 372.4 s |  39.9 s | 31.5 s |   skip |

### Probability simplex `ProbSimplex()` — quadratic on the simplex

| n × B          | T   | serial | batched | +Accel | Metal |
|----------------|-----|-------:|--------:|-------:|------:|
| 100 × 64       | F32 |  38 ms |    13 ms |  6.8 ms |  728 ms |
| 1000 × 256     | F32 | 18.81 s |  1.29 s | 0.45 s | 0.77 s |
| 10000 × 64     | F32 | 381.1 s |  24.9 s | 11.7 s |  8.6 s |
| 10000 × 64     | F64 | 525.6 s |  49.8 s | 39.1 s |   skip |

### When to use which backend

Crossover points are M5-Pro-specific; the qualitative picture should
generalize across Apple Silicon and to NVIDIA / AMD GPUs (relative numbers
will differ).

| Regime | Recommendation |
|---|---|
| Small (`n × B < 10⁴`) | batched CPU. Serial is much slower; GPU overhead dominates. |
| Moderate F64 (`10⁴ ≤ n × B < 10⁶`) | batched CPU. Add `using AppleAccelerate` on Apple Silicon — typical 1.3–2.4× lift on `mul!`. |
| Moderate F32 (`10⁴ ≤ n × B < 10⁶`) | batched CPU + `AppleAccelerate`. Metal is 0.5–0.9× of CPU+Accel here. |
| Large F32 (`n × B ≥ 10⁶`) | batched Metal. ~2–5× over CPU+Accel; ~50× over serial CPU. |
| Any F64 on the GPU | Not supported (Metal hardware limit). Use `Float32`. |

`Float32` is generally the right choice on Apple Silicon when tolerance
allows: AMX favors F32, Metal MPS-class kernels are best at F32.

For `batch_bilevel_solve` and the Spectraplex oracle, see
[Apple Silicon notes](apple_silicon.md).

## API Reference

```@docs
batch_solve
batch_bilevel_solve
batch_bilevel_gradient
batch_solution_jacobian
BatchCache
BatchResult
BatchSolveResult
BatchBilevelResult
```
