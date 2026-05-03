# Batched Solving

`batch_solve(expr, lmo, X0)` runs `B` independent Frank-Wolfe problems on
an `(n, B)` matrix. Problems share one constraint set and step in lockstep;
columns that have converged are masked out of subsequent updates.

CPU and GPU arrays are accepted. GPU dispatch flows through
`KernelAbstractions.get_backend(X0)`; Marguerite picks up whichever KA
backend (`CUDA`, `Metal`, `AMDGPU`) is loaded. A `ChainRulesCore.rrule`
differentiates through the solution mapping, and `batch_bilevel_solve`
computes batched hypergradients.

## Basic usage

A [`BatchedExpression`](@ref) carries per-problem callbacks for the
objective and gradient. Each callback receives a single column view of
the iterate, not the whole `(n, B)` matrix.

```julia
using Marguerite, LinearAlgebra

n, B = 10, 50
H = Matrix{Float64}(I, n, n)
C = randn(n, B)

# Per-column objective and gradient — `θ` argument unused for non-parametric problems.
f_per_col(x, _, b) = 0.5 * dot(x, H * x) + dot(view(C, :, b), x)
grad_per_col!(g, x, _, b) = (g .= H * x .+ view(C, :, b); g)
expr = BatchedExpression(f_per_col, grad_per_col!)

X0 = fill(1.0 / n, n, B)
lmo = ProbSimplex()

X, result = batch_solve(expr, lmo, X0)
```

The return value is a [`BatchSolveResult`](@ref) supporting tuple unpacking.
`result` is a [`BatchResult`](@ref) with per-problem objectives, gaps,
convergence flags, and discard counts.

`f_per_col` and `grad_per_col!` must be column independent: the value and
gradient at column `b` depend only on `x_b` (the `b`-th column of the
iterate), not on the other columns. The implicit differentiation pipeline
relies on this.

A per-column gradient is required. There is no auto-gradient fallback for
batched mode; passing `nothing` for `grad_per_col!` is rejected at
construction.

## Configuration

Solver knobs are consolidated into a [`BatchSolveConfig`](@ref):

```julia
cfg = BatchSolveConfig(
    max_iters   = 5000,
    tol         = 1e-6,
    step_rule   = AdaptiveStepSize(),
    monotonic   = true,
    diff_lambda = 1e-4,
)
X, result = batch_solve(expr, lmo, X0; config=cfg)
```

Per-call keyword arguments override matching fields of `config`:

```julia
X, result = batch_solve(expr, lmo, X0; config=cfg, max_iters=10000)
```

## GPU support

Pass a GPU matrix directly. Marguerite reads its backend via
`KernelAbstractions.get_backend(X0)` and dispatches the LMO + gap kernel
accordingly:

```julia
using CUDA
X0_gpu = CUDA.fill(1.0f0 / n, n, B)
X, result = batch_solve(expr, ProbSimplex(1.0f0), X0_gpu)
```

Supported on device: [`ScalarBox`](@ref), [`Box`](@ref),
[`ProbSimplex`](@ref), [`Simplex`](@ref).

CPU only: [`Knapsack`](@ref), [`MaskedKnapsack`](@ref),
[`WeightedSimplex`](@ref), [`Spectraplex`](@ref), [`FunctionOracle`](@ref).
Calling these with a non-CPU backend raises `ArgumentError`.

`AdaptiveStepSize` is CPU only. Pass `step_rule=MonotonicStepSize()` (the
default) for GPU.

CI exercises Metal only; CUDA and AMDGPU paths are best-effort. Bug
reports welcome.

## Adaptive step size

```julia
cfg = BatchSolveConfig(step_rule=AdaptiveStepSize(1.0))
X, result = batch_solve(expr, lmo, X0; config=cfg)
```

Each problem maintains an independent Lipschitz estimate.

## Parametric problems

For problems parameterized by shared ``\theta``:

```math
\min_{x_b \in C(\theta)} f_b(x_b, \theta) \quad b = 1, \ldots, B
```

```julia
f_param(x, θ, b) = 0.5 * dot(x, H * x) - dot(θ, x)
grad_param!(g, x, θ, b) = (g .= H * x .- θ; g)
expr = BatchedExpression(f_param, grad_param!)
θ = randn(n)

X, result = batch_solve(expr, lmo, X0, θ)
```

If `lmo` is a [`ParametricOracle`](@ref), the constraint set is
materialized at ``\theta`` automatically.

## Implicit differentiation

The parametric `batch_solve` has a `ChainRulesCore.rrule`. The gradient
``\partial\theta`` is the sum of per-problem KKT adjoint contributions
(``\theta`` is shared across all problems):

```julia
using ChainRulesCore

(X_star, result), pb = rrule(batch_solve, expr, lmo, X0, θ)
dX = randn(size(X_star))
_, _, _, _, dθ = pb(dX)
```

The pullback runs `B` independent scalar pullbacks on the host (no
batched HVP, no batched factorization). The forward solve and the
gradient evaluations stay batched on the device; the KKT adjoint solves
serialize column by column. Per pullback work is `O(B)` adjoint solves on
length-`n` columns.

For the full Jacobian ``\partial X^* / \partial\theta``:

```julia
J, result = batch_solution_jacobian(expr, lmo, X0, θ)
```

## Bilevel optimization

`batch_bilevel_solve` computes hypergradients for batched bilevel problems:

```math
\min_\theta \sum_{b=1}^B L_b(x^*_b(\theta))
\quad \text{where} \quad
x^*_b(\theta) = \arg\min_{x \in C(\theta)} f_b(x, \theta)
```

Both outer and inner objectives use the `BatchedExpression` form:

```julia
outer_f(x, _, b) = sum(x .^ 2)
outer_grad!(g, x, _, b) = (g .= 2 .* x; g)
inner_f(x, θ, b) = 0.5 * dot(x, H * x) - dot(θ, x)
inner_grad!(g, x, θ, b) = (g .= H * x .- θ; g)

outer = BatchedExpression(outer_f, outer_grad!)
inner = BatchedExpression(inner_f, inner_grad!)

X, dθ, cg_results = batch_bilevel_solve(outer, inner, lmo, X0, θ)
```

`dθ` is the summed hypergradient across all problems. Per-problem CG
diagnostics are in `cg_results`.

## Pre-allocated cache

For repeated solves (e.g., inside a training loop), pre-allocate a
[`BatchCache`](@ref) to avoid repeated allocation:

```julia
cache = BatchCache(X0)
for epoch in 1:100
    X, result = batch_solve(expr, lmo, X0; cache=cache)
end
```

The cache infers its backend from `X0`. Reusing it with a differently
backed `X0` raises `ArgumentError`.

## Adaptive Tikhonov

`diff_lambda` is the starting Tikhonov regularization on the reduced
Hessian. If the residual ratio after the adjoint solve exceeds 1e-3,
Marguerite scales `lambda` by 5× (capped at 1.0), refactors, and resolves
once. The increased value persists across pullback calls on the same
state, so subsequent solves at the same problem benefit without an extra
retry.

## Performance tips

- Pre-allocate `BatchCache` to avoid `O(nB)` allocation per solve.
- Use `ScalarBox` or `ProbSimplex` on GPU: their kernels do the LMO and
  gap reduction in a single launch.
- For training loops where the next iterate is close to the previous
  solution, set `tol = 1e-3` rather than the default `1e-4`.

## Benchmarks

Numbers below are from `examples/bench_batched_oracles.jl` on a test
M-series Mac, Julia 1.12. The benchmark sweeps `(n, B, T, oracle)` and times
three conditions per cell at a fixed iteration regime (`tol=0`,
`max_iters=500`) so wall time differences reflect per-iter cost, not
convergence rate noise.

Conditions:
* `serial_cpu`  — `solve()` called `B` times sequentially
* `batched_cpu` — `batch_solve()` on a CPU `Matrix{T}`
* `batched_gpu` — `batch_solve()` on a device matrix. The numbers shown
  are for `MtlMatrix{T}` (Metal). On Metal, `Float64` is skipped because
  Apple Silicon GPU hardware does not support FP64; CUDA / AMDGPU support
  F64 natively.

The `+Accel` columns come from a second run with `BENCH_USE_ACCELERATE=1`,
which loads `AppleAccelerate` and forwards BLAS / LAPACK through Apple's
Accelerate framework (uses the AMX matrix coprocessor on Apple Silicon).

### When to use which backend

Crossover points are from one test machine; the regimes generalize across
vendors, the absolute numbers do not. For backend setup and per-vendor
support details, see the [GPU Backends](gpu_backends.md) page.

| Regime | Recommendation |
|---|---:|
| Small (`n × B < 10⁴`) | batched CPU. Serial is much slower; GPU overhead dominates. |
| Moderate F64 (`10⁴ ≤ n × B < 10⁶`) | batched CPU. On Apple Silicon, add `using AppleAccelerate` for ~1.3–2.4× on `mul!`. |
| Moderate F32 (`10⁴ ≤ n × B < 10⁶`) | batched CPU + `AppleAccelerate` on Apple Silicon. CUDA / AMDGPU expected to win earlier than Metal at this scale; measurements pending. |
| Large F32 (`n × B ≥ 10⁶`) | batched GPU. Metal: ~2–5× over CPU+Accel, ~50× over serial CPU. CUDA / AMDGPU expected similar. |
| F64 on Metal | Not supported (Apple Silicon GPU hardware limit). Use `Float32`. CUDA / AMDGPU support F64 natively. |

On Apple Silicon, prefer `Float32` when tolerance allows: AMX favors F32 and
Metal MPS kernels are tuned for F32. CUDA and AMDGPU handle both precisions
natively.

## API Reference

```@docs
BatchedExpression
BatchSolveConfig
batch_solve
batch_bilevel_solve
batch_bilevel_gradient
batch_solution_jacobian
BatchCache
BatchResult
BatchSolveResult
BatchBilevelResult
```
