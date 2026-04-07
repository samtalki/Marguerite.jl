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
