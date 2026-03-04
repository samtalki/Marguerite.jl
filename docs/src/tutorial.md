# Tutorial

## Basic usage

Minimize a quadratic on the probability simplex:

```julia
using Marguerite, LinearAlgebra

Q = [4.0 1.0; 1.0 2.0]; c = [-3.0, -1.0]
f(x) = 0.5 * dot(x, Q * x) + dot(c, x)
∇f!(g, x) = (g .= Q * x .+ c)

x, result = solve(f, ∇f!, ProbSimplex(), [0.5, 0.5];
                   max_iters=10000, tol=1e-3)
```

The return is a tuple `(x, result)` where `result::Result` contains diagnostics:

```julia
result.objective   # f(x*)
result.gap         # Frank-Wolfe duality gap
result.iterations  # iterations taken
result.converged   # gap ≤ tol * |f(x)|
result.discards    # rejected monotonic updates
```

## Automatic gradient

Omit `∇f!` and the gradient is computed automatically via ForwardDiff (the default backend):

```julia
x, result = solve(f, ProbSimplex(), [0.5, 0.5])
```

Pass `backend=` to use a different [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl) backend.
See [Implicit Differentiation](@ref) for details on AD backend selection.

## Parameterized solve (differentiable)

When `f` depends on parameters `θ`, pass them as a positional argument:

```julia
f(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
∇f!(g, x, θ) = (g .= x .- θ)

θ = [0.8, 0.2]
x, result = solve(f, ∇f!, ProbSimplex(), [0.5, 0.5], θ;
                   max_iters=10000, tol=1e-4)
```

This signature has a `ChainRulesCore.rrule` defined, so AD through `solve`
computes ``\partial x^* / \partial \theta`` via implicit differentiation.

## Custom oracles

Any callable `(v, g) -> v` works as an oracle:

```julia
# L1 ball oracle
function l1_ball!(v, g)
    fill!(v, 0.0)
    i = argmin(g)
    v[i] = g[i] < 0 ? 1.0 : -1.0
    return v
end

x, result = solve(f, ∇f!, l1_ball!, [0.0, 0.0])
```

## Pre-allocated cache

For hot loops, pre-allocate buffers:

```julia
cache = Marguerite.Cache{Float64}(n)
for θ in parameter_sweep
    x, result = solve(f, ∇f!, lmo, x0, θ; cache=cache)
end
```
