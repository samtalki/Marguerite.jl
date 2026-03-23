# Tutorial

## Basic usage

Minimize a quadratic on the probability simplex:

```julia
using Marguerite, LinearAlgebra

Q = [4.0 1.0; 1.0 2.0]; c = [-3.0, -1.0]
f(x) = 0.5 * dot(x, Q * x) + dot(c, x)
∇f!(g, x) = (g .= Q * x .+ c)

x, result = solve(f, ProbSimplex(), [0.5, 0.5]; grad=∇f!)
```

The return is a tuple `(x, result)` where `result::Result` contains diagnostics:

```julia
result.objective   # f(x*)
result.gap         # Frank-Wolfe duality gap
result.iterations  # iterations taken
result.converged   # gap ≤ tol * (1 + |f(x)|)
result.discards    # rejected monotonic updates
```

## Automatic gradient

Omit `grad=` and the gradient is computed automatically via ForwardDiff (the default backend):

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
x, result = solve(f, ProbSimplex(), [0.5, 0.5], θ; grad=∇f!)
```

This signature has a `ChainRulesCore.rrule` defined, so AD through `solve`
computes ``\partial x^* / \partial \theta`` via implicit differentiation.

## Custom oracles

Plain functions `(v, g) -> v` are auto-wrapped as `FunctionOracle` by `solve`.
Non-function callable structs should subtype `AbstractOracle` directly or be
wrapped explicitly with `FunctionOracle(my_callable)` for specialized dispatch
(e.g. `active_set`, sparse vertex protocol).

For **differentiated** calls (`solve(..., θ)`, `bilevel_solve`, `bilevel_gradient`),
custom oracles should also define [`active_set`](@ref). Otherwise Marguerite
throws an error by default; pass `assume_interior=true` only if you intentionally
want the interior approximation.

```julia
# L1 ball oracle: min ⟨g, v⟩ over {v : ||v||₁ ≤ 1}
function l1_ball!(v, g)
    fill!(v, 0.0)
    i = 1; best = abs(g[1])
    for j in 2:length(g)
        aj = abs(g[j])
        if aj > best; best = aj; i = j; end
    end
    v[i] = g[i] >= 0 ? -1.0 : 1.0
    return v
end

x, result = solve(f, l1_ball!, [0.0, 0.0]; grad=∇f!)
```

## Pre-allocated cache

For hot loops, pre-allocate buffers:

```julia
cache = Marguerite.Cache{Float64}(n)
for θ in parameter_sweep
    x, result = solve(f, lmo, x0, θ; grad=∇f!, cache=cache)
end
```
