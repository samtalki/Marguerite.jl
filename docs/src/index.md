```@meta
CurrentModule = Marguerite
```

# Marguerite.jl

A minimal, differentiable Frank-Wolfe solver for constrained convex optimization in Julia.

Named in honor of [Marguerite Frank](https://en.wikipedia.org/wiki/Marguerite_Frank) (1927–2024), co-inventor of the Frank-Wolfe algorithm (1956).

## Quick start

```julia
using Marguerite, LinearAlgebra

Q = [4.0 1.0; 1.0 2.0]; c = [-3.0, -1.0]
f(x) = 0.5 * dot(x, Q * x) + dot(c, x)
∇f!(g, x) = (g .= Q * x .+ c)

x, result = solve(f, ProbSimplex(), [0.5, 0.5];
                   grad=∇f!, max_iters=10000, tol=1e-3)
```

Omit `grad=` for automatic differentiation via ForwardDiff.

## Documentation guide

- **[Tutorial](@ref)** — basic usage, automatic gradients, parameterized solve, custom oracles
- **[Oracles](@ref)** — built-in constraint sets and the oracle interface
- **[Parametric Oracles](@ref)** — differentiable constraint sets parameterized by θ
- **[Examples](@ref)** — sparse recovery benchmark (Frank-Wolfe vs. interior point)
- **[Convergence](@ref)** — ``O(1/t)`` rate, Frank-Wolfe gap, sparsity plots
- **[Implicit Differentiation](@ref)** — implicit differentiation, rrule, CG tuning
- **[Bilevel Optimization](@ref)** — `bilevel_solve`, `bilevel_gradient`, manual rrule usage
- **[API Reference](@ref)** — complete docstrings for all exports

## Installation

Once registered in the Julia General registry:

```julia
using Pkg
Pkg.add("Marguerite")
```

Until then, install directly from the repository:

```julia
using Pkg
Pkg.add(url="https://github.com/samtalki/Marguerite.jl")
```
