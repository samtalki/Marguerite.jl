<!-- Copyright 2026 Samuel Talkington and contributors
   SPDX-License-Identifier: Apache-2.0 -->

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

x, result = solve(f, ∇f!, ProbSimplex(), [0.5, 0.5];
                   max_iters=10000, tol=1e-3)
```

Omit `∇f!` for automatic differentiation via ForwardDiff.

## Documentation guide

- **[Tutorial](@ref)** — basic usage, automatic gradients, parameterized solve, custom oracles
- **[Oracles](@ref "Linear Oracles")** — built-in constraint sets and the oracle interface
- **[Examples](@ref)** — sparse recovery benchmark (Frank-Wolfe vs. interior point)
- **[Convergence](@ref)** — ``O(1/t)`` rate, Frank-Wolfe gap, sparsity plots
- **[Implicit Differentiation](@ref)** — implicit differentiation, rrule, CG tuning
- **[Bilevel Optimization](@ref)** — `bilevel_solve`, `bilevel_gradient`, manual rrule usage
- **[API Reference](@ref)** — complete docstrings for all exports

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/samtalki/Marguerite.jl")
```
