```@meta
CurrentModule = Marguerite
```

# Marguerite.jl

*A minimal, performant, and differentiable Frank-Wolfe solver.*

Named after **Marguerite Frank** (1927--), co-inventor of the Frank-Wolfe algorithm (1956) -- an often-forgotten woman in optimization from an era when that was extraordinarily rare.

## What is Marguerite.jl?

Marguerite.jl provides a single entry point, [`solve`](@ref), for constrained convex minimization via the conditional gradient (Frank-Wolfe) method:

```math
\min_{x \in \mathcal{C}} f(x)
```

where ``\mathcal{C}`` is a compact convex set accessed through a **linear minimization oracle** (LMO).

### Key features

- **One function**: `solve(f, ∇f!, lmo, x0; ...)` -- that's the entire API
- **Five built-in oracles**: [`Simplex`](@ref), [`ProbabilitySimplex`](@ref), [`Knapsack`](@ref), [`Box`](@ref), [`WeightedSimplex`](@ref)
- **Zero-allocation inner loop**: Pre-allocated [`Cache`](@ref) buffers, `@inbounds` hot paths
- **Auto-gradient**: Optional automatic differentiation via [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl)
- **Differentiable solve**: `ChainRulesCore.rrule` for ``\partial x^* / \partial \theta`` via implicit differentiation
- **Bring your own oracle**: Any callable `(v, g) -> v` works -- no subtyping required

## Quick start

```julia
using Marguerite, LinearAlgebra

Q = [2.0 0.5; 0.5 1.0]; c = [-1.0, -0.5]
f(x) = 0.5 * dot(x, Q * x) + dot(c, x)
∇f!(g, x) = (g .= Q * x .+ c)

x, result = solve(f, ∇f!, ProbabilitySimplex(), [0.5, 0.5])
```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/samtalki/Marguerite.jl")
```
