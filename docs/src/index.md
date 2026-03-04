```@meta
CurrentModule = Marguerite
```

# Marguerite.jl

A minimal, differentiable Frank-Wolfe solver for constrained convex optimization in Julia.

Named in honor of [Marguerite Frank](https://en.wikipedia.org/wiki/Marguerite_Frank) (1927–2024), co-inventor of the Frank-Wolfe algorithm (1956).

## What is Marguerite.jl?

Marguerite.jl provides a single entry point, [`solve`](@ref), for constrained convex minimization via the conditional gradient (Frank-Wolfe) method:

```math
\min_{x \in \mathcal{C}} f(x)
```

where ``\mathcal{C}`` is a compact convex set accessed through a **linear minimization oracle** (LMO). 100% pure Julia.

### Key features

- **One function:** `solve(f, ∇f!, lmo, x0; ...)`, that's the entire API
- **100% pure Julia:** easy to read, audit, and extend
- **Six built-in oracles:** [`Simplex`](@ref) (+ [`ProbabilitySimplex`](@ref) alias), [`Knapsack`](@ref), [`MaskedKnapsack`](@ref), [`Box`](@ref), [`WeightedSimplex`](@ref)
- **Parametric constraint sets:** [`ParametricBox`](@ref), [`ParametricSimplex`](@ref), [`ParametricWeightedSimplex`](@ref) — differentiate through constraint parameters via KKT adjoint
- **Zero-allocation inner loop:** pre-allocated [`Cache`](@ref) buffers, `@inbounds` hot paths
- **Auto-gradient:** optional automatic differentiation via [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) through [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl)
- **Differentiable solve:** `ChainRulesCore.rrule` for ``\partial x^* / \partial \theta`` via implicit differentiation, with KKT adjoint for boundary solutions
- **Bring your own oracle:** any callable `(v, g) -> v` works, no subtyping required

### The killer app: bilevel optimization

Marguerite's differentiable `solve` enables **bilevel optimization** — learning
parameters of constrained problems by backpropagating through the solver.
No unrolling. ``O(1)`` memory. Exact gradients at convergence.
With [parametric constraint sets](@ref "Parametric Oracles"), you can also differentiate
through ``\mathcal{C}(\theta)`` itself.

```math
\min_\theta \; L(x^*(\theta)) \quad \text{s.t.} \quad x^*(\theta) = \arg\min_{x \in \mathcal{C}(\theta)} f(x, \theta)
```

```julia
using Marguerite, LinearAlgebra

f(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
∇f!(g, x, θ) = (g .= x .- θ)
outer_loss(x) = sum((x .- x_target) .^ 2)

x_star, θ_grad, cg_result = bilevel_solve(outer_loss, f, ∇f!, ProbSimplex(), x0, θ;
                                          max_iters=5000, tol=1e-6)
θ .-= η .* θ_grad  # ∇_θ L(x*(θ))
```

See [Bilevel Optimization](@ref) for a fully-worked example.

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
