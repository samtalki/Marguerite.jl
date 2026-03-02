<p align="center">
  <img src="resources/marguerite_logo.svg" width="200"/>
</p>

# Marguerite.jl

*A minimal, performant, and differentiable Frank-Wolfe solver.*

Named after **Marguerite Frank** (1927--), co-inventor of the Frank-Wolfe algorithm (1956). An often-forgotten woman in optimization from an era when that was extraordinarily rare. Also a flower, fitting Julia's botanical naming tradition.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://samtalki.github.io/Marguerite.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://samtalki.github.io/Marguerite.jl/dev/)
[![Build Status](https://github.com/samtalki/Marguerite.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/samtalki/Marguerite.jl/actions/workflows/CI.yml?query=branch%3Amain)

**[Documentation](https://samueltalkington.com/research/marguerite/)**

## Quick Start

```julia
using Marguerite, LinearAlgebra

Q = [2.0 0.5; 0.5 1.0]; c = [-1.0, -0.5]
f(x) = 0.5 * dot(x, Q * x) + dot(c, x)
∇f!(g, x) = (g .= Q * x .+ c)

x, result = solve(f, ∇f!, ProbabilitySimplex(), [0.5, 0.5])
```

Or skip the gradient -- Marguerite computes it automatically via Mooncake:

```julia
x, result = solve(f, ProbabilitySimplex(), [0.5, 0.5])
```

## Features

### One function, four signatures

```julia
# Manual gradient:
x, result = solve(f, ∇f!, lmo, x0; max_iters=1000, tol=1e-7)

# Auto gradient (Mooncake default):
x, result = solve(f, lmo, x0)

# Parameterized (differentiable w.r.t. θ):
x, result = solve(f, ∇f!, lmo, x0, θ)

# Auto gradient + differentiable:
x, result = solve(f, lmo, x0, θ)
```

### Built-in oracles

| Oracle | Constraint Set | Complexity |
|--------|---------------|------------|
| `Simplex(r)` | $x \geq 0, \sum x_i \leq r$ | $O(n)$ |
| `ProbSimplex(r)` | $x \geq 0, \sum x_i = r$ | $O(n)$ |
| `Knapsack(q, backbone, m)` | $x \in [0,1]^m, \sum x_i \leq q$, backbone fixed | $O(m \log k)$ |
| `Box(lb, ub)` | $\ell_i \leq x_i \leq u_i$ | $O(n)$ |
| `WeightedSimplex(α, β, lb)` | $x \geq \ell, \alpha^\top x \leq \beta$ | $O(m)$ |

Any callable `(v, g) -> v` also works as an oracle -- no subtyping required.

### Implicit differentiation

When parameters `θ` are passed, a `ChainRulesCore.rrule` computes $\partial x^* / \partial \theta$ via the implicit function theorem. The Hessian system is solved by conjugate gradient with Hessian-vector products (no explicit Hessian). O(1) memory in the backward pass.

### Bilevel optimization

`bilevel_solve` computes the gradient of an outer loss through the inner Frank-Wolfe solve:

```julia
using Marguerite, LinearAlgebra

f(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
∇f!(g, x, θ) = (g .= x .- θ)
outer_loss(x) = sum((x .- x_target) .^ 2)

x_star, θ̄, cg_result = bilevel_solve(outer_loss, f, ∇f!, ProbSimplex(), x0, θ;
                                       max_iters=5000, tol=1e-6)
θ .-= η .* θ̄  # gradient step on outer parameters
```

No unrolling through iterations. Exact gradients at convergence.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/samtalki/Marguerite.jl")
```

## References

- M. Frank & P. Wolfe, "An algorithm for quadratic programming," *Naval Research Logistics*, 1956.
- S. Carderera, M. Besançon & S. Pokutta, "Scalable Frank-Wolfe on Generalized Self-concordant Functions via Simple Steps," 2024.
- S. Lacoste-Julien & M. Jaggi, "On the Global Linear Convergence of Frank-Wolfe Optimization Variants," 2015.
- A. Palmieri, M. Rinaldi & F. Salzo, "On the Use of the Frank-Wolfe Algorithm for Bilevel Optimization," 2024.
