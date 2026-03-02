<p align="center">
  <img src="resources/marguerite_logo.svg" width="200"/>
</p>

# Marguerite.jl

*Constrained convex optimization as easy as `x = A\b` — and bilevel, too. All in pure Julia.*

Named in honor of [Marguerite Frank](https://en.wikipedia.org/wiki/Marguerite_Frank) (1927–2024), co-inventor of the Frank-Wolfe algorithm (1956).

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://samtalki.github.io/Marguerite.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://samtalki.github.io/Marguerite.jl/dev/)
[![Build Status](https://github.com/samtalki/Marguerite.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/samtalki/Marguerite.jl/actions/workflows/CI.yml?query=branch%3Amain)

**[Documentation](https://samueltalkington.com/research/marguerite/)**

---

## The Problem

$$\min_{x \in \mathcal{C}} f(x)$$

where $\mathcal{C}$ is a compact convex set. Frank-Wolfe solves this using a **linear minimization oracle** (LMO): no projections, just $\arg\min_{v \in \mathcal{C}} \langle g, v \rangle$ at each step.

## Quick Start

```julia
using Marguerite, LinearAlgebra

Q = [2.0 0.5; 0.5 1.0]; c = [-1.0, -0.5]
f(x) = 0.5 * dot(x, Q * x) + dot(c, x)
∇f!(g, x) = (g .= Q * x .+ c)

x, result = solve(f, ∇f!, ProbabilitySimplex(), [0.5, 0.5])
```

Or skip the gradient — Marguerite computes it automatically via [Mooncake](https://github.com/compintell/Mooncake.jl):

```julia
x, result = solve(f, ProbabilitySimplex(), [0.5, 0.5])
```

## Why Marguerite?

- **One function, four signatures:** `solve` is the entire API
- **100% pure Julia:** easy to read, audit, and extend
- **Zero-allocation inner loop:** pre-allocated buffers, `@inbounds` hot paths
- **Any callable `(v, g) -> v` works as an oracle:** no subtyping required
- **Differentiable solve:** `ChainRulesCore.rrule` for $\partial x^{\ast} / \partial \theta$ via implicit differentiation
- **Bilevel optimization:** learn parameters of constrained problems by backpropagating through the solver

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

## Built-in Oracles

| Oracle | Constraint Set | Complexity |
|--------|---------------|------------|
| `Simplex(r)` | $x \geq 0, \sum x_i \leq r$ | $O(n)$ |
| `ProbSimplex(r)` | $x \geq 0, \sum x_i = r$ | $O(n)$ |
| `Knapsack(budget, m)` | $x \in [0,1]^m, \sum x_i \leq q$ | $O(m \log q)$ |
| `Box(lb, ub)` | $\ell_i \leq x_i \leq u_i$ | $O(n)$ |
| `WeightedSimplex(α, β, lb)` | $x \geq \ell, \alpha^\top x \leq \beta$ | $O(m)$ |

Any callable `(v, g) -> v` also works as an oracle, no subtyping required.

## Bilevel Optimization

$$\min_\theta \; L(x^{\ast}(\theta)) \quad \text{s.t.} \quad x^{\ast}(\theta) = \arg\min_{x \in \mathcal{C}} f(x, \theta)$$

`bilevel_solve` computes the gradient of an outer loss through the inner Frank-Wolfe solve. No unrolling through iterations. Exact gradients at convergence.

```julia
using Marguerite, LinearAlgebra

f(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
∇f!(g, x, θ) = (g .= x .- θ)
outer_loss(x) = sum((x .- x_target) .^ 2)

x_star, θ_grad, cg_result = bilevel_solve(outer_loss, f, ∇f!, ProbSimplex(), x0, θ;
                                          max_iters=5000, tol=1e-6)
θ .-= η .* θ_grad  # ∇_θ L(x*(θ))
```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/samtalki/Marguerite.jl")
```

## References

- M. Frank & P. Wolfe, "An algorithm for quadratic programming," *Naval Research Logistics*, 1956.
- S. Carderera, M. Besançon & S. Pokutta, "Scalable Frank-Wolfe on Generalized Self-concordant Functions via Simple Steps," 2024.
- S. Lacoste-Julien & M. Jaggi, "On the Global Linear Convergence of Frank-Wolfe Optimization Variants," 2015.
- A. Palmieri, F. Rinaldi, S. Salzo & S. Venturini, "Iteration Complexity of Frank-Wolfe and Its Variants for Bilevel Optimization," 2026.
