<p align="center">
  <img src="docs/src/assets/logo.svg" width="200"/>
</p>

# Marguerite.jl

A minimal, differentiable Frank-Wolfe solver for constrained convex optimization in Julia.

Named in honor of [Marguerite Frank](https://en.wikipedia.org/wiki/Marguerite_Frank) (1927--2024), co-inventor of the Frank-Wolfe algorithm (1956).

[![Docs](https://img.shields.io/badge/docs-blue.svg)](https://samueltalkington.com/research/marguerite/)
[![Build Status](https://github.com/samtalki/Marguerite.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/samtalki/Marguerite.jl/actions/workflows/CI.yml?query=branch%3Amain)

## The Problem

$$\min_{x \in \mathcal{C}} f(x)$$

where $\mathcal{C}$ is a compact convex set. Frank-Wolfe solves this using a **linear minimization oracle** (LMO) -- no projections, just $\arg\min_{v \in \mathcal{C}} \langle g, v \rangle$ at each step.

## Quick Start

```julia
using Marguerite, LinearAlgebra

Q = [2.0 0.5; 0.5 1.0]; c = [-1.0, -0.5]
f(x) = 0.5 * dot(x, Q * x) + dot(c, x)
∇f!(g, x) = (g .= Q * x .+ c)

x, result = solve(f, ∇f!, ProbabilitySimplex(), [0.5, 0.5])
```

Or skip the gradient -- Marguerite computes it automatically via [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl):

```julia
x, result = solve(f, ProbabilitySimplex(), [0.5, 0.5])
```

## Features

- **One function, four signatures.** `solve` is the entire API. Manual or automatic gradients, with or without differentiable parameters.
- **Zero-allocation inner loop.** Pre-allocated `Cache` buffers, `@inbounds` hot paths.
- **Any callable oracle.** Any `(v, g) -> v` works -- no subtyping required. Five built-in oracles cover simplices, knapsacks, boxes, and weighted simplices. See the [oracle documentation](https://samueltalkington.com/research/marguerite/oracles/).
- **Differentiable solve.** `ChainRulesCore.rrule` computes $\partial x^{\ast} / \partial \theta$ via implicit differentiation -- no unrolling through iterations.
- **Bilevel optimization.** `bilevel_solve` backpropagates through the solver to learn parameters of constrained problems.

```julia
# Manual gradient:
x, result = solve(f, ∇f!, lmo, x0; max_iters=1000, tol=1e-7)

# Auto gradient (ForwardDiff default):
x, result = solve(f, lmo, x0)

# Parameterized (differentiable w.r.t. θ):
x, result = solve(f, ∇f!, lmo, x0, θ)

# Auto gradient + differentiable:
x, result = solve(f, lmo, x0, θ)
```

## Bilevel Optimization

$$\min_\theta \; L(x^{\ast}(\theta)) \quad \text{s.t.} \quad x^{\ast}(\theta) = \arg\min_{x \in \mathcal{C}} f(x, \theta)$$

```julia
using Marguerite, LinearAlgebra

f(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
∇f!(g, x, θ) = (g .= x .- θ)
outer_loss(x) = sum((x .- x_target) .^ 2)

x_star, θ_grad, cg_result = bilevel_solve(outer_loss, f, ∇f!, ProbSimplex(), x0, θ;
                                          max_iters=5000, tol=1e-6)
θ .-= η .* θ_grad  # ∇_θ L(x*(θ))
```

Exact gradients at convergence via the implicit function theorem. The Hessian system is solved by conjugate gradient with Hessian-vector products -- no explicit Hessian construction. See the [bilevel documentation](https://samueltalkington.com/research/marguerite/bilevel/).

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
