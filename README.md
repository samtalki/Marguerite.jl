<p align="center">
  <img src="docs/src/assets/logo.svg" width="320"/>
</p>

# Marguerite.jl

A minimal, differentiable Frank-Wolfe solver for constrained convex optimization in Julia.

Named in honor of [Marguerite Frank](https://en.wikipedia.org/wiki/Marguerite_Frank) (1927--2024), co-inventor of the Frank-Wolfe algorithm (1956).

[![Docs](https://img.shields.io/badge/docs-blue.svg)](https://samueltalkington.com/research/marguerite/)
[![Build Status](https://github.com/samtalki/Marguerite.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/samtalki/Marguerite.jl/actions/workflows/CI.yml?query=branch%3Amain)

Solves constrained convex programs of the form

$$\min_{x \in \mathcal{C}} f(x)$$

where $\mathcal{C}$ is a compact convex set accessed through a **linear minimization oracle** (LMO).

## Quick Start

With a user-provided gradient:

```julia
using Marguerite, LinearAlgebra

Q = [4.0 1.0; 1.0 2.0]; c = [-3.0, -1.0]
f(x) = 0.5 * dot(x, Q * x) + dot(c, x)
∇f!(g, x) = (g .= Q * x .+ c)

x, result = solve(f, ∇f!, ProbabilitySimplex(), [0.5, 0.5])
```

Omit `∇f!` for automatic differentiation via [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl). For bilevel optimization, `bilevel_solve` differentiates through the solver to compute $\nabla_\theta L(x^*(\theta))$:

```julia
f(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
∇f!(g, x, θ) = (g .= x .- θ)
outer_loss(x) = sum((x .- x_target).^2)

x_star, θ_grad, _ = bilevel_solve(outer_loss, f, ∇f!, ProbabilitySimplex(), x0, θ)
θ .-= η .* θ_grad
```

## Features

- Single entry point: `solve(f, ∇f!, lmo, x0; ...)`, with or without automatic gradients and differentiable parameters
- Pre-allocated buffers for allocation-free inner loops (`@inbounds` hot paths)
- Six built-in oracles: simplex, probability simplex, knapsack, masked knapsack, box, weighted simplex
- Custom oracles: any `(v, g) -> v` callable works, no subtyping required
- Differentiable solve via `ChainRulesCore.rrule` for $\partial x^* / \partial \theta$ (implicit differentiation)
- Bilevel optimization: `bilevel_solve` backpropagates through the solver to learn parameters of constrained problems

## Documentation

See the [full documentation](https://samueltalkington.com/research/marguerite/) for tutorials, examples, and API reference.

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
