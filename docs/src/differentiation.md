<!-- Copyright 2026 Samuel Talkington and contributors
   SPDX-License-Identifier: Apache-2.0 -->

# Implicit Differentiation

## Theory

At convergence, ``x^*(\theta)`` satisfies the first-order optimality condition

```math
\nabla_x f(x^*; \theta) = 0
```

on the optimal face. By the implicit function theorem:

```math
\frac{\partial x^*}{\partial \theta} = -[\nabla^2_{xx} f]^{-1} \nabla^2_{x\theta} f
```

For the **pullback** (reverse-mode), given cotangent ``\bar{x}``:

```math
\bar{\theta} = -\left(\frac{\partial \nabla_x f}{\partial \theta}\right)^\top u, \quad \text{where } \nabla^2_{xx} f \cdot u = \bar{x}
```

## Implementation

The CG linear solve and cross-derivative computation use DifferentiationInterface:

| Operation | Map | Best mode | DI function |
|-----------|-----|-----------|-------------|
| Gradient of ``f`` (auto) | ``\mathbb{R}^n \to \mathbb{R}`` | Forward | `DI.gradient!` |
| HVP in CG | JVP of ``\mathbb{R}^n \to \mathbb{R}^n`` | Forward | `DI.hvp` |
| Cross-derivative | gradient of ``\theta \mapsto \langle \nabla_x f, u \rangle`` | Forward or Reverse | `DI.gradient` |

The Hessian system is solved by conjugate gradient (CG) with Hessian-vector products,
avoiding explicit Hessian construction. Tikhonov regularization ``(\nabla^2_{xx} f + \lambda I)``
ensures well-conditioned systems near singular Hessians.

## Usage

```julia
using Marguerite, LinearAlgebra

f(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
∇f!(g, x, θ) = (g .= x .- θ)

θ = [0.8, 0.2]
x, result = solve(f, ∇f!, ProbSimplex(), [0.5, 0.5], θ;
                   max_iters=10000, tol=1e-4)
```

The `ChainRulesCore.rrule` is defined on the 5-argument `solve` signatures
(those accepting `θ`). The rrule pullback computes Hessian-vector products
internally using `SECOND_ORDER_BACKEND` (forward-over-forward via
`AutoForwardDiff`).

## AD backend selection

Marguerite uses ForwardDiff as the default backend. All AD goes through
DifferentiationInterface, so you can override with any DI-compatible backend:

```julia
import DifferentiationInterface as DI

x, result = solve(f, ProbSimplex(), [0.5, 0.5];
                   backend=DI.AutoForwardDiff())
```

## Tuning the CG solver

The implicit differentiation backward pass solves a linear system
``(\nabla^2_{xx} f + \lambda I) u = \bar{x}`` via conjugate gradient.
Three keyword arguments control this solve:

| Keyword | Default | Description |
|---------|---------|-------------|
| `diff_cg_maxiter` | `50` | Maximum CG iterations |
| `diff_cg_tol` | `1e-6` | CG convergence tolerance on residual norm |
| `diff_λ` | `1e-4` | Tikhonov regularization strength |

These can be passed to `solve` (θ-accepting variants), `bilevel_solve`,
`bilevel_gradient`, or directly to `rrule`:

```julia
x, result = solve(f, ∇f!, lmo, x0, θ;
                   diff_cg_maxiter=100, diff_cg_tol=1e-8, diff_λ=1e-3)
```

If the CG solver does not converge within `diff_cg_maxiter` iterations, a
warning is emitted (at most 3 times per session). Increase `diff_cg_maxiter`
or relax `diff_cg_tol` if you see this warning.

For bilevel optimization using the rrule directly, see [Bilevel Optimization](@ref).

## rrule

```@docs
ChainRulesCore.rrule(::typeof(solve), ::Any, ::Any, ::Any, ::Any, ::Any)
```
