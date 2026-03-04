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

## KKT Adjoint for Constrained Solutions

When the solution ``x^*`` lies on the boundary of ``\mathcal{C}`` (i.e., some
constraints are active), the unconstrained optimality condition
``\nabla_x f = 0`` no longer holds. Marguerite automatically detects active
constraints via [`active_set`](@ref) and solves the full KKT adjoint system:

```math
\begin{bmatrix} \nabla^2_{xx} f & G^\top \\ G & 0 \end{bmatrix}
\begin{bmatrix} u \\ \mu \end{bmatrix} =
\begin{bmatrix} \bar{x} \\ 0 \end{bmatrix}
```

where ``G`` is the matrix of active constraint normals. This is solved via a
reduced-space approach:

1. Partition variables into **bound** (pinned to constraint boundaries) and **free**
2. Project ``\bar{x}_{\text{free}}`` onto the null space of equality constraint normals
3. CG solve in the reduced space: ``(P H_{\text{free}} P + \lambda I) w = P \bar{x}_{\text{free}}``
4. Recover multipliers ``\mu`` from the KKT residual

For interior solutions (no active constraints), this reduces to the unconstrained
Hessian solve described above.

## Parametric Constraints

When using a [`ParametricOracle`](@ref), the constraint set itself depends on
``\theta``. The total gradient has two components:

```math
\bar{\theta} = \bar{\theta}_{\text{obj}} + \bar{\theta}_{\text{constraint}}
```

The objective contribution ``\bar{\theta}_{\text{obj}}`` comes from the KKT adjoint solve
above. The constraint contribution ``\bar{\theta}_{\text{constraint}} = \nabla_\theta \Phi(\theta)``
is computed via AD through the scalar function
``\Phi(\theta) = \mu^\top h(\theta)``, where ``h(\theta)`` are the active
constraint RHS values. For constraints with ``\theta``-dependent normals
(e.g. `ParametricWeightedSimplex`), the scalar also captures normal-variation
sensitivity.

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
warning is emitted (with limited frequency). Increase `diff_cg_maxiter`
or relax `diff_cg_tol` if you see this warning.

## Bilevel optimization via rrule

For bilevel problems, call the `rrule` directly to get the pullback.
The default backends handle everything automatically — ForwardDiff
for gradients and `SECOND_ORDER_BACKEND` (forward-over-forward) for HVPs:

```julia
using ChainRulesCore: rrule

(x_star, result), pb = rrule(solve, f, ∇f!, lmo, x0, θ;
                              max_iters=5000)
```

The pullback accepts a tuple `(x̄, result_tangent)` where `x̄` is the cotangent
of the solution and `result_tangent` is typically `nothing`:

```julia
tangents = pb((x̄, nothing))
# tangents = (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), θ̄)
#             solve      f          ∇f!        lmo        x0         θ
```

Only `θ̄` (the last element) is nonzero. The other entries are `NoTangent()`
since `f`, `∇f!`, `lmo`, and `x0` are not differentiated.

The auto-gradient variant `rrule(solve, f, lmo, x0, θ; ...)` returns one fewer
`NoTangent` (no `∇f!` argument).

See [Bilevel Optimization](@ref) for a complete worked example with gradient
descent on the outer problem.

## Parametric oracle usage

When the constraint set depends on ``\theta``, use a [`ParametricOracle`](@ref):

```julia
using Marguerite, LinearAlgebra

f(x, θ) = 0.5 * dot(x, x) - dot(θ[1:2], x)
∇f!(g, x, θ) = (g .= x .- θ[1:2])

plmo = ParametricBox(θ -> fill(θ[3], 2), θ -> fill(θ[4], 2))
θ = [0.8, 0.2, 0.0, 1.0]
x, result = solve(f, ∇f!, plmo, [0.5, 0.5], θ; max_iters=5000, tol=1e-6)
```

The `rrule` for this signature computes ``\bar{\theta}`` through both the
objective and constraint parameters via KKT adjoint differentiation.

## rrule

```@docs
ChainRulesCore.rrule(::typeof(solve), ::Any, ::Any, ::Any, ::Any, ::Any)
```
