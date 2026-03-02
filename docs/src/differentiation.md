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
| Gradient of ``f`` (auto) | ``\mathbb{R}^n \to \mathbb{R}`` | Reverse | `DI.gradient!` |
| HVP in CG | JVP of ``\mathbb{R}^n \to \mathbb{R}^n`` | Forward | `DI.hvp` |
| Cross-derivative | VJP of ``\mathbb{R}^p \to \mathbb{R}^n`` | Reverse | `DI.pullback` |

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
(those accepting `θ`). Any AD framework that supports ChainRules (Mooncake, Zygote,
Diffractor) can differentiate through the solve.

## AD backend selection

Marguerite uses Mooncake as the default backend. All AD goes through
DifferentiationInterface, so you can override with any DI-compatible backend:

- **Mooncake** (default): Best for reverse-mode, used automatically
- **ForwardDiff** (`DI.AutoForwardDiff()`): Best for forward-mode, small dimensions

```julia
import DifferentiationInterface as DI
import ForwardDiff

x, result = solve(f, ProbSimplex(), [0.5, 0.5];
                   backend=DI.AutoForwardDiff())
```

## Bilevel optimization via rrule

For bilevel problems, call the `rrule` directly to get the pullback.
Use `ForwardDiff` as the backend -- it handles the Hessian-vector products
needed by the implicit differentiation pullback:

```julia
using ChainRulesCore: rrule
import DifferentiationInterface as DI
import ForwardDiff

backend = DI.AutoForwardDiff()
(x_star, result), pb = rrule(solve, f, ∇f!, lmo, x0, θ;
                              max_iters=5000, backend=backend)
```

The pullback accepts a tuple `(x̄, result_tangent)` where `x̄` is the cotangent
of the solution and `result_tangent` is typically `nothing`:

```julia
tangents = pb((x̄, nothing))
# tangents = (NoTangent, NoTangent, NoTangent, NoTangent, NoTangent, θ̄)
#             solve      f          ∇f!        lmo        x0         θ
```

Only `θ̄` (the last element) is nonzero. The other entries are `NoTangent()`
since `f`, `∇f!`, `lmo`, and `x0` are not differentiated.

The auto-gradient variant `rrule(solve, f, lmo, x0, θ; ...)` returns one fewer
`NoTangent` (no `∇f!` argument).

See [Bilevel Optimization](@ref) for a complete worked example with gradient
descent on the outer problem.
