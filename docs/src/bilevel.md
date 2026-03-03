<!-- Copyright 2026 Samuel Talkington and contributors
   SPDX-License-Identifier: Apache-2.0 -->

# Bilevel Optimization

## Why bilevel optimization?

Many machine learning and engineering problems have **nested structure**: an outer
problem (learning, design) wraps an inner constrained optimization:

```math
\min_\theta \; L\bigl(x^*(\theta)\bigr), \quad \text{where} \quad
x^*(\theta) = \arg\min_{x \in \mathcal{C}} f(x;\, \theta)
```

Examples include hyperparameter tuning, meta-learning, adversarial training,
and inverse optimization. Projects like Meta's Theseus and `cvxpylayers` have
shown that **differentiable optimization layers** are a powerful primitive.

Marguerite brings this to **constrained** problems via Frank-Wolfe's
projection-free approach: any set with a linear minimization oracle works,
no projection operator needed.

## How it works

Marguerite uses **implicit differentiation** -- it does not unroll the solver
iterations. At convergence, the optimality condition
``\nabla_x f(x^*;\, \theta) \approx 0`` on the optimal face gives (via the implicit function theorem):

```math
\bar{\theta} = -\left(\frac{\partial \nabla_x f}{\partial \theta}\right)^\top u,
\quad [\nabla^2_{xx} f + \lambda I]\, u = \bar{x}
```

The Hessian system is solved by conjugate gradient with Hessian-vector products.
This gives:
- **``O(1)`` memory** -- no computational graph stored
- **Exact gradients at convergence** -- not a truncated unrolling approximation
- **Backend-agnostic** -- works with any DifferentiationInterface backend

## High-level API

Marguerite provides `bilevel_solve` and `bilevel_gradient` for one-call bilevel
optimization. These handle the forward solve, outer loss gradient, and implicit
pullback internally:

```@example bilevel
using Marguerite, LinearAlgebra, Random
Random.seed!(123)

n = 5
A = randn(n, n)
H = A'A + 0.5I

f(x, θ) = 0.5 * dot(x, H * x) - dot(θ, x)
∇f!(g, x, θ) = (g .= H * x .- θ)

lmo = ProbabilitySimplex()
x0 = fill(1.0 / n, n)

x_target = zeros(n)
x_target[1] = 0.6; x_target[2] = 0.3; x_target[3] = 0.1

outer_loss(x) = sum((x .- x_target).^2)
θ = H * x_target
η = 0.01

losses = Float64[]
for k in 1:80
    x_star, θ_grad, _ = bilevel_solve(outer_loss, f, ∇f!, lmo, x0, θ;
                                       max_iters=10000, tol=1e-4)
    push!(losses, outer_loss(x_star))
    θ .= θ .- η .* θ_grad
end

x_final, _ = solve(f, ∇f!, lmo, x0, θ; max_iters=10000, tol=1e-4)
println("Final loss: ", round(losses[end]; sigdigits=3))
println("x*(θ):     ", round.(x_final; digits=3))
println("x_target:  ", x_target)
```

For just the gradient (without the solution), use `bilevel_gradient`:

```julia
θ_grad = bilevel_gradient(outer_loss, f, ∇f!, lmo, x0, θ; max_iters=10000, tol=1e-4)
```

Both functions accept `diff_cg_maxiter`, `diff_cg_tol`, and `diff_λ` to tune
the CG solver used in the implicit differentiation backward pass.
See [Implicit Differentiation](@ref) for details.

## Advanced: manual rrule

For full control, call the `rrule` directly. This is useful when your outer loss
depends on `θ` directly (not just through `x*(θ)`), or when you need access to
the inner `Result` diagnostics.

**Inner problem**: minimize a parameterized quadratic on the probability simplex.

```math
x^*(\theta) = \arg\min_{x \in \Delta_n} \;\tfrac{1}{2} x^\top H x - \theta^\top x
```

**Outer problem**: find ``\theta`` such that ``x^*(\theta)`` matches a target.

```math
\min_\theta \; \|x^*(\theta) - x_{\text{target}}\|^2
```

```@example bilevel
using ChainRulesCore: rrule

solve_kw = (; max_iters=10000, tol=1e-4)

nothing  # hide
```

The key pattern: call `rrule` directly to get the pullback, then compute
the outer loss gradient:

```@example bilevel
function bilevel_step(θ)
    (x_star, result), pb = rrule(solve, f, ∇f!, lmo, x0, θ; solve_kw...)
    loss = sum((x_star .- x_target).^2)
    x̄ = 2.0 .* (x_star .- x_target)
    tangents = pb((x̄, nothing))
    return x_star, loss, tangents[end]  # (x*, L, ∂L/∂θ)
end

nothing  # hide
```

Run gradient descent on the outer problem:

```@example bilevel
θ = H * x_target  # warm start
η = 0.01

losses = Float64[]
for k in 1:80
    x_star, loss, θ̄ = bilevel_step(θ)
    push!(losses, loss)
    θ .= θ .- η .* θ̄
end

x_final, _ = solve(f, ∇f!, lmo, x0, θ; solve_kw...)
println("Final loss: ", round(losses[end]; sigdigits=3))
println("x*(θ):     ", round.(x_final; digits=3))
println("x_target:  ", x_target)
```

```@example bilevel
using UnicodePlots
lineplot(1:80, log10.(losses);
         title="Outer Loss (log₁₀)",
         xlabel="outer iteration", ylabel="log₁₀(loss)",
         name="loss", width=60)
```

## Why Frank-Wolfe for bilevel?

Frank-Wolfe is uniquely suited to bilevel optimization with complex constraints:

1. **Projection-free**: Only needs a linear minimization oracle, not a projection.
   Many constraint sets (matroid polytopes, flow polytopes, nuclear norm balls)
   have cheap LMOs but expensive projections.
2. **Sparse iterates**: Solutions are convex combinations of vertices, giving
   interpretable sparse structure.
3. **Theoretical guarantees**: Palmieri et al. (2026) establish
   ``O(\tau^{-2} \log \tau^{-1})`` complexity for Frank-Wolfe in bilevel settings.

## References

- A. Palmieri, F. Rinaldi, S. Salzo & S. Venturini, "Iteration Complexity of Frank-Wolfe and Its Variants for Bilevel Optimization," 2026.
- E. Grazzi, L. Franceschi, M. Pontil & S. Salzo, "On the Iteration Complexity of Hypergradient Computation," ICML 2020.
- L. Franceschi, P. Frasconi, S. Salzo, R. Grazzi & M. Pontil, "Bilevel Programming for Hyperparameter Optimization and Meta-Learning," ICML 2018.
- A. Agrawal, B. Amos, S. Barratt, S. Boyd, S. Diamond & Z. Kolter, "Differentiable Convex Optimization Layers," NeurIPS 2019.
