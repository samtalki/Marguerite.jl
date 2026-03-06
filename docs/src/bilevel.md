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

Marguerite supports constrained bilevel problems via Frank-Wolfe's
projection-free approach: any set with a linear minimization oracle works,
no projection operator needed.

## How it works

Marguerite uses **implicit differentiation** -- it does not unroll the solver
iterations. At convergence, the optimality condition
``\nabla_x f(x^*;\, \theta) \approx 0`` on the optimal face gives (via the implicit function theorem):

```math
d\theta = -\left(\frac{\partial \nabla_x f}{\partial \theta}\right)^\top u,
\quad [\nabla^2_{xx} f + \lambda I]\, u = dx
```

The Hessian system is solved by conjugate gradient with Hessian-vector products.
This gives:
- **No graph storage** -- implicit differentiation needs only the converged solution, not the full iteration history
- **Exact gradients at convergence** -- not a truncated unrolling approximation
- **Backend-agnostic** -- works with any DifferentiationInterface backend

!!! tip "Step size control in the outer loop"
    The inner solver's `monotonic=true` flag ensures monotonic descent within the
    **inner** Frank-Wolfe loop only. The **outer** bilevel gradient descent loop
    needs its own step size control — a fixed learning rate can overshoot and
    increase the outer loss. The examples below use Armijo backtracking line search
    (halving η until the outer loss decreases) to guarantee monotonic convergence.
    In practice, any standard optimizer (Adam, L-BFGS, etc.) can be used instead.

## High-level API

Marguerite provides `bilevel_solve` and `bilevel_gradient` for one-call bilevel
optimization. These handle the forward solve, outer loss gradient, and implicit
pullback internally:

```@example bilevel
using Marguerite, LinearAlgebra

n = 3
H = Diagonal([2.0, 1.5, 1.0])

f(x, θ) = 0.5 * dot(x, H * x) - dot(θ, x)
∇f!(g, x, θ) = (g .= H * x .- θ)

lmo = ProbSimplex()
x0 = fill(1.0 / n, n)

x_target = zeros(n)
x_target[1] = 0.6; x_target[2] = 0.3; x_target[3] = 0.1

outer_loss(x) = sum((x .- x_target).^2)
θ = H * x_target
η = 0.1

losses = Float64[]
x_curr = copy(x0)
for k in 1:50
    x_star, θ_grad, _ = bilevel_solve(outer_loss, f, lmo, x_curr, θ;
                                       grad=∇f!)
    x_curr .= x_star
    loss_k = outer_loss(x_star)
    push!(losses, loss_k)

    # Armijo backtracking on the outer step
    η_k = η
    for _ in 1:10
        θ_cand = θ .- η_k .* θ_grad
        x_cand, _ = solve(f, lmo, x_curr, θ_cand; grad=∇f!)
        outer_loss(x_cand) ≤ loss_k && break
        η_k *= 0.5
    end
    θ .= θ .- η_k .* θ_grad
end

x_final, _ = solve(f, lmo, x_curr, θ; grad=∇f!)
println("Final loss: ", round(losses[end]; sigdigits=3))
println("x*(θ):     ", round.(x_final; digits=3))
println("x_target:  ", x_target)
```

```@example bilevel
using UnicodePlots
lineplot(1:50, log10.(losses);
         title="Outer Loss (log₁₀)",
         xlabel="outer iteration", ylabel="log₁₀(loss)",
         name="loss", width=60)
```

For just the gradient (without the solution), use `bilevel_gradient`:

```julia
θ_grad = bilevel_gradient(outer_loss, f, lmo, x0, θ;
                          grad=∇f!)
```

Both functions accept `diff_cg_maxiter`, `diff_cg_tol`, and `diff_lambda` to tune
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

solve_kw = (;)

nothing  # hide
```

The key pattern: call `rrule` directly to get the pullback, then compute
the outer loss gradient:

```@example bilevel
function bilevel_step(x_curr, θ)
    (x_star, result), pb = rrule(solve, f, lmo, x_curr, θ; grad=∇f!, solve_kw...)
    loss = sum((x_star .- x_target).^2)
    dx = 2.0 .* (x_star .- x_target)
    tangents = pb((dx, nothing))
    return x_star, loss, tangents[end]  # (x*, L, ∂L/∂θ)
end

nothing  # hide
```

Run gradient descent on the outer problem:

```@example bilevel
θ = H * x_target  # warm start
η = 0.1

losses = Float64[]
x_curr = copy(x0)
for k in 1:50
    x_star, loss, dθ = bilevel_step(x_curr, θ)
    x_curr .= x_star
    push!(losses, loss)

    # Armijo backtracking on the outer step
    η_k = η
    for _ in 1:10
        θ_cand = θ .- η_k .* dθ
        x_cand, _ = solve(f, lmo, x_curr, θ_cand; grad=∇f!, solve_kw...)
        outer_loss(x_cand) ≤ loss && break
        η_k *= 0.5
    end
    θ .= θ .- η_k .* dθ
end

x_final, _ = solve(f, lmo, x_curr, θ; grad=∇f!, solve_kw...)
println("Final loss: ", round(losses[end]; sigdigits=3))
println("x*(θ):     ", round.(x_final; digits=3))
println("x_target:  ", x_target)
```

```@example bilevel
using UnicodePlots
lineplot(1:50, log10.(losses);
         title="Outer Loss (log₁₀)",
         xlabel="outer iteration", ylabel="log₁₀(loss)",
         name="loss", width=60)
```

## Parametric constraint sets

When the constraint set itself depends on ``\theta``, use a
[`ParametricOracle`](@ref) instead of a plain oracle. This enables
differentiation through both the objective and the constraints:

```math
\min_\theta \; L(x^*(\theta)), \quad
x^*(\theta) = \arg\min_{x \in C(\theta)} f(x, \theta)
```

```julia
using Marguerite, LinearAlgebra

n = 3
outer_loss(x) = sum((x .- [0.3, 0.5, 0.2]).^2)
x0 = fill(1.0 / n, n)
theta = [0.3, 0.5, 0.2, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

f(x, θ) = 0.5 * dot(x, x) - dot(θ[1:n], x)
∇f!(g, x, θ) = (g .= x .- θ[1:n])

# Box constraints with θ-dependent bounds
plmo = ParametricBox(θ -> fill(θ[n+1], n), θ -> fill(θ[n+2], n))

x_star, θ_grad, cg_result = bilevel_solve(outer_loss, f, plmo, x0, theta;
                                           grad=∇f!)
# x_star ≈ [0.3, 0.5, 0.2], θ_grad has 9 components (objective + box bounds)
```

!!! note
    This snippet shows the API pattern only. See [High-level API](@ref) above
    for a complete bilevel loop with convergence output.

The gradient ``d\theta`` accounts for both how ``\theta`` affects the
objective and how it shifts the constraint boundaries, computed via KKT
adjoint differentiation. See [Implicit Differentiation](@ref) for the
mathematical details.

## Why Frank-Wolfe for bilevel?

Frank-Wolfe has properties that suit bilevel optimization with complex constraints:

1. **Projection-free**: Only needs a linear minimization oracle, not a projection.
   Many constraint sets (matroid polytopes, flow polytopes, nuclear norm balls)
   have cheap LMOs but expensive projections.
2. **Sparse iterates**: Solutions are convex combinations of vertices, giving
   interpretable sparse structure.
3. **Theoretical guarantees**: Palmieri et al. (2026) establish iteration complexity
   bounds for Frank-Wolfe in bilevel settings.

## References

- A. Palmieri, F. Rinaldi, S. Salzo & S. Venturini, ["Iteration Complexity of Frank-Wolfe and Its Variants for Bilevel Optimization,"](https://arxiv.org/abs/2602.23076) 2026.
- R. Grazzi, L. Franceschi, M. Pontil & S. Salzo, ["On the Iteration Complexity of Hypergradient Computation,"](https://arxiv.org/abs/2006.16218) *ICML*, 2020.
- L. Franceschi, P. Frasconi, S. Salzo, R. Grazzi & M. Pontil, ["Bilevel Programming for Hyperparameter Optimization and Meta-Learning,"](https://arxiv.org/abs/1806.04910) *ICML*, 2018.
- A. Agrawal, B. Amos, S. Barratt, S. Boyd, S. Diamond & Z. Kolter, ["Differentiable Convex Optimization Layers,"](https://arxiv.org/abs/1910.12430) *NeurIPS*, 2019.
