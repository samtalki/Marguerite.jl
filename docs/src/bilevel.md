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

## Worked example

**Inner problem**: minimize a parameterized quadratic on the probability simplex.

```math
x^*(\theta) = \arg\min_{x \in \Delta_n} \;\tfrac{1}{2} x^\top H x - \theta^\top x
```

**Outer problem**: find ``\theta`` such that ``x^*(\theta)`` matches a target.

```math
\min_\theta \; \|x^*(\theta) - x_{\text{target}}\|^2
```

```@example bilevel
using Marguerite, LinearAlgebra, Random
using ChainRulesCore: rrule
import DifferentiationInterface as DI
import ForwardDiff
Random.seed!(123)

n = 5
A = randn(n, n)
H = A'A + 0.5I  # random PD matrix

f(x, θ) = 0.5 * dot(x, H * x) - dot(θ, x)
∇f!(g, x, θ) = (g .= H * x .- θ)

lmo = ProbabilitySimplex()
x0 = fill(1.0 / n, n)
# ForwardDiff is recommended: the backward pass needs forward-mode HVPs
backend = DI.AutoForwardDiff()
solve_kw = (; max_iters=10000, tol=1e-6, backend=backend)

# Target solution on the simplex
x_target = zeros(n)
x_target[1] = 0.6; x_target[2] = 0.3; x_target[3] = 0.1

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
η = 0.1

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
3. **Theoretical guarantees**: Palmieri et al. (2024) establish
   ``O(\tau^{-2} \log \tau^{-1})`` complexity for Frank-Wolfe in bilevel settings.

## References

- A. Palmieri, M. Rinaldi & F. Salzo, "On the Use of the Frank-Wolfe Algorithm for Bilevel Optimization," 2024.
- E. Grazzi, L. Franceschi, M. Pontil & S. Salzo, "On the Iteration Complexity of Hypergradient Computation," ICML 2020.
- L. Franceschi, P. Frasconi, S. Salzo, R. Grazzi & M. Pontil, "Bilevel Programming for Hyperparameter Optimization and Meta-Learning," ICML 2018.
- A. Agrawal, B. Amos, S. Barratt, S. Boyd, S. Diamond & Z. Kolter, "Differentiable Convex Optimization Layers," NeurIPS 2019.
