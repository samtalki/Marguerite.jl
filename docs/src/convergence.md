<!-- Copyright 2026 Samuel Talkington and contributors
   SPDX-License-Identifier: Apache-2.0 -->

# Convergence

## ``O(1/t)`` convergence rate

For smooth convex objectives over compact sets, Frank-Wolfe achieves:

```math
f(x_t) - f(x^*) \leq \frac{2 L D^2}{t + 2}
```

where ``L`` is the smoothness constant and ``D`` is the diameter of ``\mathcal{C}``.
The step size ``\gamma_t = 2/(t+2)`` from [`MonotonicStepSize`](@ref) achieves this rate.

## The Frank-Wolfe gap

The **duality gap** ``g_t = \langle \nabla f(x_t),\, x_t - v_t \rangle`` is a computable
convergence certificate: it upper-bounds the primal gap ``f(x_t) - f^*`` without
knowing ``f^*``. This is what `solve` uses for its convergence check.

## Sparsity

Frank-Wolfe iterates are convex combinations of vertices. After ``t`` iterations,
``x_t`` has at most ``t + 1`` nonzero entries. This gives **vertex-sparse** solutions
naturally -- useful for feature selection and portfolio problems.

## Example

Quadratic minimization on the probability simplex ``\Delta_{20}``:

```@example convergence
using Marguerite, LinearAlgebra, Random, UnicodePlots
using Marguerite: MonotonicStepSize
Random.seed!(42)

n = 20
A = randn(n, n)
Q = A'A + 0.1I
c = randn(n)

f(x) = 0.5 * dot(x, Q * x) + dot(c, x)
∇f!(g, x) = (g .= Q * x .+ c)

lmo = ProbabilitySimplex()
# Start from a vertex for clean sparsity tracking
x0 = zeros(n); x0[1] = 1.0

# Solve to high accuracy for reference
x_opt, _ = solve(f, ∇f!, lmo, x0; max_iters=50000, tol=1e-12, monotonic=false)
f_opt = f(x_opt)

# Hand-written FW loop to collect history
x = copy(x0)
g = zeros(n); v = zeros(n)
step = MonotonicStepSize()
max_iters = 2000

primal_gaps = Float64[]
fw_gaps = Float64[]
sparsities = Int[]

for t in 0:(max_iters - 1)
    ∇f!(g, x)
    lmo(v, g)
    gap = dot(g, x .- v)
    push!(primal_gaps, f(x) - f_opt)
    push!(fw_gaps, gap)
    push!(sparsities, count(xi -> abs(xi) > 1e-12, x))
    γ = step(t)
    x .= x .+ γ .* (v .- x)
end

nothing  # hide
```

### Primal gap (log-log)

The gap decays as ``O(1/t)``:

```@example convergence
lineplot(1:max_iters, primal_gaps;
         xscale=:log10, yscale=:log10,
         title="Primal Gap f(xₜ) - f*",
         xlabel="iteration", ylabel="gap",
         name="primal gap", width=60)
```

### Frank-Wolfe duality gap

```@example convergence
lineplot(1:max_iters, fw_gaps;
         yscale=:log10,
         title="Frank-Wolfe Duality Gap",
         xlabel="iteration", ylabel="⟨∇f, x-v⟩",
         name="FW gap", width=60)
```

### Iterate sparsity

The number of nonzeros grows slowly -- Frank-Wolfe produces vertex-sparse iterates:

```@example convergence
lineplot(1:max_iters, sparsities;
         title="Iterate Sparsity (nnz)",
         xlabel="iteration", ylabel="nnz(xₜ)",
         name="nnz", width=60)
```

## Monotonic mode

By default, `solve` uses `monotonic=true`, which rejects updates that increase
the objective. This prevents oscillation but can slow convergence slightly.
Set `monotonic=false` for clean ``O(1/t)`` curves as shown above.
