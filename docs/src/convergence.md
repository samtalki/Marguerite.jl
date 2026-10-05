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

lmo = ProbSimplex()
# Start from a vertex for clean sparsity tracking
x0 = zeros(n); x0[1] = 1.0

# Solve to high accuracy for reference
x_opt, _ = solve(f, lmo, x0; grad=∇f!, max_iters=50000, tol=1e-12, monotonic=false)
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
scatterplot(1:max_iters, primal_gaps;
         xscale=:log10, yscale=:log10,
         title="Primal Gap f(xₜ) - f*",
         xlabel="iteration", ylabel="gap",
         name="primal gap", width=60)
```

### Frank-Wolfe duality gap

```@example convergence
scatterplot(1:max_iters, fw_gaps;
         yscale=:log10,
         title="Frank-Wolfe Duality Gap",
         xlabel="iteration", ylabel="⟨∇f, x-v⟩",
         name="FW gap", width=60)
```

### Iterate sparsity

The number of nonzeros grows slowly -- Frank-Wolfe produces vertex-sparse iterates:

```@example convergence
scatterplot(1:max_iters, sparsities;
         title="Iterate Sparsity (nnz)",
         xlabel="iteration", ylabel="nnz(xₜ)",
         name="nnz", width=60)
```

## Monotonic mode

By default, `solve` uses `monotonic=true`, which rejects updates that increase
the objective. This prevents oscillation but can slow convergence slightly.
Set `monotonic=false` for clean ``O(1/t)`` curves as shown above.

The monotonic filter rejects a trial point when
``f(x_{\text{trial}}) > f(x) + \varepsilon \cdot \max(1, |f(x)|)``
where ``\varepsilon = \mathrm{eps}(T)`` is machine epsilon at the working precision.
This threshold scales with the objective magnitude to avoid spurious discards due
to floating-point noise.

## Traces, lower bounds and stopping

`solve` accepts a `callback(state)` that runs once after every iteration, so a
trace no longer needs a hand-written loop. `state` carries the iteration count
`t`, the iterate `x`, its objective `obj` and Frank-Wolfe gap `gap`, the step
`γ`, whether the step was `accepted`, the `elapsed` seconds and `lower_bound`,
the running maximum of ``f(x_t) - g_t``. For convex ``f`` that maximum is a
lower bound on ``\min_{x \in C} f(x)``, so `obj - lower_bound` certifies the
primal gap at every iteration. Returning `true` from the callback stops the
solve. `time_limit` caps the wall-clock time and `rel_tol` adds the purely
relative stop ``g_t \le \mathrm{rel\_tol} \cdot |f(x_t)|`` (set `tol=0` to use
it alone). `Result` records the final `lower_bound` and `elapsed`, and its gap
always belongs to the returned iterate.

The certified gap `obj - lower_bound` is computable without ``f^*`` and always
bounds the true primal gap from above:

```@example convergence
trace = Tuple{Int, Float64, Float64}[]   # (t, objective, certified primal gap)
x_cb, res_cb = solve(f, lmo, x0; grad=∇f!, max_iters=2000,
                     callback = s -> (push!(trace, (s.t, s.obj, s.obj - s.lower_bound)); false))
(certified = res_cb.objective - res_cb.lower_bound, true_gap = res_cb.objective - f_opt)
```

## Step size rules

Besides the open-loop [`MonotonicStepSize`](@ref) and the backtracking
[`AdaptiveStepSize`](@ref), two rules use the direction ``d = v - x``.
[`ShortStep`](@ref)`(L)` takes the minimizer over ``[0, 1]`` of the quadratic
upper bound with a fixed smoothness constant,
``\gamma_t = \min\{1, g_t / (L \|d\|^2)\}``; it never evaluates ``f`` to choose
``\gamma`` and, when ``L`` bounds the curvature, every step decreases ``f``.
[`SecantLineSearch`](@ref) searches along ``d`` with at most `max_trials`
points per iteration: it first doubles the previous step and reads the slope
``\langle \nabla f(x + \gamma d), d \rangle`` there, accepts that point if the
slope is not positive, and otherwise takes the secant step to the zero of the
slope, which is exact on quadratics, followed by halvings while ``f`` rises.
The gradient computed at its first trial is reused as the next gradient when
that trial is accepted. Both rules run on the CPU.

## Pairwise steps

On polytopes, plain Frank-Wolfe steps toward one vertex at a time and zig-zags
when the solution lies on a face. `variant=:pairwise` instead moves along
``v^+ - v^-``, where ``v^+`` is the Frank-Wolfe vertex and ``v^-`` is the
*away vertex*: the vertex of the smallest face containing ``x`` that maximizes
``\langle \nabla f(x), v \rangle``, returned by [`away_vertex!`](@ref). This
removes weight from the worst vertex of the current face without storing an
active set. The step from the chosen rule is capped at the largest feasible
step [`pairwise_max_step`](@ref); steps that reach it move ``x`` to a smaller
face and are counted in `Result.drop_steps`. The Frank-Wolfe gap still drives
the stopping test. [`MaskedKnapsack`](@ref) implements both oracle methods;
other oracles can opt in by defining them.

## References

- M. Frank & P. Wolfe, ["An algorithm for quadratic programming,"](https://doi.org/10.1002/nav.3800030109) *Naval Research Logistics*, 1956.
- M. Jaggi, ["Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization,"](https://proceedings.mlr.press/v28/jaggi13.html) *ICML*, 2013.
