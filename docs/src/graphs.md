# Graph Problems

## Laplacian Pseudoinverse Quadratic Form

Consider a weighted graph ``G = (V, E)`` with ``n`` nodes and ``m`` edges.
Given edge switching variables ``s \in [0,1]^m`` and edge weights ``w``, the
**weighted graph Laplacian** is

```math
L(s) = A^\top \operatorname{diag}(w \odot s)\, A
```

where ``A \in \mathbb{R}^{m \times n}`` is the oriented incidence matrix.
For a demand vector ``d \perp \mathbf{1}``, the **effective resistance quadratic form**

```math
d^\top L^\dagger(s)\, d
```

measures the congestion cost of routing demand ``d`` over the active edges.
We minimize this over a masked knapsack:

```math
\min_{s \in C}\; d^\top L^\dagger(s)\, d, \qquad
C = \bigl\{s \in [0,1]^m : \textstyle\sum_i s_i \le q,\; s_e = 1\;\forall e \in T\bigr\}
```

where ``T`` is a random spanning tree backbone (ensuring connectivity) and
``q = \lceil 1.3\, n \rceil`` gives 30% headroom above the tree minimum.

## Helpers

We pre-allocate a dense grounded Laplacian and update it in-place each iteration
with `fill!` + O(1) index writes. Both the objective and gradient share a `GraphCache`:

```@example graphs
using Marguerite, LinearAlgebra, SparseArrays, Random, Statistics
using JuMP, Clarabel
using UnicodePlots

struct GraphCache{T}
    w::Vector{T}
    d::Vector{T}
    inc::Vector{Tuple{Int,Int}}
    n::Int
    L_red::Matrix{T}    # grounded Laplacian (dense, pre-allocated)
    x::Vector{T}        # voltage workspace
    d_red::Vector{T}    # d[1:n-1]
end

function GraphCache(w::Vector{T}, d::Vector{T}, inc, n) where T
    GraphCache{T}(w, d, inc, n,
        zeros(T, n-1, n-1), zeros(T, n), d[1:n-1])
end

function laplacian_solve!(cache::GraphCache{T}, s) where T
    n = cache.n; nr = n - 1
    L = cache.L_red
    fill!(L, zero(T))
    # node n is grounded; skip its entries
    @inbounds for (e, (i, j)) in enumerate(cache.inc)
        v = cache.w[e] * s[e]
        if i ≤ nr;  L[i,i] += v;  end
        if j ≤ nr;  L[j,j] += v;  end
        if i ≤ nr && j ≤ nr
            L[i,j] -= v;  L[j,i] -= v
        end
    end
    F = cholesky!(Hermitian(L))
    ldiv!(view(cache.x, 1:nr), F, cache.d_red)
    cache.x[n] = zero(T)
    return dot(cache.d, cache.x)
end

struct GraphObj{T}
    cache::GraphCache{T}
end
(f::GraphObj)(s) = laplacian_solve!(f.cache, s)

struct GraphGrad!{T}
    cache::GraphCache{T}
end

function (∇f::GraphGrad!)(g, s)
    laplacian_solve!(∇f.cache, s)
    @inbounds for e in eachindex(∇f.cache.inc)
        i, j = ∇f.cache.inc[e]
        Δ = ∇f.cache.x[i] - ∇f.cache.x[j]
        g[e] = -∇f.cache.w[e] * Δ * Δ
    end
    return g
end

nothing  # hide
```

## Random graph generation

We generate random connected graphs by first building a random spanning tree
(guaranteeing connectivity), then adding extra edges.

```@example graphs
function random_graph(n, m; rng=Random.default_rng())
    inc = Tuple{Int,Int}[]
    # random spanning tree via random permutation parents
    perm = randperm(rng, n)
    for i in 2:n
        push!(inc, (perm[i], perm[rand(rng, 1:i-1)]))
    end
    backbone_idx = collect(1:n-1)  # first n-1 edges = tree
    # add extra edges (no self-loops; parallel edges are allowed and treated as independent variables)
    while length(inc) < m
        u, v = rand(rng, 1:n), rand(rng, 1:n)
        u != v && push!(inc, (u, v))
    end
    w = abs.(randn(rng, length(inc))) .+ 0.1  # positive weights
    d = randn(rng, n); d .-= mean(d)          # centered demand
    return inc, w, d, backbone_idx
end

nothing  # hide
```

## Clarabel SOCP formulation

The interior-point baseline uses a perspective relaxation. For each edge ``e``
with flow ``f_e`` and switching variable ``s_e``, we introduce an epigraph
variable ``u_e`` and the rotated second-order cone constraint
``[u_e;\, s_e;\, f_e] \in \mathcal{Q}_r^3``, enforcing ``u_e \ge f_e^2 / (2 s_e)``.
At optimality, the objective ``\sum_e 2 u_e / w_e`` equals ``d^\top L^\dagger(s)\, d``.

```math
\begin{aligned}
\min_{s,\, f,\, u} \quad & \sum_{e=1}^{m} \frac{2\, u_e}{w_e} \\
\text{s.t.} \quad & \sum_{e} s_e \le B \\
& s_e = 1, \quad e \in \mathcal{T} \\
& A^\top f = d \\
& [u_e,\, s_e,\, f_e] \in \mathcal{Q}_r^3, \quad \forall\, e \\
& 0 \le s_e \le 1, \quad u_e \ge 0, \quad \forall\, e
\end{aligned}
```

```@example graphs
function solve_clarabel(inc, w, d, backbone_idx, n, budget)
    m = length(inc)
    model = Model(Clarabel.Optimizer)
    set_silent(model)

    @variable(model, 0 <= s[1:m] <= 1)
    @variable(model, f[1:m])
    @variable(model, u[1:m] >= 0)

    # knapsack
    @constraint(model, sum(s) <= budget)
    # backbone fixed
    @constraint(model, [e in backbone_idx], s[e] == 1)
    # KCL: A'f = d (flow conservation)
    for v in 1:n
        @constraint(model,
            sum(f[e] for (e, (i, j)) in enumerate(inc) if i == v) -
            sum(f[e] for (e, (i, j)) in enumerate(inc) if j == v) == d[v])
    end
    # perspective cones: [u_e; s_e; f_e] ∈ RotatedSOC
    for e in 1:m
        @constraint(model, [u[e], s[e], f[e]] in RotatedSecondOrderCone())
    end

    @objective(model, Min, sum(2 * u[e] / w[e] for e in 1:m))
    optimize!(model)
    return objective_value(model)
end

nothing  # hide
```

## Benchmark

We draw 100 random dense graphs with ``n = 200`` nodes and ``m = 15000`` edges,
solving each with both Marguerite (Frank-Wolfe + `MaskedKnapsack`) and Clarabel
(SOCP). The small ``n`` keeps each FW Cholesky solve trivial (~``200 \times 200``),
while the large ``m`` gives Clarabel 45,000 conic constraints to handle.

```@example graphs
n_nodes = 200
m_edges = 15_000
n_trials = 100
budget = ceil(Int, 1.3 * n_nodes)

rng = Random.MersenneTwister(12345)

fw_times  = Float64[]
cl_times  = Float64[]
fw_objs   = Float64[]
cl_objs   = Float64[]

for trial in 1:n_trials
    inc, w, d, backbone_idx = random_graph(n_nodes, m_edges; rng=rng)
    m = length(inc)

    # --- Frank-Wolfe ---
    cache  = GraphCache(w, d, inc, n_nodes)
    f_obj  = GraphObj(cache)    # shares cache — safe because solver always calls ∇f! before f on trial points
    ∇f_obj = GraphGrad!(cache)
    lmo = MaskedKnapsack(budget, backbone_idx, m)
    s0 = zeros(m); s0[backbone_idx] .= 1.0

    # warmup on first trial
    if trial == 1
        solve(f_obj, lmo, s0; grad=∇f_obj, max_iters=5, tol=1e-3, verbose=false)
    end

    t_fw = @elapsed begin
        s_fw, res_fw = solve(f_obj, lmo, s0; grad=∇f_obj, max_iters=500, tol=1e-3, verbose=false)
    end
    push!(fw_times, t_fw)
    push!(fw_objs, res_fw.objective)

    # --- Clarabel SOCP ---
    if trial == 1
        solve_clarabel(inc, w, d, backbone_idx, n_nodes, budget)
    end

    t_cl = @elapsed begin
        obj_cl = solve_clarabel(inc, w, d, backbone_idx, n_nodes, budget)
    end
    push!(cl_times, t_cl)
    push!(cl_objs, obj_cl)
end

println("Frank-Wolfe  — median: ", round(median(fw_times); sigdigits=3), "s, ",
        "mean: ", round(mean(fw_times); sigdigits=3), "s")
println("Clarabel     — median: ", round(median(cl_times); sigdigits=3), "s, ",
        "mean: ", round(mean(cl_times); sigdigits=3), "s")
println("Speedup      — ", round(median(cl_times) / median(fw_times); sigdigits=3), "×")
nothing  # hide
```

## Timing comparison

```@example graphs
histogram(fw_times; nbins=20,
          title="Frank-Wolfe solve times (n=$n_nodes, m=$m_edges, $n_trials trials)",
          xlabel="seconds", width=70, color=:blue)
```

```@example graphs
histogram(cl_times; nbins=20,
          title="Clarabel SOCP solve times (n=$n_nodes, m=$m_edges, $n_trials trials)",
          xlabel="seconds", width=70, color=:red)
```

```@example graphs
boxplot(["FW", "Clarabel"], [fw_times, cl_times];
        title="Solve time comparison", xlabel="seconds", width=70)
```

## Objective agreement

Both methods solve the same continuous relaxation — the objectives should agree
up to FW's convergence tolerance:

```@example graphs
rel_gaps = abs.(fw_objs .- cl_objs) ./ max.(abs.(cl_objs), 1e-10)
println("Relative objective gap — median: ", round(median(rel_gaps); sigdigits=3),
        ", max: ", round(maximum(rel_gaps); sigdigits=3))
nothing  # hide
```
