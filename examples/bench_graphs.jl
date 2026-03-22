# Full 100-trial benchmark: Frank-Wolfe vs Clarabel SOCP on random graph problems.
# This is the standalone version of the benchmark summarized in docs/src/graphs.md.

using Marguerite, LinearAlgebra, SparseArrays, Random, Statistics
using JuMP, Clarabel
using UnicodePlots

# --- helpers (same as docs/src/graphs.md) ---

struct GraphCache{T}
    w::Vector{T}
    d::Vector{T}
    inc::Vector{Tuple{Int,Int}}
    n::Int
    L_red::Matrix{T}
    x::Vector{T}
    d_red::Vector{T}
end

function GraphCache(w::Vector{T}, d::Vector{T}, inc, n) where T
    GraphCache{T}(w, d, inc, n,
        zeros(T, n-1, n-1), zeros(T, n), d[1:n-1])
end

function laplacian_solve!(cache::GraphCache{T}, s) where T
    n = cache.n; nr = n - 1
    L = cache.L_red
    fill!(L, zero(T))
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

function random_graph(n, m; rng=Random.default_rng())
    inc = Tuple{Int,Int}[]
    perm = randperm(rng, n)
    for i in 2:n
        push!(inc, (perm[i], perm[rand(rng, 1:i-1)]))
    end
    backbone_idx = collect(1:n-1)
    while length(inc) < m
        u, v = rand(rng, 1:n), rand(rng, 1:n)
        u != v && push!(inc, (u, v))
    end
    w = abs.(randn(rng, length(inc))) .+ 0.1
    d = randn(rng, n); d .-= mean(d)
    return inc, w, d, backbone_idx
end

function solve_clarabel(inc, w, d, backbone_idx, n, budget)
    m = length(inc)
    model = Model(Clarabel.Optimizer)
    set_silent(model)

    @variable(model, 0 <= s[1:m] <= 1)
    @variable(model, f[1:m])
    @variable(model, u[1:m] >= 0)

    @constraint(model, sum(s) <= budget)
    @constraint(model, [e in backbone_idx], s[e] == 1)
    for v in 1:n
        @constraint(model,
            sum(f[e] for (e, (i, j)) in enumerate(inc) if i == v) -
            sum(f[e] for (e, (i, j)) in enumerate(inc) if j == v) == d[v])
    end
    for e in 1:m
        @constraint(model, [u[e], s[e], f[e]] in RotatedSecondOrderCone())
    end

    @objective(model, Min, sum(2 * u[e] / w[e] for e in 1:m))
    optimize!(model)
    return objective_value(model)
end

# --- benchmark ---

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
    f_obj  = GraphObj(cache)
    ∇f_obj = GraphGrad!(cache)
    lmo = MaskedKnapsack(budget, backbone_idx, m)
    s0 = zeros(m); s0[backbone_idx] .= 1.0

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

    trial % 10 == 0 && println("trial $trial / $n_trials done")
end

# --- results ---

println("\n=== Results ($n_trials trials, n=$n_nodes, m=$m_edges) ===")
println("Frank-Wolfe  — median: ", round(median(fw_times); sigdigits=3), "s, ",
        "mean: ", round(mean(fw_times); sigdigits=3), "s")
println("Clarabel     — median: ", round(median(cl_times); sigdigits=3), "s, ",
        "mean: ", round(mean(cl_times); sigdigits=3), "s")
println("Speedup      — ", round(median(cl_times) / median(fw_times); sigdigits=3), "×")

rel_gaps = abs.(fw_objs .- cl_objs) ./ max.(abs.(cl_objs), 1e-10)
println("Relative gap — median: ", round(median(rel_gaps); sigdigits=3),
        ", max: ", round(maximum(rel_gaps); sigdigits=3))

println()
println(histogram(fw_times; nbins=20,
          title="Frank-Wolfe solve times",
          xlabel="seconds", width=70, color=:blue))
println(histogram(cl_times; nbins=20,
          title="Clarabel SOCP solve times",
          xlabel="seconds", width=70, color=:red))
println(boxplot(["FW", "Clarabel"], [fw_times, cl_times];
        title="Solve time comparison", xlabel="seconds", width=70))
