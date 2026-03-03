# bench/bench_fw_advantage.jl
#
# Benchmark: Marguerite vs Clarabel on high-dimensional sparse recovery
#
# Problem: min 0.5 ||Ax - b||²  s.t.  x ∈ probability simplex
#
# A ∈ R^{m×n} with m << n. FW only needs O(mn) gradient evals + O(n) LMO,
# while interior-point must form Q = A'A ∈ R^{n×n} explicitly.
#
# Sizes: (m=50, n=5000), (m=100, n=20000), (m=100, n=50000)
# Clarabel included for n=5000 only (larger sizes are prohibitive).

using Marguerite
using LinearAlgebra
using BenchmarkTools
using Random
using Printf
using JuMP
using Clarabel

# ── problem setup ─────────────────────────────────────────────────

function make_problem(m, n; seed=42)
    rng = Random.Xoshiro(seed)
    A = randn(rng, m, n)
    # sparse ground truth on the simplex (5 active components)
    x_true = zeros(n)
    support = sort(randperm(rng, n)[1:min(5, n)])
    x_true[support] .= 1.0 / length(support)
    b = A * x_true
    return A, b, x_true
end

# ── Marguerite objective + gradient (zero-alloc callable structs) ─

struct LeastSquaresObj{M<:AbstractMatrix, V<:AbstractVector}
    A::M
    b::V
    r::V  # workspace: residual A*x - b
end

LeastSquaresObj(A::Matrix{Float64}, b::Vector{Float64}) =
    LeastSquaresObj(A, b, similar(b))

function (obj::LeastSquaresObj)(x)
    mul!(obj.r, obj.A, x)
    obj.r .-= obj.b
    return 0.5 * dot(obj.r, obj.r)
end

struct LeastSquaresGrad!{M<:AbstractMatrix, V<:AbstractVector}
    A::M
    b::V
    r::V
end

LeastSquaresGrad!(A::Matrix{Float64}, b::Vector{Float64}) =
    LeastSquaresGrad!(A, b, similar(b))

function (∇f::LeastSquaresGrad!)(g, x)
    mul!(∇f.r, ∇f.A, x)
    ∇f.r .-= ∇f.b
    mul!(g, ∇f.A', ∇f.r)  # g = A'(Ax - b)
    return g
end

# ── Clarabel via JuMP ────────────────────────────────────────────

function solve_clarabel(A, b)
    m, n = size(A)
    Q = A' * A  # n×n — this is the bottleneck
    q = -A' * b

    model = Model(Clarabel.Optimizer)
    set_silent(model)
    @variable(model, x[1:n] >= 0)
    @constraint(model, sum(x) == 1.0)
    @objective(model, Min, 0.5 * x' * Q * x + q' * x)
    optimize!(model)

    x_val = value.(x)
    obj = 0.5 * norm(A * x_val - b)^2
    return x_val, obj
end

# ── benchmark runner ─────────────────────────────────────────────

function run_bench(m, n; run_clarabel=false, fw_iters=2000)
    println("\n", "="^60)
    @printf("  m = %d, n = %s\n", m, format_n(n))
    println("="^60)

    A, b, x_true = make_problem(m, n)

    # --- Marguerite ---
    f_obj = LeastSquaresObj(A, b)
    ∇f_obj = LeastSquaresGrad!(A, b)
    # wrap callable structs in closures for ::Function dispatch
    f = x -> f_obj(x)
    ∇f! = (g, x) -> ∇f_obj(g, x)
    lmo = ProbSimplex()
    # start from vertex e_1 so FW iterate stays sparse
    x0 = zeros(n); x0[1] = 1.0

    # warmup
    solve(f, ∇f!, lmo, copy(x0); max_iters=5, step_rule=MonotonicStepSize())

    print("  Marguerite FW:  ")
    t_fw = @btime solve($f, $∇f!, $lmo, x0_; max_iters=$fw_iters, step_rule=MonotonicStepSize()) setup=(x0_ = begin; v = zeros($n); v[1] = 1.0; v; end) evals=1

    # get actual solution for reporting
    x_fw, result = t_fw
    obj_fw = result.objective
    nnz_fw = count(>(1e-8), x_fw)
    recovery_err = norm(x_fw - x_true)

    @printf("    objective   = %.6e\n", obj_fw)
    @printf("    nnz(x)      = %d / %d  (%.1f%% sparse)\n", nnz_fw, n, 100 * (1 - nnz_fw / n))
    @printf("    recovery    = %.6e\n", recovery_err)
    @printf("    converged   = %s  (iters=%d, gap=%.2e)\n",
            result.converged, result.iterations, result.gap)

    # --- Clarabel ---
    if run_clarabel
        # warmup with tiny problem
        A_tiny, b_tiny, _ = make_problem(5, 20; seed=99)
        solve_clarabel(A_tiny, b_tiny)

        print("  Clarabel QP:   ")
        x_c, obj_c = @btime solve_clarabel($A, $b) evals=1
        nnz_c = count(>(1e-8), x_c)
        @printf("    objective   = %.6e\n", obj_c)
        @printf("    nnz(x)      = %d / %d\n", nnz_c, n)
    end
end

format_n(n) = n >= 1000 ? @sprintf("%dk", n ÷ 1000) : string(n)

# ── main ─────────────────────────────────────────────────────────

println("Benchmark: Frank-Wolfe vs Interior Point on sparse recovery")
println("  min 0.5 ||Ax - b||²  s.t.  x ∈ Δ_n  (probability simplex)")
println("  A ∈ R^{m×n}, m << n")

run_bench(50,  5_000;  run_clarabel=true,  fw_iters=2000)
run_bench(100, 20_000; run_clarabel=false, fw_iters=2000)
run_bench(100, 50_000; run_clarabel=false, fw_iters=2000)

println("\n", "="^60)
println("  Done.")
println("="^60)
