using Marguerite
using Test
using LinearAlgebra
using Random
using UnicodePlots

@testset "FW Convergence" begin
    # Problem: min_{x ∈ Δ_20} 0.5 x'Qx + c'x
    # with random PD matrix Q
    Random.seed!(42)
    n = 20
    A = randn(n, n)
    Q = A'A + 0.1I  # positive definite
    c = randn(n)

    f(x) = 0.5 * dot(x, Q * x) + dot(c, x)
    ∇f!(g, x) = (g .= Q * x .+ c)

    lmo = ProbabilitySimplex()
    # Start from a vertex so sparsity bound nnz ≤ t+1 holds
    x0 = zeros(n); x0[1] = 1.0
    max_iters = 2000

    # --- Solve to optimality for reference ---
    x_opt, res_opt = solve(f, ∇f!, lmo, x0;
                           max_iters=50000, tol=1e-12, monotonic=false)
    f_opt = f(x_opt)

    # --- Hand-written FW loop to collect per-iteration history ---
    x = copy(x0)
    g = zeros(n)
    v = zeros(n)
    step = Marguerite.MonotonicStepSize()

    primal_gaps = Float64[]
    fw_gaps = Float64[]
    sparsities = Int[]

    for t in 0:(max_iters - 1)
        ∇f!(g, x)
        lmo(v, g)

        # Frank-Wolfe gap: ⟨∇f, x - v⟩
        gap = dot(g, x .- v)
        push!(primal_gaps, f(x) - f_opt)
        push!(fw_gaps, gap)
        push!(sparsities, count(xi -> abs(xi) > 1e-12, x))

        # Update
        γ = step(t)
        x .= x .+ γ .* (v .- x)
    end

    # --- Assertions ---
    @testset "Primal gap decreases by 100x" begin
        @test primal_gaps[end] < primal_gaps[1] / 100
    end

    @testset "Sparsity ≤ t+1" begin
        for t in 1:min(50, max_iters)
            @test sparsities[t] ≤ t + 1
        end
    end

    @testset "Manual loop matches solve()" begin
        x_solve, res_solve = solve(f, ∇f!, lmo, x0;
                                   max_iters=max_iters, tol=0.0, monotonic=false)
        @test isapprox(f(x), f(x_solve); atol=1e-6)
    end

    # --- Plots (printed to test output) ---
    println("\n── FW Convergence Diagnostics ──\n")

    iters = 1:max_iters

    println(lineplot(iters, primal_gaps;
                     xscale=:log10, yscale=:log10,
                     title="Primal Gap f(xₜ) - f*",
                     xlabel="iteration", ylabel="gap",
                     name="primal gap", width=60))
    println()

    println(lineplot(iters, fw_gaps;
                     yscale=:log10,
                     title="Frank-Wolfe Duality Gap",
                     xlabel="iteration", ylabel="⟨∇f, x-v⟩",
                     name="FW gap", width=60))
    println()

    println(lineplot(iters, sparsities;
                     title="Iterate Sparsity (nnz)",
                     xlabel="iteration", ylabel="nnz(xₜ)",
                     name="nnz", width=60))
end
