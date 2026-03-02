using Marguerite
using Test
using LinearAlgebra
using Random
using ChainRulesCore: ChainRulesCore, rrule, NoTangent
import DifferentiationInterface as DI
import ForwardDiff
using UnicodePlots

@testset "Bilevel Optimization" begin
    Random.seed!(123)
    n = 5

    # Random PD Hessian (not identity -- makes x*(θ) nontrivial)
    A = randn(n, n)
    H = A'A + 0.5I

    # Inner problem: min_{x ∈ Δ_n} 0.5 x'Hx - θ'x
    f(x, θ) = 0.5 * dot(x, H * x) - dot(θ, x)
    ∇f!(g, x, θ) = (g .= H * x .- θ)

    lmo = ProbabilitySimplex()
    x0 = fill(1.0 / n, n)
    # Use ForwardDiff for HVPs -- Mooncake can't do reverse-over-reverse here
    backend = DI.AutoForwardDiff()
    solve_kw = (; max_iters=10000, tol=1e-6, backend=backend)

    # Target on the simplex
    x_target = zeros(n)
    x_target[1] = 0.6; x_target[2] = 0.3; x_target[3] = 0.1

    # Bilevel step: solve inner, compute outer loss gradient via rrule
    function bilevel_step(θ)
        (x_star, result), pb = rrule(solve, f, ∇f!, lmo, x0, θ; solve_kw...)
        loss = sum((x_star .- x_target).^2)
        x̄ = 2.0 .* (x_star .- x_target)
        tangents = pb((x̄, nothing))
        θ̄ = tangents[end]
        return x_star, loss, θ̄
    end

    @testset "Bilevel convergence" begin
        θ = H * x_target  # warm start: near-optimal θ for identity H case
        η = 0.1
        outer_iters = 80

        losses = Float64[]
        errors = Float64[]

        for k in 1:outer_iters
            x_star, loss, θ̄ = bilevel_step(θ)
            push!(losses, loss)
            push!(errors, norm(x_star .- x_target))
            θ .= θ .- η .* θ̄
        end

        @test losses[end] < 1e-4
        x_final, _ = solve(f, ∇f!, lmo, x0, θ; solve_kw...)
        @test isapprox(x_final, x_target; atol=1e-2)

        # --- Plot ---
        println("\n── Bilevel Optimization ──\n")
        println(lineplot(1:outer_iters, log10.(losses);
                         title="Outer Loss (log₁₀)",
                         xlabel="outer iteration", ylabel="log₁₀(loss)",
                         name="loss", width=60))
        println()
        println(lineplot(1:outer_iters, errors;
                         title="Solution Error ‖x*(θ) - x_target‖",
                         xlabel="outer iteration", ylabel="error",
                         name="‖x*-x_t‖", width=60))
    end

    @testset "AD gradient matches finite differences" begin
        # Use identity Hessian for clean FD: x*(θ) = θ when θ is interior.
        # FW's O(1/t) convergence needs many iterations for tight FD checks,
        # so we use the simple H=I case where convergence is fastest.
        f_id(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
        ∇f_id!(g, x, θ) = (g .= x .- θ)

        θ_test = [0.3, 0.25, 0.2, 0.15, 0.1]
        fd_kw = (; max_iters=50000, tol=1e-8, backend=backend)

        (x_ad, _), pb = rrule(solve, f_id, ∇f_id!, lmo, x0, θ_test; fd_kw...)
        x̄ = 2.0 .* (x_ad .- x_target)
        θ̄_ad = pb((x̄, nothing))[end]

        # Finite-difference gradient of outer loss w.r.t. θ
        # Use ε large enough to dominate FW's O(1/t) solver noise
        ε = 1e-3
        θ̄_fd = zeros(n)
        for j in 1:n
            eⱼ = zeros(n); eⱼ[j] = 1.0
            x_plus, _ = solve(f_id, ∇f_id!, lmo, x0, θ_test .+ ε .* eⱼ; fd_kw...)
            x_minus, _ = solve(f_id, ∇f_id!, lmo, x0, θ_test .- ε .* eⱼ; fd_kw...)
            loss_plus = sum((x_plus .- x_target).^2)
            loss_minus = sum((x_minus .- x_target).^2)
            θ̄_fd[j] = (loss_plus - loss_minus) / (2ε)
        end

        @test isapprox(θ̄_ad, θ̄_fd; atol=0.02)
    end
end
