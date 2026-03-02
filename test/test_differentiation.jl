using Marguerite
using Test
using LinearAlgebra
import DifferentiationInterface as DI
import ForwardDiff
using ChainRulesCore: ChainRulesCore, rrule, NoTangent

@testset "Differentiation" begin

    @testset "Finite-difference Jacobian consistency" begin
        # f(x, θ) = 0.5 ||x||^2 - θ'x on probability simplex
        # For θ on the simplex interior with θ_i > 0, the unconstrained
        # optimum is x* = θ, which lies on the simplex when sum(θ)=1.
        # So x*(θ) ≈ θ and dx*/dθ ≈ I locally.

        f(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
        ∇f!(g, x, θ) = (g .= x .- θ)

        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]
        kw = (; max_iters=50000, tol=1e-5)

        x_star, res = solve(f, ∇f!, ProbabilitySimplex(), x0, θ₀; kw...)

        # Finite difference: ∂x*/∂θ_j
        ε = 1e-4
        jac_fd = zeros(2, 2)
        for j in 1:2
            eⱼ = zeros(2); eⱼ[j] = 1.0
            x_plus, _ = solve(f, ∇f!, ProbabilitySimplex(), x0, θ₀ .+ ε .* eⱼ; kw...)
            x_minus, _ = solve(f, ∇f!, ProbabilitySimplex(), x0, θ₀ .- ε .* eⱼ; kw...)
            jac_fd[:, j] = (x_plus .- x_minus) ./ (2ε)
        end

        # The FD Jacobian should have positive diagonal (increasing θ_j increases x_j)
        @test jac_fd[1, 1] > 0.3
        @test jac_fd[2, 2] > 0.3
        # Off-diagonal should be negative (simplex constraint: increasing θ_1 decreases x_2)
        @test jac_fd[1, 2] < 0.1
        @test jac_fd[2, 1] < 0.1
    end

    @testset "CG solver" begin
        A = [4.0 1.0; 1.0 3.0]
        b = [1.0, 2.0]

        hvp_fn(d) = A * d
        u = Marguerite._cg_solve(hvp_fn, b; maxiter=100, tol=1e-10, λ=0.0)
        @test norm(A * u - b) < 1e-6
    end

    @testset "CG solver with regularization" begin
        # Near-singular system
        A = [1.0 0.999; 0.999 1.0]
        b = [1.0, 1.0]

        hvp_fn(d) = A * d
        u = Marguerite._cg_solve(hvp_fn, b; maxiter=100, tol=1e-10, λ=1e-2)
        # (A + λI)u ≈ b
        @test norm((A + 1e-2 * I) * u - b) < 1e-6
    end

    @testset "backend kwarg does not leak to inner solve" begin
        f(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
        ∇f!(g, x, θ) = (g .= x .- θ)
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]
        # Pre-fix: backend leaked through kwargs... to inner solve, causing MethodError
        x, res = solve(f, ∇f!, ProbabilitySimplex(), x0, θ₀;
                       backend=DI.AutoForwardDiff(), max_iters=1000, tol=1e-2)
        @test res.objective < 0
    end

    @testset "ForwardDiff Dual type promotion in rrule pullback" begin
        f(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
        ∇f!(g, x, θ) = (g .= x .- θ)
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]
        kw = (; max_iters=50000, tol=1e-8)

        (x_star, res), pb = rrule(solve, f, ∇f!, ProbabilitySimplex(), x0, θ₀;
                                  backend=DI.AutoForwardDiff(), kw...)
        # Pre-fix: similar(x_star) created Float64 buffer; writing Duals threw InexactError
        x̄ = 2 .* x_star
        tangents = pb((x̄, nothing))
        θ̄ = tangents[6]
        @test length(θ̄) == 2
        @test all(isfinite, θ̄)
    end

    @testset "Auto-gradient + θ rrule (no ∇f!)" begin
        # Use the same identity-Hessian problem as bilevel FD test.
        # x̄ = 2(x*-x_target) has sum=0 (both on simplex), so the
        # unconstrained IFT matches the true constrained sensitivity.
        f(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
        ∇f!(g, x, θ) = (g .= x .- θ)
        n = 5
        θ₀ = [0.3, 0.25, 0.2, 0.15, 0.1]
        x0 = fill(1.0 / n, n)
        x_target = [0.4, 0.3, 0.15, 0.1, 0.05]
        backend = DI.AutoForwardDiff()
        kw = (; max_iters=50000, tol=1e-8, backend=backend)

        # Auto-gradient rrule (4 positional args: solve, f, lmo, x0, θ)
        (x_star, _), pb = rrule(solve, f, ProbabilitySimplex(), x0, θ₀; kw...)

        # Pullback structure: 5-tuple (solve, f, lmo, x0, θ)
        x̄ = 2 .* (x_star .- x_target)
        tangents = pb((x̄, nothing))
        @test length(tangents) == 5
        @test tangents[1] isa NoTangent
        @test tangents[2] isa NoTangent
        @test tangents[3] isa NoTangent
        @test tangents[4] isa NoTangent
        θ̄ = tangents[5]
        @test length(θ̄) == n
        @test all(isfinite, θ̄)

        # Cross-check: auto-gradient rrule should match manual-gradient rrule
        (x_star_m, _), pb_m = rrule(solve, f, ∇f!, ProbabilitySimplex(), x0, θ₀; kw...)
        θ̄_m = pb_m((2 .* (x_star_m .- x_target), nothing))[6]
        @test isapprox(θ̄, θ̄_m; atol=0.01)

        # Cross-check against finite differences of L(θ) = ||x*(θ) - x_target||²
        L(θ_) = begin
            x_, _ = solve(f, ∇f!, ProbabilitySimplex(), x0, θ_; kw...)
            sum((x_ .- x_target) .^ 2)
        end
        ε = 1e-3
        θ̄_fd = zeros(n)
        for j in 1:n
            eⱼ = zeros(n); eⱼ[j] = 1.0
            θ̄_fd[j] = (L(θ₀ .+ ε .* eⱼ) - L(θ₀ .- ε .* eⱼ)) / (2ε)
        end
        @test isapprox(θ̄, θ̄_fd; atol=0.05)
    end

    @testset "ZeroTangent pullback returns all NoTangent" begin
        f(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
        ∇f!(g, x, θ) = (g .= x .- θ)
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]
        backend = DI.AutoForwardDiff()

        # Manual-gradient rrule: 6-tuple
        (_, _), pb = rrule(solve, f, ∇f!, ProbabilitySimplex(), x0, θ₀;
                           backend=backend, max_iters=1000, tol=1e-4)
        tangents = pb((ChainRulesCore.ZeroTangent(), nothing))
        @test length(tangents) == 6
        @test all(t -> t isa NoTangent, tangents)

        # Auto-gradient rrule: 5-tuple
        (_, _), pb2 = rrule(solve, f, ProbabilitySimplex(), x0, θ₀;
                            backend=backend, max_iters=1000, tol=1e-4)
        tangents2 = pb2((ChainRulesCore.ZeroTangent(), nothing))
        @test length(tangents2) == 5
        @test all(t -> t isa NoTangent, tangents2)
    end
end
