using Marguerite
using Test
using LinearAlgebra
import DifferentiationInterface as DI
import ForwardDiff

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
end
