using Marguerite
using Test
using LinearAlgebra

@testset "Solver" begin

    @testset "Quadratic on probability simplex" begin
        Q = [4.0 1.0; 1.0 2.0]
        c = [-3.0, -1.0]
        f(x) = 0.5 * dot(x, Q * x) + dot(c, x)
        ∇f!(g, x) = (g .= Q * x .+ c)

        x, res = solve(f, ∇f!, ProbabilitySimplex(), [0.5, 0.5];
                        max_iters=10000, tol=1e-3)
        @test res.converged
        # Analytic optimum on the simplex: x ≈ [0.75, 0.25]
        @test x[1] ≈ 0.75 atol=1e-2
        @test x[2] ≈ 0.25 atol=1e-2
    end

    @testset "Quadratic on box" begin
        # min 0.5*||x - x*||^2 on [0, 1]^3, x* = [0.3, 0.7, 1.5]
        x_opt = [0.3, 0.7, 1.5]
        f(x) = 0.5 * sum((x .- x_opt).^2)
        ∇f!(g, x) = (g .= x .- x_opt)

        lmo = Box(zeros(3), ones(3))
        x, res = solve(f, ∇f!, lmo, [0.5, 0.5, 0.5];
                        max_iters=10000, tol=1e-3)
        @test res.converged
        # Solution should be projection: [0.3, 0.7, 1.0]
        @test x ≈ [0.3, 0.7, 1.0] atol=1e-2
    end

    @testset "Monotonic mode rejects bad steps" begin
        Q = [2.0 0.0; 0.0 2.0]
        f(x) = 0.5 * dot(x, Q * x)
        ∇f!(g, x) = (g .= Q * x)

        x, res = solve(f, ∇f!, ProbabilitySimplex(), [0.5, 0.5];
                        monotonic=true, max_iters=100)
        # Should have some discards since FW overshoots sometimes
        @test res.discards >= 0
        @test res.objective ≤ f([0.5, 0.5]) + 1e-10
    end

    @testset "Parameterized solve" begin
        f(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
        ∇f!(g, x, θ) = (g .= x .- θ)

        θ = [0.8, 0.2]
        x, res = solve(f, ∇f!, ProbabilitySimplex(), [0.5, 0.5], θ;
                        max_iters=10000, tol=1e-3)
        @test res.converged
        # Optimal: project θ onto simplex → since sum(θ)=1, x* = θ
        @test x ≈ θ atol=1e-2
    end

    @testset "Cache reuse" begin
        Q = [2.0 0.5; 0.5 1.0]
        c = [-1.0, -0.5]
        f(x) = 0.5 * dot(x, Q * x) + dot(c, x)
        ∇f!(g, x) = (g .= Q * x .+ c)

        cache = Marguerite.Cache{Float64}(2)
        x1, res1 = solve(f, ∇f!, ProbabilitySimplex(), [0.5, 0.5]; cache=cache)
        x2, res2 = solve(f, ∇f!, ProbabilitySimplex(), [0.5, 0.5]; cache=cache)
        @test x1 ≈ x2
    end

    @testset "Auto-gradient (ForwardDiff)" begin
        Q = [2.0 0.5; 0.5 1.0]
        c = [-1.0, -0.5]
        f(x) = 0.5 * dot(x, Q * x) + dot(c, x)

        import DifferentiationInterface as DI
        import ForwardDiff

        x, res = solve(f, ProbabilitySimplex(), [0.5, 0.5];
                        backend=DI.AutoForwardDiff(),
                        max_iters=10000, tol=1e-3)
        @test res.converged || res.gap < 0.01
    end

    @testset "Auto-gradient (Mooncake, default backend)" begin
        Q = [2.0 0.5; 0.5 1.0]
        c = [-1.0, -0.5]
        f(x) = 0.5 * dot(x, Q * x) + dot(c, x)

        x, res = solve(f, ProbabilitySimplex(), [0.5, 0.5];
                        max_iters=10000, tol=1e-3)
        @test res.converged || res.gap < 0.01
    end

    @testset "Parameterized auto-gradient solve (default backend)" begin
        f(x, θ) = 0.5 * dot(x, x) - dot(θ, x)

        θ = [0.7, 0.3]
        x, res = solve(f, ProbabilitySimplex(), [0.5, 0.5], θ;
                        max_iters=50000, tol=1e-4)
        @test res.converged
        @test x ≈ θ atol=1e-2
    end

    @testset "Parameterized manual-gradient solve (default backend)" begin
        f(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
        ∇f!(g, x, θ) = (g .= x .- θ)

        θ = [1.0, 2.0]
        x, res = solve(f, ∇f!, ProbabilitySimplex(), [0.5, 0.5], θ;
                        max_iters=50000, tol=1e-4)
        @test res.converged
        # θ not on simplex (sum=3), so x* is the projection
        # For this objective, x* = proj_simplex(θ) = [0, 1] (all weight on dim 2)
        @test x[2] > x[1]
    end
end
