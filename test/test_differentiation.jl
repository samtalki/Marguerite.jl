# Copyright 2026 Samuel Talkington and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

using Marguerite
using Test
using LinearAlgebra
import DifferentiationInterface as DI
using ChainRulesCore: ChainRulesCore, rrule, NoTangent

@testset "Differentiation" begin

    # Shared objective/gradient — hoisted so all testsets use the same function
    # types, avoiding redundant rule compilation per anonymous closure.
    _f(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
    _∇f!(g, x, θ) = (g .= x .- θ)

    @testset "Finite-difference Jacobian consistency" begin
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]
        kw = (; max_iters=5000, tol=1e-4)

        x_star, res = solve(_f, _∇f!, ProbabilitySimplex(), x0, θ₀; kw...)

        ε = 1e-4
        jac_fd = zeros(2, 2)
        for j in 1:2
            eⱼ = zeros(2); eⱼ[j] = 1.0
            x_plus, _ = solve(_f, _∇f!, ProbabilitySimplex(), x0, θ₀ .+ ε .* eⱼ; kw...)
            x_minus, _ = solve(_f, _∇f!, ProbabilitySimplex(), x0, θ₀ .- ε .* eⱼ; kw...)
            jac_fd[:, j] = (x_plus .- x_minus) ./ (2ε)
        end

        @test jac_fd[1, 1] > 0.3
        @test jac_fd[2, 2] > 0.3
        @test jac_fd[1, 2] < 0.1
        @test jac_fd[2, 1] < 0.1
    end

    @testset "CG solver" begin
        A = [4.0 1.0; 1.0 3.0]
        b = [1.0, 2.0]

        hvp_fn(d) = A * d
        u, cg_result = Marguerite._cg_solve(hvp_fn, b; maxiter=100, tol=1e-10, λ=0.0)
        @test norm(A * u - b) < 1e-6
        @test cg_result isa Marguerite.CGResult
        @test cg_result.converged
        @test cg_result.residual_norm < 1e-10
    end

    @testset "CG solver with regularization" begin
        A = [1.0 0.999; 0.999 1.0]
        b = [1.0, 1.0]

        hvp_fn(d) = A * d
        u, cg_result = Marguerite._cg_solve(hvp_fn, b; maxiter=100, tol=1e-10, λ=1e-2)
        @test norm((A + 1e-2 * I) * u - b) < 1e-6
        @test cg_result.converged
    end

    @testset "CG non-convergence warning" begin
        A = [4.0 1.0; 1.0 3.0]
        b = [1.0, 2.0]
        hvp_fn(d) = A * d
        u, cg_result = @test_warn "CG solve did not converge" Marguerite._cg_solve(hvp_fn, b; maxiter=1, tol=1e-15, λ=0.0)
        @test !cg_result.converged
        @test cg_result.iterations == 1
        @test cg_result.residual_norm > 1e-15
    end

    @testset "diff_* kwargs on rrule (manual gradient)" begin
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]

        (x_star, _), pb = rrule(solve, _f, _∇f!, ProbabilitySimplex(), x0, θ₀;
                                max_iters=1000, tol=1e-4,
                                diff_cg_maxiter=100, diff_cg_tol=1e-8, diff_λ=1e-3)
        x̄ = 2 .* x_star
        tangents = pb((x̄, nothing))
        θ̄ = tangents[6]
        @test length(θ̄) == 2
        @test all(isfinite, θ̄)
    end

    @testset "diff_* kwargs on rrule (auto gradient)" begin
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]

        (x_star, _), pb = rrule(solve, _f, ProbabilitySimplex(), x0, θ₀;
                                max_iters=1000, tol=1e-4,
                                diff_cg_maxiter=100, diff_cg_tol=1e-8, diff_λ=1e-3)
        x̄ = 2 .* x_star
        tangents = pb((x̄, nothing))
        θ̄ = tangents[5]
        @test length(θ̄) == 2
        @test all(isfinite, θ̄)
    end

    @testset "backend kwarg does not leak to inner solve" begin
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]
        x, res = solve(_f, _∇f!, ProbabilitySimplex(), x0, θ₀;
                       backend=DI.AutoForwardDiff(), max_iters=1000, tol=1e-2)
        @test res.objective < 0
    end

    @testset "Type promotion in rrule pullback" begin
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]
        kw = (; max_iters=5000, tol=1e-4)

        (x_star, res), pb = rrule(solve, _f, _∇f!, ProbabilitySimplex(), x0, θ₀; kw...)
        x̄ = 2 .* x_star
        tangents = pb((x̄, nothing))
        θ̄ = tangents[6]
        @test length(θ̄) == 2
        @test all(isfinite, θ̄)
    end

    @testset "Auto-gradient + θ rrule (no ∇f!)" begin
        n = 2
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]
        x_target = [0.6, 0.4]
        kw = (; max_iters=5000, tol=1e-4)

        (x_star, _), pb = rrule(solve, _f, ProbabilitySimplex(), x0, θ₀; kw...)

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
        (x_star_m, _), pb_m = rrule(solve, _f, _∇f!, ProbabilitySimplex(), x0, θ₀; kw...)
        θ̄_m = pb_m((2 .* (x_star_m .- x_target), nothing))[6]
        @test isapprox(θ̄, θ̄_m; atol=0.01)

        # Cross-check against finite differences
        L(θ_) = begin
            x_, _ = solve(_f, _∇f!, ProbabilitySimplex(), x0, θ_; kw...)
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
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]

        (_, _), pb = rrule(solve, _f, _∇f!, ProbabilitySimplex(), x0, θ₀;
                           max_iters=1000, tol=1e-4)
        tangents = pb((ChainRulesCore.ZeroTangent(), nothing))
        @test length(tangents) == 6
        @test all(t -> t isa NoTangent, tangents)

        (_, _), pb2 = rrule(solve, _f, ProbabilitySimplex(), x0, θ₀;
                            max_iters=1000, tol=1e-4)
        tangents2 = pb2((ChainRulesCore.ZeroTangent(), nothing))
        @test length(tangents2) == 5
        @test all(t -> t isa NoTangent, tangents2)
    end

    @testset "rrule default hvp_backend (manual grad)" begin
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]
        kw = (; max_iters=5000, tol=1e-4)

        (x_star, _), pb = rrule(solve, _f, _∇f!, ProbabilitySimplex(), x0, θ₀; kw...)
        @test length(x_star) == 2

        θ̄ = pb((2 .* x_star, nothing))[end]
        @test length(θ̄) == 2
        @test all(isfinite, θ̄)
    end

    # ------------------------------------------------------------------
    # Parameterized constraint set tests
    # ------------------------------------------------------------------

    @testset "ParameterizedBox rrule (manual gradient)" begin
        n = 3
        # f(x, θ) = 0.5||x||² - θ'x, box bounds from θ
        # θ = [lb₁, lb₂, lb₃, ub₁, ub₂, ub₃]
        _f_box(x, θ) = 0.5 * dot(x, x) - dot(θ[1:length(x)], x)
        _∇f_box!(g, x, θ) = (g .= x .- θ[1:length(x)])

        plmo = ParameterizedBox(θ -> θ[1:n], θ -> θ[n+1:2n])
        θ₀ = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]  # box [0,1]^3
        x0 = [0.5, 0.5, 0.5]
        kw = (; max_iters=5000, tol=1e-6)

        # Objective pulls x toward θ[1:3] = 0, but lb = 0, so x* = [0,0,0]
        (x_star, res), pb = rrule(solve, _f_box, _∇f_box!, plmo, x0, θ₀; kw...)
        @test all(x_star .≈ 0.0)

        x̄ = ones(n)
        tangents = pb((x̄, nothing))
        θ̄ = tangents[6]
        @test length(θ̄) == 2n
        @test all(isfinite, θ̄)

        # Finite-difference cross-check
        ε = 1e-4
        θ̄_fd = zeros(2n)
        L(θ_) = begin
            x_, _ = solve(_f_box, _∇f_box!, plmo, x0, θ_; kw...)
            dot(ones(n), x_)
        end
        for j in 1:2n
            eⱼ = zeros(2n); eⱼ[j] = 1.0
            θ̄_fd[j] = (L(θ₀ .+ ε .* eⱼ) - L(θ₀ .- ε .* eⱼ)) / (2ε)
        end
        @test isapprox(θ̄, θ̄_fd; atol=0.1)
    end

    @testset "ParameterizedProbSimplex rrule" begin
        n = 2
        # f(x, θ) = 0.5||x||² - θ[1:n]'x, r = θ[end]
        _f_simp(x, θ) = 0.5 * dot(x, x) - dot(θ[1:length(x)], x)
        _∇f_simp!(g, x, θ) = (g .= x .- θ[1:length(x)])

        plmo = ParameterizedProbSimplex(θ -> θ[end])
        θ₀ = [0.7, 0.3, 1.0]  # θ[1:2] are objective params, θ[3] = radius
        x0 = [0.5, 0.5]
        kw = (; max_iters=5000, tol=1e-6)

        (x_star, res), pb = rrule(solve, _f_simp, _∇f_simp!, plmo, x0, θ₀; kw...)
        @test sum(x_star) ≈ 1.0 atol=1e-3

        x̄ = 2.0 .* x_star
        tangents = pb((x̄, nothing))
        θ̄ = tangents[6]
        @test length(θ̄) == 3
        @test all(isfinite, θ̄)

        # Finite-difference cross-check (FW convergence on simplex is O(1/t),
        # so FD is less accurate for the radius component; use wider tolerance)
        ε = 1e-4
        θ̄_fd = zeros(3)
        fd_kw = (; max_iters=50000, tol=1e-6)
        L(θ_) = begin
            x_, _ = solve(_f_simp, _∇f_simp!, plmo, x0, θ_; fd_kw...)
            dot(x_, x_)
        end
        for j in 1:3
            eⱼ = zeros(3); eⱼ[j] = 1.0
            θ̄_fd[j] = (L(θ₀ .+ ε .* eⱼ) - L(θ₀ .- ε .* eⱼ)) / (2ε)
        end
        @test isapprox(θ̄[1:2], θ̄_fd[1:2]; atol=0.1)
        @test isapprox(θ̄[3], θ̄_fd[3]; atol=0.25)
    end

    @testset "ParameterizedBox rrule (auto gradient)" begin
        n = 2
        _f_box2(x, θ) = 0.5 * dot(x, x) - dot(θ[1:length(x)], x)
        plmo = ParameterizedBox(θ -> θ[1:n], θ -> θ[n+1:2n])
        θ₀ = [0.3, 0.7, 1.0, 1.0]
        x0 = [0.5, 0.5]
        kw = (; max_iters=5000, tol=1e-6)

        (x_star, res), pb = rrule(solve, _f_box2, plmo, x0, θ₀; kw...)
        @test all(isfinite, x_star)

        x̄ = ones(n)
        tangents = pb((x̄, nothing))
        θ̄ = tangents[5]
        @test length(θ̄) == 2n
        @test all(isfinite, θ̄)
    end

    @testset "ZeroTangent with ParameterizedOracle returns all NoTangent" begin
        n = 2
        _f_z(x, θ) = 0.5 * dot(x, x) - dot(θ[1:length(x)], x)
        _∇f_z!(g, x, θ) = (g .= x .- θ[1:length(x)])
        plmo = ParameterizedBox(θ -> zeros(n), θ -> ones(n))

        θ₀ = [0.5, 0.5]
        x0 = [0.5, 0.5]

        (_, _), pb = rrule(solve, _f_z, _∇f_z!, plmo, x0, θ₀; max_iters=1000, tol=1e-4)
        tangents = pb((ChainRulesCore.ZeroTangent(), nothing))
        @test length(tangents) == 6
        @test all(t -> t isa NoTangent, tangents)

        (_, _), pb2 = rrule(solve, _f_z, plmo, x0, θ₀; max_iters=1000, tol=1e-4)
        tangents2 = pb2((ChainRulesCore.ZeroTangent(), nothing))
        @test length(tangents2) == 5
        @test all(t -> t isa NoTangent, tangents2)
    end

    @testset "Boundary solution (vertex of simplex) -- KKT correctness" begin
        # θ = [10.0, 0.0] pushes x* to vertex e_1 of the probability simplex.
        # At a vertex, the unconstrained Hessian solve would be wrong because
        # the active set matters. The KKT adjoint should handle this correctly.
        n = 2
        θ₀ = [10.0, 0.0]
        x0 = [0.5, 0.5]
        x_target = [0.8, 0.2]
        kw = (; max_iters=10000, tol=1e-6)

        (x_star, _), pb = rrule(solve, _f, _∇f!, ProbabilitySimplex(), x0, θ₀; kw...)
        # x* should be at vertex e_1 = [1.0, 0.0]
        @test x_star[1] ≈ 1.0 atol=1e-3
        @test x_star[2] ≈ 0.0 atol=1e-3

        x̄ = 2.0 .* (x_star .- x_target)
        tangents = pb((x̄, nothing))
        θ̄ = tangents[6]
        @test length(θ̄) == 2
        @test all(isfinite, θ̄)

        # FD cross-check
        ε = 1e-3
        L(θ_) = begin
            x_, _ = solve(_f, _∇f!, ProbabilitySimplex(), x0, θ_; kw...)
            sum((x_ .- x_target).^2)
        end
        θ̄_fd = zeros(n)
        for j in 1:n
            eⱼ = zeros(n); eⱼ[j] = 1.0
            θ̄_fd[j] = (L(θ₀ .+ ε .* eⱼ) - L(θ₀ .- ε .* eⱼ)) / (2ε)
        end
        @test isapprox(θ̄, θ̄_fd; atol=0.15)
    end

end
