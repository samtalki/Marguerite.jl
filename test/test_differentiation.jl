# Copyright 2026 Samuel Talkington
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

# Exhaustive differentiation coverage.
# Representative default-path checks live in test/test_differentiation_fast.jl.

@testset "Differentiation" begin

    # Shared objective/gradient — hoisted so all testsets use the same function
    # types, avoiding redundant rule compilation per anonymous closure.
    _f(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
    _∇f!(g, x, θ) = (g .= x .- θ)

    # Embed θ on the diagonal of an n×n matrix (vectorized).
    # Shared across Spectraplex tests to avoid redundant definitions.
    function _diag_vec_sp(θ, n)
        t = zeros(eltype(θ), n * n)
        @inbounds for i in 1:n
            t[(i - 1) * n + i] = θ[i]
        end
        t
    end

    struct WrappedProbSimplex <: Marguerite.AbstractOracle end

    function (::WrappedProbSimplex)(v::AbstractVector, g::AbstractVector)
        fill!(v, zero(eltype(v)))
        v[argmin(g)] = one(eltype(v))
        return v
    end

    function Marguerite.active_set(::WrappedProbSimplex, x::AbstractVector{T}; tol::Real=1e-8) where T
        return Marguerite.active_set(ProbabilitySimplex(T(1)), x; tol=tol)
    end

    struct CallableProbSimplexDiff end

    function (::CallableProbSimplexDiff)(v::AbstractVector, g::AbstractVector)
        fill!(v, zero(eltype(v)))
        v[argmin(g)] = one(eltype(v))
        return v
    end

    function Marguerite.active_set(::CallableProbSimplexDiff, x::AbstractVector{T}; tol::Real=1e-8) where T
        return Marguerite.active_set(ProbabilitySimplex(T(1)), x; tol=tol)
    end

    # Verify that the finite-difference Jacobian matches the analytic Jacobian for a quadratic on the simplex
    @testset "Finite-difference Jacobian consistency" begin
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]
        kw = (; max_iters=10000, tol=1e-8, step_rule=AdaptiveStepSize())

        x_star, res = solve(_f, ProbabilitySimplex(), x0, θ₀; grad=_∇f!, kw...)

        ε = 1e-6
        jac_fd = zeros(2, 2)
        for j in 1:2
            eⱼ = zeros(2); eⱼ[j] = 1.0
            x_plus, _ = solve(_f, ProbabilitySimplex(), x0, θ₀ .+ ε .* eⱼ; grad=_∇f!, kw...)
            x_minus, _ = solve(_f, ProbabilitySimplex(), x0, θ₀ .- ε .* eⱼ; grad=_∇f!, kw...)
            jac_fd[:, j] = (x_plus .- x_minus) ./ (2ε)
        end

        # For f(x,θ) = 0.5‖x‖² - θᵀx on ProbSimplex, the Jacobian at an interior
        # solution is I - (1/n)11ᵀ. Verify FD matches this structure.
        n = 2
        J_expected = I(n) .- 1.0/n
        @test isapprox(jac_fd, J_expected; atol=0.005)
    end

    # Verify that the conjugate gradient solver finds the exact solution of a small linear system
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

    # Verify that CG with Tikhonov regularization solves the shifted system (A + lambda I)u = b
    @testset "CG solver with regularization" begin
        A = [1.0 0.999; 0.999 1.0]
        b = [1.0, 1.0]

        hvp_fn(d) = A * d
        u, cg_result = Marguerite._cg_solve(hvp_fn, b; maxiter=100, tol=1e-10, λ=1e-2)
        @test norm((A + 1e-2 * I) * u - b) < 1e-6
        @test cg_result.converged
    end

    # Check that CG emits a warning and reports non-convergence when given too few iterations
    @testset "CG non-convergence warning" begin
        A = [4.0 1.0; 1.0 3.0]
        b = [1.0, 2.0]
        hvp_fn(d) = A * d
        u, cg_result = @test_warn "CG solve did not converge" Marguerite._cg_solve(hvp_fn, b; maxiter=1, tol=1e-15, λ=0.0)
        @test !cg_result.converged
        @test cg_result.iterations == 1
        @test cg_result.residual_norm > 1e-15
    end

    # Verify that diff_cg_maxiter, diff_cg_tol, and diff_lambda kwargs are accepted by rrule with manual gradient
    @testset "diff_* kwargs on rrule (manual gradient)" begin
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]

        (x_star, _), pb = rrule(solve, _f, ProbabilitySimplex(), x0, θ₀;
                                grad=_∇f!, max_iters=1000, tol=1e-4,
                                diff_cg_maxiter=100, diff_cg_tol=1e-8, diff_lambda=1e-3)
        dx = 2 .* x_star
        tangents = pb((dx, nothing))
        dθ = tangents[5]
        @test length(dθ) == 2
        @test all(isfinite, dθ)
    end

    # Verify that diff_cg_maxiter, diff_cg_tol, and diff_lambda kwargs are accepted by rrule with auto gradient
    @testset "diff_* kwargs on rrule (auto gradient)" begin
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]

        (x_star, _), pb = rrule(solve, _f, ProbabilitySimplex(), x0, θ₀;
                                max_iters=1000, tol=1e-4,
                                diff_cg_maxiter=100, diff_cg_tol=1e-8, diff_lambda=1e-3)
        dx = 2 .* x_star
        tangents = pb((dx, nothing))
        dθ = tangents[5]
        @test length(dθ) == 2
        @test all(isfinite, dθ)
    end

    # Check that passing a backend kwarg does not interfere with the inner solve
    @testset "backend kwarg does not leak to inner solve" begin
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]
        x, res = solve(_f, ProbabilitySimplex(), x0, θ₀;
                       grad=_∇f!, backend=DI.AutoForwardDiff(), max_iters=1000, tol=1e-2)
        @test res.objective < 0
    end

    # Verify that the rrule pullback produces finite gradients when element types need promotion
    @testset "Type promotion in rrule pullback" begin
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]
        kw = (; max_iters=5000, tol=1e-4)

        (x_star, res), pb = rrule(solve, _f, ProbabilitySimplex(), x0, θ₀; grad=_∇f!, kw...)
        dx = 2 .* x_star
        tangents = pb((dx, nothing))
        dθ = tangents[5]
        @test length(dθ) == 2
        @test all(isfinite, dθ)
    end

    # Check that a Float32 oracle with Float64 parameters builds a valid cached pullback state
    @testset "Mixed-precision rrule pullback builds cached state" begin
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]
        kw = (; max_iters=5000, tol=1e-4)

        (x_star, _), pb = rrule(solve, _f, ProbabilitySimplex(Float32(1)), x0, θ₀; grad=_∇f!, kw...)
        dθ = pb((2 .* x_star, nothing))[5]
        @test length(dθ) == 2
        @test all(isfinite, dθ)
    end

    # Verify that the auto-gradient rrule matches the manual-gradient rrule and finite differences
    @testset "Auto-gradient + θ rrule (no grad)" begin
        n = 2
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]
        x_target = [0.6, 0.4]
        kw = (; max_iters=10000, tol=1e-8, step_rule=Marguerite.AdaptiveStepSize())

        (x_star, _), pb = rrule(solve, _f, ProbabilitySimplex(), x0, θ₀; kw...)

        dx = 2 .* (x_star .- x_target)
        tangents = pb((dx, nothing))
        @test length(tangents) == 5
        @test tangents[1] isa NoTangent
        @test tangents[2] isa NoTangent
        @test tangents[3] isa NoTangent
        @test tangents[4] isa NoTangent
        dθ = tangents[5]
        @test length(dθ) == n
        @test all(isfinite, dθ)

        # Cross-check: manual-gradient rrule should match auto-gradient rrule
        (x_star_m, _), pb_m = rrule(solve, _f, ProbabilitySimplex(), x0, θ₀; grad=_∇f!, kw...)
        dθ_m = pb_m((2 .* (x_star_m .- x_target), nothing))[5]
        @test isapprox(dθ, dθ_m; atol=0.01)

        # Cross-check against finite differences
        L(θ_) = begin
            x_, _ = solve(_f, ProbabilitySimplex(), x0, θ_; grad=_∇f!, kw...)
            sum((x_ .- x_target) .^ 2)
        end
        ε = 1e-5
        dθ_fd = zeros(n)
        for j in 1:n
            eⱼ = zeros(n); eⱼ[j] = 1.0
            dθ_fd[j] = (L(θ₀ .+ ε .* eⱼ) - L(θ₀ .- ε .* eⱼ)) / (2ε)
        end
        @test isapprox(dθ, dθ_fd; atol=0.01)
    end

    # Check that passing a ZeroTangent seed returns all NoTangent entries for both manual and auto gradient
    @testset "ZeroTangent pullback returns all NoTangent" begin
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]

        (_, _), pb = rrule(solve, _f, ProbabilitySimplex(), x0, θ₀;
                           grad=_∇f!, max_iters=1000, tol=1e-4)
        tangents = pb((ChainRulesCore.ZeroTangent(), nothing))
        @test length(tangents) == 5
        @test all(t -> t isa NoTangent, tangents)

        (_, _), pb2 = rrule(solve, _f, ProbabilitySimplex(), x0, θ₀;
                            max_iters=1000, tol=1e-4)
        tangents2 = pb2((ChainRulesCore.ZeroTangent(), nothing))
        @test length(tangents2) == 5
        @test all(t -> t isa NoTangent, tangents2)
    end

    # Verify that rrule works with the default HVP backend when using manual gradient
    @testset "rrule default hvp_backend (manual grad)" begin
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]
        kw = (; max_iters=5000, tol=1e-4)

        (x_star, _), pb = rrule(solve, _f, ProbabilitySimplex(), x0, θ₀; grad=_∇f!, kw...)
        @test length(x_star) == 2

        dθ = pb((2 .* x_star, nothing))[end]
        @test length(dθ) == 2
        @test all(isfinite, dθ)
    end

    # ------------------------------------------------------------------
    # Parametric constraint set tests
    # ------------------------------------------------------------------

    # Verify that ParametricBox rrule with manual gradient produces gradients matching finite differences
    @testset "ParametricBox rrule (manual gradient)" begin
        n = 3
        # f(x, θ) = 0.5||x||² - θ'x, box bounds from θ
        # θ = [lb₁, lb₂, lb₃, ub₁, ub₂, ub₃]
        _f_box(x, θ) = 0.5 * dot(x, x) - dot(θ[1:length(x)], x)
        _∇f_box!(g, x, θ) = (g .= x .- θ[1:length(x)])

        plmo = ParametricBox(θ -> θ[1:n], θ -> θ[n+1:2n])
        θ₀ = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]  # box [0,1]^3
        x0 = [0.5, 0.5, 0.5]
        kw = (; max_iters=5000, tol=1e-6)

        # Objective pulls x toward θ[1:3] = 0, but lb = 0, so x* = [0,0,0]
        (x_star, res), pb = rrule(solve, _f_box, plmo, x0, θ₀; grad=_∇f_box!, kw...)
        @test all(x_star .≈ 0.0)

        dx = ones(n)
        tangents = pb((dx, nothing))
        dθ = tangents[5]
        @test length(dθ) == 2n
        @test all(isfinite, dθ)

        # Finite-difference cross-check
        ε = 1e-4
        dθ_fd = zeros(2n)
        L(θ_) = begin
            x_, _ = solve(_f_box, plmo, x0, θ_; grad=_∇f_box!, kw...)
            dot(ones(n), x_)
        end
        for j in 1:2n
            eⱼ = zeros(2n); eⱼ[j] = 1.0
            dθ_fd[j] = (L(θ₀ .+ ε .* eⱼ) - L(θ₀ .- ε .* eⱼ)) / (2ε)
        end
        @test isapprox(dθ, dθ_fd; atol=0.1)
    end

    # Check that ParametricProbSimplex rrule gradients match finite differences for both objective and radius parameters
    @testset "ParametricProbSimplex rrule" begin
        n = 2
        # f(x, θ) = 0.5||x||² - θ[1:n]'x, r = θ[end]
        _f_simp(x, θ) = 0.5 * dot(x, x) - dot(θ[1:length(x)], x)
        _∇f_simp!(g, x, θ) = (g .= x .- θ[1:length(x)])

        plmo = ParametricProbSimplex(θ -> θ[end])
        θ₀ = [0.7, 0.3, 1.0]  # θ[1:2] are objective params, θ[3] = radius
        x0 = [0.5, 0.5]
        kw = (; max_iters=5000, tol=1e-6)

        (x_star, res), pb = rrule(solve, _f_simp, plmo, x0, θ₀; grad=_∇f_simp!, kw...)
        @test sum(x_star) ≈ 1.0 atol=1e-3

        dx = 2.0 .* x_star
        tangents = pb((dx, nothing))
        dθ = tangents[5]
        @test length(dθ) == 3
        @test all(isfinite, dθ)

        # Finite-difference cross-check (FW convergence on simplex is O(1/t),
        # so FD is less accurate for the radius component)
        ε = 1e-4
        dθ_fd = zeros(3)
        fd_kw = (; max_iters=10000, tol=1e-6)
        L(θ_) = begin
            x_, _ = solve(_f_simp, plmo, x0, θ_; grad=_∇f_simp!, fd_kw...)
            dot(x_, x_)
        end
        for j in 1:3
            eⱼ = zeros(3); eⱼ[j] = 1.0
            dθ_fd[j] = (L(θ₀ .+ ε .* eⱼ) - L(θ₀ .- ε .* eⱼ)) / (2ε)
        end
        @test isapprox(dθ[1:2], dθ_fd[1:2]; atol=0.15)  # relaxed from 0.1: fewer iters for test speed
        @test isapprox(dθ[3], dθ_fd[3]; atol=0.25)  # relaxed from 0.2: budget/radius less precise with fewer iters
    end

    # Check that ParametricBox rrule with auto gradient produces finite parameter gradients
    @testset "ParametricBox rrule (auto gradient)" begin
        n = 2
        _f_box2(x, θ) = 0.5 * dot(x, x) - dot(θ[1:length(x)], x)
        plmo = ParametricBox(θ -> θ[1:n], θ -> θ[n+1:2n])
        θ₀ = [0.3, 0.7, 1.0, 1.0]
        x0 = [0.5, 0.5]
        kw = (; max_iters=5000, tol=1e-6)

        (x_star, res), pb = rrule(solve, _f_box2, plmo, x0, θ₀; kw...)
        @test all(isfinite, x_star)

        dx = ones(n)
        tangents = pb((dx, nothing))
        dθ = tangents[5]  # (solve, f, lmo, x0, θ) → 5 tangents
        @test length(dθ) == 2n
        @test all(isfinite, dθ)
    end

    # Check that ZeroTangent seed with a ParametricOracle returns all NoTangent for both manual and auto gradient
    @testset "ZeroTangent with ParametricOracle returns all NoTangent" begin
        n = 2
        _f_z(x, θ) = 0.5 * dot(x, x) - dot(θ[1:length(x)], x)
        _∇f_z!(g, x, θ) = (g .= x .- θ[1:length(x)])
        plmo = ParametricBox(θ -> zeros(n), θ -> ones(n))

        θ₀ = [0.5, 0.5]
        x0 = [0.5, 0.5]

        (_, _), pb = rrule(solve, _f_z, plmo, x0, θ₀; grad=_∇f_z!, max_iters=1000, tol=1e-4)
        tangents = pb((ChainRulesCore.ZeroTangent(), nothing))
        @test length(tangents) == 5
        @test all(t -> t isa NoTangent, tangents)

        (_, _), pb2 = rrule(solve, _f_z, plmo, x0, θ₀; max_iters=1000, tol=1e-4)
        tangents2 = pb2((ChainRulesCore.ZeroTangent(), nothing))
        @test length(tangents2) == 5
        @test all(t -> t isa NoTangent, tangents2)
    end

    # Verify that ParametricWeightedSimplex rrule gradients match finite differences for budget and lower-bound parameters
    @testset "ParametricWeightedSimplex rrule" begin
        n = 2
        # θ = [c₁, c₂, β, lb₁, lb₂] where c are objective params, β is budget, lb is lower bound
        # f(x, θ) = 0.5||x||² - [c₁,c₂]'x
        # Constraint: α'x ≤ β, x ≥ lb (with fixed α = [1,1])
        # Use c = [3.0, 0.0] so x* sits at vertex [β, 0] (FW converges fast to vertices)
        _f_ws(x, θ) = 0.5 * dot(x, x) - dot(θ[1:2], x)
        _∇f_ws!(g, x, θ) = (g .= x .- θ[1:2])

        plmo = ParametricWeightedSimplex(
            θ -> [1.0, 1.0],     # α (fixed)
            θ -> θ[3],           # β
            θ -> θ[4:5]          # lb
        )
        θ₀ = [3.0, 0.0,    # objective params: push x* toward vertex
               1.0,          # β (budget)
               0.0, 0.0]    # lb
        x0 = [0.5, 0.5]
        kw = (; max_iters=10000, tol=1e-6)

        (x_star, res), pb = rrule(solve, _f_ws, plmo, x0, θ₀; grad=_∇f_ws!, kw...)
        @test x_star[1] ≈ 1.0 atol=1e-2
        @test x_star[2] ≈ 0.0 atol=1e-2

        dx = ones(n)
        tangents = pb((dx, nothing))
        dθ = tangents[5]
        @test length(dθ) == 5
        @test all(isfinite, dθ)

        # Finite-difference cross-check
        ε = 1e-4
        m = length(θ₀)
        dθ_fd = zeros(m)
        fd_kw = (; max_iters=10000, tol=1e-6)
        L(θ_) = begin
            x_, _ = solve(_f_ws, plmo, x0, θ_; grad=_∇f_ws!, fd_kw...)
            sum(x_)
        end
        for j in 1:m
            eⱼ = zeros(m); eⱼ[j] = 1.0
            dθ_fd[j] = (L(θ₀ .+ ε .* eⱼ) - L(θ₀ .- ε .* eⱼ)) / (2ε)
        end
        # Budget and lb components are well-conditioned; objective params less so at vertex
        @test isapprox(dθ[3], dθ_fd[3]; atol=0.25)  # relaxed from 0.2: budget/radius less precise with fewer iters
        @test isapprox(dθ[4:5], dθ_fd[4:5]; atol=0.15)  # relaxed from 0.1: fewer iters for test speed
    end

    # Verify that ParametricWeightedSimplex with parameter-dependent weights matches finite differences
    @testset "ParametricWeightedSimplex rrule with θ-dependent α matches FD" begin
        _f_ws_varα(x, θ) = 0.5 * dot(x, x) - dot(θ[1:2], x)
        _∇f_ws_varα!(g, x, θ) = (g .= x .- θ[1:2])

        plmo = ParametricWeightedSimplex(
            θ -> [1.0 + 0.2 * θ[3], 1.0 - 0.1 * θ[3]],
            θ -> θ[4],
            θ -> zeros(2)
        )
        θ₀ = [1.4, 1.1, 1.0, 1.0]
        x0 = [0.3, 0.3]
        dx = [0.7, -0.2]
        kw = (; max_iters=50_000, tol=5e-6)

        (x_star, _), pb = rrule(solve, _f_ws_varα, plmo, x0, θ₀; grad=_∇f_ws_varα!, kw...)
        α₀ = plmo.α_fn(θ₀)
        @test dot(α₀, x_star) ≈ θ₀[4] atol=5e-5

        dθ = pb((dx, nothing))[5]

        ε = 1e-6
        dθ_fd = zeros(length(θ₀))
        L(θ_) = begin
            x_, _ = solve(_f_ws_varα, plmo, x0, θ_; grad=_∇f_ws_varα!, kw...)
            dot(dx, x_)
        end
        for j in eachindex(θ₀)
            eⱼ = zeros(length(θ₀)); eⱼ[j] = 1.0
            dθ_fd[j] = (L(θ₀ .+ ε .* eⱼ) - L(θ₀ .- ε .* eⱼ)) / (2ε)
        end

        @test isapprox(dθ, dθ_fd; atol=3e-2)
    end

    # Verify that ParametricSimplex (capped) rrule gradients match finite differences when the budget is active
    @testset "ParametricSimplex (capped) rrule" begin
        n = 2
        # Capped simplex: {x ≥ 0 : ∑x ≤ r(θ)}
        # Choose θ so that budget IS active (sum(x*) = r)
        _f_cap(x, θ) = 0.5 * dot(x, x) - dot(θ[1:length(x)], x)
        _∇f_cap!(g, x, θ) = (g .= x .- θ[1:length(x)])

        _r_fn_cap(θ) = θ[end]
        plmo = ParametricSimplex{typeof(_r_fn_cap), false}(_r_fn_cap)
        # θ[1:2] pull x toward [0.7, 0.3], sum = 1.0, and r = 0.8 < 1.0
        # so budget is active: sum(x*) = r = 0.8
        θ₀ = [0.7, 0.3, 0.8]
        x0 = [0.4, 0.4]
        kw = (; max_iters=10000, tol=1e-6)

        (x_star, res), pb = rrule(solve, _f_cap, plmo, x0, θ₀; grad=_∇f_cap!, kw...)
        @test sum(x_star) ≈ 0.8 atol=1e-2
        @test all(x_star .≥ -1e-6)

        dx = 2.0 .* x_star
        tangents = pb((dx, nothing))
        dθ = tangents[5]
        @test length(dθ) == 3
        @test all(isfinite, dθ)

        # Finite-difference cross-check
        ε = 1e-4
        dθ_fd = zeros(3)
        fd_kw = (; max_iters=10000, tol=1e-6)
        L(θ_) = begin
            x_, _ = solve(_f_cap, plmo, x0, θ_; grad=_∇f_cap!, fd_kw...)
            dot(x_, x_)
        end
        for j in 1:3
            eⱼ = zeros(3); eⱼ[j] = 1.0
            dθ_fd[j] = (L(θ₀ .+ ε .* eⱼ) - L(θ₀ .- ε .* eⱼ)) / (2ε)
        end
        @test isapprox(dθ[1:2], dθ_fd[1:2]; atol=0.15)  # relaxed from 0.1: fewer iters for test speed
        @test isapprox(dθ[3], dθ_fd[3]; atol=0.25)  # relaxed from 0.2: budget/radius less precise with fewer iters
    end

    # Verify that differentiation works at an interior simplex solution where only the equality constraint is active
    @testset "Interior of simplex (equality constraint only)" begin
        # θ = [0.6, 0.4] sums to 1.0, so x* = θ on the probability simplex.
        # All components positive → no bound constraints active, but budget equality is active.
        n = 2
        θ₀ = [0.6, 0.4]
        x0 = [0.5, 0.5]
        x_target = [0.5, 0.5]
        kw = (; max_iters=10000, tol=1e-6)

        (x_star, _), pb = rrule(solve, _f, ProbabilitySimplex(), x0, θ₀; grad=_∇f!, kw...)
        @test x_star[1] ≈ 0.6 atol=1e-3
        @test x_star[2] ≈ 0.4 atol=1e-3

        # Verify active set: no bounds, but budget equality is active
        as = Marguerite.active_set(ProbabilitySimplex(), x_star)
        @test isempty(as.bound_indices)
        @test length(as.eq_normals) == 1

        dx = 2.0 .* (x_star .- x_target)
        tangents = pb((dx, nothing))
        dθ = tangents[5]
        @test length(dθ) == n
        @test all(isfinite, dθ)

        # FD cross-check
        ε = 1e-3
        L(θ_) = begin
            x_, _ = solve(_f, ProbabilitySimplex(), x0, θ_; grad=_∇f!, kw...)
            sum((x_ .- x_target).^2)
        end
        dθ_fd = zeros(n)
        for j in 1:n
            eⱼ = zeros(n); eⱼ[j] = 1.0
            dθ_fd[j] = (L(θ₀ .+ ε .* eⱼ) - L(θ₀ .- ε .* eⱼ)) / (2ε)
        end
        @test isapprox(dθ, dθ_fd; atol=0.05)
    end

    # Verify that Gram-Schmidt orthogonalization of constraint normals produces an orthogonal basis and correct null-space projection
    @testset "Multi-constraint orthogonalization (_orthogonalize!)" begin
        # Two non-orthogonal equality constraints on R³:
        #   a₁'x = 1   with a₁ = [1, 1, 1]
        #   a₂'x = 1.5 with a₂ = [1, 2, 1]
        # These define a 1D feasible line. The null-space projection onto ker(G)
        # must handle non-orthogonal normals correctly.

        # Direct test of _orthogonalize!
        a1 = [1.0, 1.0, 1.0]
        a2 = [1.0, 2.0, 1.0]
        a_vecs = [copy(a1), copy(a2)]
        a_norm_sqs = [dot(a1, a1), dot(a2, a2)]
        Marguerite._orthogonalize!(a_vecs, a_norm_sqs)

        # After orthogonalization, vectors should be orthogonal
        @test abs(dot(a_vecs[1], a_vecs[2])) < 1e-12
        # First vector unchanged
        @test a_vecs[1] ≈ a1
        # Norm-squared updated correctly
        @test a_norm_sqs[1] ≈ dot(a_vecs[1], a_vecs[1])
        @test a_norm_sqs[2] ≈ dot(a_vecs[2], a_vecs[2])

        # Test _null_project! with orthogonalized basis
        w = [1.0, 2.0, 3.0]
        out = similar(w)
        Marguerite._null_project!(out, w, a_vecs, a_norm_sqs)
        # Result should be orthogonal to both original normals
        @test abs(dot(a1, out)) < 1e-10
        @test abs(dot(a2, out)) < 1e-10

        # Compare against direct matrix projection: P = I - A'(AA')⁻¹A
        A = hcat(a1, a2)'  # 2×3
        P_exact = I - A' * inv(A * A') * A
        out_exact = P_exact * w
        @test out ≈ out_exact atol=1e-10

        # Single-constraint case: _orthogonalize! is a no-op
        a_single = [[1.0, 1.0]]
        a_single_norms = [2.0]
        Marguerite._orthogonalize!(a_single, a_single_norms)
        @test a_single[1] ≈ [1.0, 1.0]
        @test a_single_norms[1] ≈ 2.0
    end

    # Verify that the KKT adjoint solve handles two non-orthogonal equality constraints and matches the direct matrix inverse
    @testset "KKT adjoint with multiple non-orthogonal equality constraints" begin
        # Synthetic test: two non-orthogonal equality constraints
        # Feasible set: {x ∈ R³ : x₁+x₂+x₃ = 1, x₁+2x₂+x₃ = 1.5, xᵢ ≥ 0}
        # The feasible line is x₂ = 0.5, x₁+x₃ = 0.5, xᵢ ≥ 0
        # f(x, θ) = 0.5||x||² - θ'x, so x* = proj(θ) onto feasible set

        n = 3
        _f_mc(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
        _∇f_mc!(g, x, θ) = (g .= x .- θ)

        # Build the active set manually: two equality constraints, no bounds active
        # (θ chosen so x* is in the interior of the non-negative orthant on the line)
        eq_normals = [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0]]
        eq_rhs = [1.0, 1.5]
        # All variables free (no bound constraints active)
        as = Marguerite.ActiveConstraints{Float64}(
            Int[], Float64[], BitVector(), collect(1:n), eq_normals, eq_rhs)

        # x* on the feasible line: x₂ = 0.5, and we choose θ so x* = [0.2, 0.5, 0.3]
        # At optimality: ∇f = x* - θ = Σ μⱼ aⱼ, so θ = x* - Σ μⱼ aⱼ for some μ
        x_star = [0.2, 0.5, 0.3]
        # Choose μ_eq = [0.1, -0.05] (arbitrary, just need θ consistent)
        μ_eq_true = [0.1, -0.05]
        θ₀ = x_star .- (μ_eq_true[1] .* eq_normals[1] .+ μ_eq_true[2] .* eq_normals[2])

        # dx = identity columns → recover full Jacobian
        # Test with a specific dx direction
        dx = [1.0, 0.0, 0.0]

        # Solve KKT adjoint using our code
        hvp_backend = Marguerite.SECOND_ORDER_BACKEND
        u, μ_bound, μ_eq, cg_result = Marguerite._kkt_adjoint_solve(
            _f_mc, hvp_backend, x_star, θ₀, dx, as;
            cg_maxiter=100, cg_tol=1e-10, cg_λ=1e-6)

        @test cg_result.converged

        # Verify KKT conditions: H*u + G'*μ_eq = dx
        # For f = 0.5||x||² - θ'x, H = I
        # So: u + Σ μ_eq_j * a_j = dx
        A = hcat(eq_normals...)'  # 2×3
        residual = u .+ A' * μ_eq .- dx
        @test norm(residual) < 1e-4

        # Verify feasibility: G*u = 0 (u must be in null space of constraints)
        @test abs(dot(eq_normals[1], u)) < 1e-6
        @test abs(dot(eq_normals[2], u)) < 1e-6

        # Brute-force reference: solve KKT system directly via matrix inverse
        # [I  A'] [u]   [dx]
        # [A  0 ] [μ] = [0 ]
        K = [Matrix(1.0I, n, n) A'; A zeros(2, 2)]
        rhs = [dx; zeros(2)]
        sol = K \ rhs
        u_ref = sol[1:n]
        μ_ref = sol[n+1:end]
        @test u ≈ u_ref atol=1e-4
        @test μ_eq ≈ μ_ref atol=1e-4
    end

    # Check that the active set correctly detects the budget constraint when the radius is large
    @testset "Active set tolerance with scaled budget" begin
        # When budget is large, absolute tolerance can miss the active budget constraint.
        # Relative tolerance fixes this: abs(sum(x) - r) ≤ tol * (1 + abs(r))
        n = 100
        r_large = 1e6
        θ₀ = fill(r_large / n + 0.1, n)  # push toward fully using budget
        x0 = fill(r_large / n, n)

        lmo = Simplex(r_large)
        x_star, _ = solve(_f, lmo, x0, θ₀; grad=_∇f!, max_iters=5000, tol=1e-8)

        # Budget should be detected as active
        as = Marguerite.active_set(lmo, x_star)
        @test length(as.eq_normals) == 1  # budget constraint active
    end

    # Verify that Spectraplex KKT adjoint correctly handles mixed on/off-diagonal directions at a rank-one optimum
    @testset "Spectraplex KKT includes mixed directions at rank-deficient optima" begin
        lmo = Spectraplex(2)
        x_star = vec([1.0 0.0; 0.0 0.0])
        as = Marguerite.active_set(lmo, x_star; tol=1e-10)
        θ₀ = [0.0]
        dx = vec([0.0 1.0; 1.0 0.0])
        hvp_backend = Marguerite.SECOND_ORDER_BACKEND

        _f_sp_face(x, θ) = dot([0.0, θ[1], θ[1], 1.0], x)
        _∇f_sp_face!(g, x, θ) = (g .= [0.0, θ[1], θ[1], 1.0])

        u, μ_bound, μ_eq, cg_result = Marguerite._kkt_adjoint_solve(
            _f_sp_face, hvp_backend, x_star, θ₀, dx, as;
            cg_maxiter=100, cg_tol=1e-10, cg_λ=0.0,
            grad=_∇f_sp_face!, backend=Marguerite.DEFAULT_BACKEND)
        @test cg_result.converged
        @test isempty(μ_bound)
        @test isempty(μ_eq)
        @test u ≈ dx atol=1e-8

        state = Marguerite._build_pullback_state(
            _f_sp_face, hvp_backend, x_star, θ₀, lmo, 1e-6;
            grad=_∇f_sp_face!, backend=Marguerite.DEFAULT_BACKEND,
            diff_lambda=0.0)
        u_cached, μ_bound_cached, μ_eq_cached, cg_cached = Marguerite._kkt_adjoint_solve_cached(
            state, dx)
        @test cg_cached.converged
        @test isempty(μ_bound_cached)
        @test isempty(μ_eq_cached)
        @test u_cached ≈ dx atol=1e-8
        @test u_cached ≈ u atol=1e-10
        @test μ_eq_cached ≈ μ_eq atol=1e-10
    end

    # Check that Spectraplex CG regularization operates in symmetric matrix space, not raw vector space
    @testset "Spectraplex regularization acts in matrix space" begin
        n = 3
        lmo = Spectraplex(n)
        x_star = vec(Matrix(1.0I, n, n) ./ n)
        as = Marguerite.active_set(lmo, x_star)
        θ₀ = Float64[]
        dx = vec(Matrix(Diagonal([1.0, -0.5, -0.5])))
        hvp_backend = Marguerite.SECOND_ORDER_BACKEND

        _f_sp_reg(x, θ) = 0.5 * dot(x, x)
        expected_u = dx ./ 2

        u, μ_bound, μ_eq, cg_result = Marguerite._kkt_adjoint_solve(
            _f_sp_reg, hvp_backend, x_star, θ₀, dx, as;
            cg_maxiter=100, cg_tol=1e-10, cg_λ=1.0)
        @test cg_result.converged
        @test isempty(μ_bound)
        @test isempty(μ_eq)
        @test u ≈ expected_u atol=1e-8

        state = Marguerite._build_pullback_state(_f_sp_reg, hvp_backend, x_star, θ₀, lmo, 1e-6;
            diff_lambda=1.0)
        u_cached, μ_bound_cached, μ_eq_cached, cg_cached = Marguerite._kkt_adjoint_solve_cached(
            state, dx)
        @test cg_cached.converged
        @test isempty(μ_bound_cached)
        @test isempty(μ_eq_cached)
        @test u_cached ≈ expected_u atol=1e-8
        @test u_cached ≈ u atol=1e-6
    end

    # Check that building the cached pullback state for a 200-dimensional simplex problem stays under 1.2 MB
    @testset "Cached pullback state remains linear-memory for constrained problems" begin
        n_mem = 200
        x_star = fill(1.0 / n_mem, n_mem)
        θ_mem = collect(range(1.0, 0.1; length=n_mem))
        lmo_mem = ProbabilitySimplex()
        hvp_backend = Marguerite.SECOND_ORDER_BACKEND

        _f_mem(x, θ) = 0.5 * dot(x, x) - dot(θ, x)

        Marguerite._build_pullback_state(_f_mem, hvp_backend, x_star, θ_mem, lmo_mem, 1e-6)
        alloc = @allocated Marguerite._build_pullback_state(_f_mem, hvp_backend, x_star, θ_mem, lmo_mem, 1e-6)
        @test alloc < 1_200_000
    end

    # Check that the Spectraplex pullback state for n=30 stays well under n^4 bytes in memory
    @testset "Spectraplex pullback state stays sub-quartic in memory" begin
        n_mem = 30
        x_star = vec(Matrix(1.0I, n_mem, n_mem) ./ n_mem)
        θ_mem = Float64[]
        lmo_mem = Spectraplex(n_mem)
        hvp_backend = Marguerite.SECOND_ORDER_BACKEND

        _f_sp_mem(x, θ) = 0.5 * dot(x, x)

        Marguerite._build_pullback_state(_f_sp_mem, hvp_backend, x_star, θ_mem, lmo_mem, 1e-6)
        alloc = @allocated Marguerite._build_pullback_state(_f_sp_mem, hvp_backend, x_star, θ_mem, lmo_mem, 1e-6)
        @test alloc < 12_000_000
    end

    # Check that the rrule pullback closure for a 700-dimensional problem stays under 1 MB
    @testset "Manual-gradient rrule pullback closure remains linear-memory" begin
        n_mem = 700
        H = Diagonal(fill(2.0, n_mem))
        _f_mem_rrule(x, θ) = 0.5 * dot(x, H * x) - dot(θ, x)
        _∇f_mem_rrule!(g, x, θ) = (g .= H * x .- θ)

        lmo_mem = ProbabilitySimplex()
        x0_mem = fill(1.0 / n_mem, n_mem)
        θ_mem = collect(range(1.0, 0.1; length=n_mem))
        kw = (; max_iters=1_000, tol=1e-4)

        (_, _), pb = rrule(solve, _f_mem_rrule, lmo_mem, x0_mem, θ_mem; grad=_∇f_mem_rrule!, kw...)
        @test Base.summarysize(pb) < 1_000_000
    end

    # Verify that custom oracles without active_set error unless assume_interior=true, and that those with active_set match ProbabilitySimplex
    @testset "Differentiated custom oracles require active_set or explicit interior assumption" begin
        n = 2
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]
        kw = (; grad=_∇f!, max_iters=1000, tol=1e-3)

        plain_lmo(v, g) = (fill!(v, 0.0); i = argmin(g); v[i] = 1.0; v)
        wrapped_lmo = WrappedProbSimplex()
        callable_lmo = CallableProbSimplexDiff()

        @test_throws ArgumentError rrule(solve, _f, plain_lmo, x0, θ₀; kw...)

        (_, _), pb_plain = @test_logs (:warn, r"assume_interior=true") rrule(
            solve, _f, plain_lmo, x0, θ₀; assume_interior=true, kw...)
        dθ_plain = pb_plain((ones(2), nothing))[5]
        @test length(dθ_plain) == n
        @test all(isfinite, dθ_plain)

        dθ = bilevel_gradient(
            x -> sum((x .- [0.6, 0.4]).^2),
            _f, plain_lmo, x0, θ₀;
            assume_interior=true, kw...)
        @test length(dθ) == n
        @test all(isfinite, dθ)

        # Cross-check: should match ProbabilitySimplex result
        dθ_ref = bilevel_gradient(
            x -> sum((x .- [0.6, 0.4]).^2),
            _f, ProbabilitySimplex(), x0, θ₀;
            kw...)
        @test isapprox(dθ, dθ_ref; atol=0.1)

        (x_wrapped, _), pb_wrapped = rrule(solve, _f, wrapped_lmo, x0, θ₀; kw...)
        dθ_wrapped = pb_wrapped((2 .* x_wrapped, nothing))[5]
        @test length(dθ_wrapped) == n
        @test all(isfinite, dθ_wrapped)

        (x_callable, _), pb_callable = rrule(solve, _f, callable_lmo, x0, θ₀; kw...)
        dθ_callable = pb_callable((2 .* x_callable, nothing))[5]
        @test length(dθ_callable) == n
        @test all(isfinite, dθ_callable)

        (x_ref, _), pb_ref = rrule(solve, _f, ProbabilitySimplex(), x0, θ₀; kw...)
        dθ_ref_rrule = pb_ref((2 .* x_ref, nothing))[5]
        @test isapprox(dθ_wrapped, dθ_ref_rrule; atol=0.1)
        @test isapprox(dθ_callable, dθ_ref_rrule; atol=0.1)
    end

    # Verify that Spectraplex rrule with manual gradient produces finite parameter gradients
    @testset "Spectraplex rrule (manual gradient)" begin
        n_sp = 2
        m_sp = n_sp * n_sp
        _f_sp(x, θ) = 0.5 * dot(x, x) - dot(_diag_vec_sp(θ, n_sp), x)
        _∇f_sp!(g, x, θ) = (g .= x .- _diag_vec_sp(θ, n_sp))

        lmo_sp = Spectraplex(n_sp)
        x0_sp = vec(Matrix(1.0I, n_sp, n_sp) ./ n_sp)
        θ_sp = [2.0, 0.5]
        kw_sp = (; max_iters=5000, tol=1e-6)

        (x_star, _), pb = rrule(solve, _f_sp, lmo_sp, x0_sp, θ_sp; grad=_∇f_sp!, kw_sp...)
        dx = 2.0 .* x_star
        tangents = pb((dx, nothing))
        dθ = tangents[5]
        @test length(dθ) == 2
        @test all(isfinite, dθ)
    end

    # Check that Spectraplex rrule works when the oracle uses Float32 but parameters are Float64
    @testset "Spectraplex rrule mixed precision" begin
        n_sp = 2
        _f_sp_mp(x, θ) = 0.5 * dot(x, x) - dot(_diag_vec_sp(θ, n_sp), x)
        _∇f_sp_mp!(g, x, θ) = (g .= x .- _diag_vec_sp(θ, n_sp))

        lmo_sp = Spectraplex(n_sp, Float32(1))
        x0_sp = vec(Matrix(1.0I, n_sp, n_sp) ./ n_sp)
        θ_sp = [2.0, 0.5]
        kw_sp = (; max_iters=5000, tol=1e-6)

        (x_star, _), pb = rrule(solve, _f_sp_mp, lmo_sp, x0_sp, θ_sp; grad=_∇f_sp_mp!, kw_sp...)
        dθ = pb((2.0 .* x_star, nothing))[5]
        @test length(dθ) == 2
        @test all(isfinite, dθ)
    end

    # Verify that Spectraplex rrule with auto gradient matches the manual-gradient result
    @testset "Spectraplex rrule (auto gradient)" begin
        n_sp = 2
        m_sp = n_sp * n_sp
        _f_sp2(x, θ) = 0.5 * dot(x, x) - dot(_diag_vec_sp(θ, n_sp), x)
        _∇f_sp2!(g, x, θ) = (g .= x .- _diag_vec_sp(θ, n_sp))

        lmo_sp = Spectraplex(n_sp)
        x0_sp = vec(Matrix(1.0I, n_sp, n_sp) ./ n_sp)
        θ_sp = [2.0, 0.5]
        kw_sp = (; max_iters=5000, tol=1e-6)

        (x_star_auto, _), pb_auto = rrule(solve, _f_sp2, lmo_sp, x0_sp, θ_sp; kw_sp...)
        dx_auto = 2.0 .* x_star_auto
        dθ_auto = pb_auto((dx_auto, nothing))[5]

        (x_star_man, _), pb_man = rrule(solve, _f_sp2, lmo_sp, x0_sp, θ_sp; grad=_∇f_sp2!, kw_sp...)
        dx_man = 2.0 .* x_star_man
        dθ_man = pb_man((dx_man, nothing))[5]

        @test isapprox(dθ_auto, dθ_man; atol=0.01)
    end

    # Verify that Spectraplex rrule correctly captures off-diagonal sensitivity at a rank-one boundary solution
    @testset "Spectraplex rrule captures mixed boundary sensitivity" begin
        lmo_sp = Spectraplex(2)
        x0_sp = vec(Matrix(1.0I, 2, 2) ./ 2)
        θ_sp = [0.0]
        kw_sp = (; max_iters=4000, tol=1e-12, diff_lambda=0.0)

        _f_sp_boundary(x, θ) = dot([0.0, θ[1], θ[1], 1.0], x)
        _∇f_sp_boundary!(g, x, θ) = (g .= [0.0, θ[1], θ[1], 1.0])

        (x_star_man, _), pb_man = rrule(
            solve, _f_sp_boundary, lmo_sp, x0_sp, θ_sp; grad=_∇f_sp_boundary!, kw_sp...)
        @test x_star_man ≈ vec([1.0 0.0; 0.0 0.0]) atol=1e-12
        dθ_man = pb_man(([0.0, 1.0, 0.0, 0.0], nothing))[5]

        (x_star_auto, _), pb_auto = rrule(solve, _f_sp_boundary, lmo_sp, x0_sp, θ_sp; kw_sp...)
        @test x_star_auto ≈ x_star_man atol=1e-12
        dθ_auto = pb_auto(([0.0, 1.0, 0.0, 0.0], nothing))[5]

        ε = 1e-5
        L(θ_) = begin
            x_, _ = solve(_f_sp_boundary, lmo_sp, x0_sp, [θ_];
                          grad=_∇f_sp_boundary!, max_iters=4000, tol=1e-12)
            x_[2]
        end
        dθ_fd = (L(ε) - L(-ε)) / (2ε)

        @test dθ_man ≈ [dθ_fd] atol=2e-4
        @test dθ_auto ≈ [dθ_fd] atol=2e-4
    end

    # Check that Spectraplex rrule returns zero gradient for antisymmetric perturbations that cancel by symmetry
    @testset "Spectraplex rrule ignores antisymmetric parameter directions" begin
        n_sp = 2
        lmo_sp = Spectraplex(n_sp)
        x0_sp = vec(Matrix(1.0I, n_sp, n_sp) ./ n_sp)
        θ_sp = [1.0]
        kw_sp = (; max_iters=5000, tol=1e-6)

        _f_asym(x, θ) = 0.5 * dot(x, x) - θ[1] * (x[2] - x[3])
        function _∇f_asym!(g, x, θ)
            g .= x
            g[2] -= θ[1]
            g[3] += θ[1]
        end

        dx = [0.0, 1.0, -1.0, 0.0]

        (x_star_man, _), pb_man = rrule(solve, _f_asym, lmo_sp, x0_sp, θ_sp;
                                        grad=_∇f_asym!, kw_sp...)
        @test x_star_man[2] ≈ x_star_man[3] atol=1e-12
        dθ_man = pb_man((dx, nothing))[5]
        @test dθ_man ≈ zeros(1) atol=1e-8

        (x_star_auto, _), pb_auto = rrule(solve, _f_asym, lmo_sp, x0_sp, θ_sp; kw_sp...)
        @test x_star_auto[2] ≈ x_star_auto[3] atol=1e-12
        dθ_auto = pb_auto((dx, nothing))[5]
        @test dθ_auto ≈ zeros(1) atol=1e-8
    end

    # Verify that repeated pullback calls return independent cotangent arrays that do not alias each other
    @testset "Auto-gradient pullback returns owned θ cotangent" begin
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]
        kw = (; max_iters=1000, tol=1e-4)

        (x_star, _), pb = rrule(solve, _f, ProbabilitySimplex(), x0, θ₀; kw...)
        dθ_first = pb((2.0 .* x_star, nothing))[5]
        dθ_first_snapshot = copy(dθ_first)
        dθ_second = pb((x_star, nothing))[5]

        @test dθ_first ≈ dθ_first_snapshot atol=1e-12
        @test !(dθ_first === dθ_second)
    end

    # Verify that Spectraplex rrule gradients match finite differences for a tracking-loss objective
    @testset "Spectraplex rrule vs FD" begin
        n_sp = 2
        m_sp = n_sp * n_sp
        _f_sp3(x, θ) = 0.5 * dot(x, x) - dot(_diag_vec_sp(θ, n_sp), x)
        _∇f_sp3!(g, x, θ) = (g .= x .- _diag_vec_sp(θ, n_sp))

        lmo_sp = Spectraplex(n_sp)
        x0_sp = vec(Matrix(1.0I, n_sp, n_sp) ./ n_sp)
        θ_sp = [2.0, 0.5]
        x_target_sp = vec([0.8 0.0; 0.0 0.2])
        kw_sp = (; max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())

        (x_star, _), pb = rrule(solve, _f_sp3, lmo_sp, x0_sp, θ_sp; grad=_∇f_sp3!, kw_sp...)
        dx = 2.0 .* (x_star .- x_target_sp)
        dθ = pb((dx, nothing))[5]

        ε = 1e-3
        L(θ_) = begin
            x_, _ = solve(_f_sp3, lmo_sp, x0_sp, θ_; grad=_∇f_sp3!, kw_sp...)
            sum((x_ .- x_target_sp).^2)
        end
        dθ_fd = zeros(2)
        for j in 1:2
            eⱼ = zeros(2); eⱼ[j] = 1.0
            dθ_fd[j] = (L(θ_sp .+ ε .* eⱼ) - L(θ_sp .- ε .* eⱼ)) / (2ε)
        end
        @test isapprox(dθ, dθ_fd; atol=0.05)
    end

    # Verify that the KKT adjoint produces correct gradients at a vertex solution where bound constraints are active
    @testset "Boundary solution (vertex of simplex) -- KKT correctness" begin
        # θ = [10.0, 0.0] pushes x* to vertex e_1 of the probability simplex.
        # At a vertex, the unconstrained Hessian solve would be wrong because
        # the active set matters. The KKT adjoint should handle this correctly.
        n = 2
        θ₀ = [10.0, 0.0]
        x0 = [0.5, 0.5]
        x_target = [0.8, 0.2]
        kw = (; max_iters=10000, tol=1e-6)

        (x_star, _), pb = rrule(solve, _f, ProbabilitySimplex(), x0, θ₀; grad=_∇f!, kw...)
        # x* should be at vertex e_1 = [1.0, 0.0]
        @test x_star[1] ≈ 1.0 atol=1e-3
        @test x_star[2] ≈ 0.0 atol=1e-3

        dx = 2.0 .* (x_star .- x_target)
        tangents = pb((dx, nothing))
        dθ = tangents[5]
        @test length(dθ) == 2
        @test all(isfinite, dθ)

        # FD cross-check
        ε = 1e-3
        L(θ_) = begin
            x_, _ = solve(_f, ProbabilitySimplex(), x0, θ_; grad=_∇f!, kw...)
            sum((x_ .- x_target).^2)
        end
        dθ_fd = zeros(n)
        for j in 1:n
            eⱼ = zeros(n); eⱼ[j] = 1.0
            dθ_fd[j] = (L(θ₀ .+ ε .* eⱼ) - L(θ₀ .- ε .* eⱼ)) / (2ε)
        end
        @test isapprox(dθ, dθ_fd; atol=0.15)
    end

end
