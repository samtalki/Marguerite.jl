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
using ChainRulesCore: ChainRulesCore, rrule, NoTangent

# Verify that implicit differentiation, pullbacks, and Jacobians work correctly across oracle types
@testset "Differentiation (fast representative coverage)" begin

    # Coverage map:
    # - simplex rrule (manual + auto) with finite-difference anchors
    # - parametric-oracle rrule
    # - KKT/CG internals and active set edge cases
    # - custom-oracle differentiation warnings/errors
    # - Spectraplex boundary sensitivity and active-face handling
    # Exhaustive variants remain in test/test_differentiation.jl.

    _f(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
    _∇f!(g, x, θ) = (g .= x .- θ)

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

    # Verify that the conjugate gradient solver finds the correct solution to a linear system
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

    # Verify that the CG solver handles regularization on a near-singular system
    @testset "CG solver with regularization" begin
        A = [1.0 0.999; 0.999 1.0]
        b = [1.0, 1.0]

        hvp_fn(d) = A * d
        u, cg_result = Marguerite._cg_solve(hvp_fn, b; maxiter=100, tol=1e-10, λ=1e-2)
        @test norm((A + 1e-2 * I) * u - b) < 1e-6
        @test cg_result.converged
    end

    # Verify that the CG solver warns and reports failure when given too few iterations
    @testset "CG non-convergence warning" begin
        A = [4.0 1.0; 1.0 3.0]
        b = [1.0, 2.0]
        hvp_fn(d) = A * d
        u, cg_result = @test_warn "CG solve did not converge" Marguerite._cg_solve(hvp_fn, b; maxiter=1, tol=1e-15, λ=0.0)
        @test !cg_result.converged
        @test cg_result.iterations == 1
        @test cg_result.residual_norm > 1e-15
    end

    # Verify that auto-gradient and manual-gradient pullbacks agree and match finite differences
    @testset "Auto-gradient + theta rrule (no grad)" begin
        n = 2
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]
        x_target = [0.6, 0.4]
        kw = (; max_iters=10000, tol=1e-8, step_rule=AdaptiveStepSize())

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

        (x_star_m, _), pb_m = rrule(solve, _f, ProbabilitySimplex(), x0, θ₀; grad=_∇f!, kw...)
        dθ_m = pb_m((2 .* (x_star_m .- x_target), nothing))[5]
        @test isapprox(dθ, dθ_m; atol=0.01)

        L(θ_) = begin
            x_, _ = solve(_f, ProbabilitySimplex(), x0, θ_; grad=_∇f!, kw...)
            sum((x_ .- x_target) .^ 2)
        end
        ε = 1e-6
        dθ_fd = zeros(n)
        for j in 1:n
            eⱼ = zeros(n); eⱼ[j] = 1.0
            dθ_fd[j] = (L(θ₀ .+ ε .* eⱼ) - L(θ₀ .- ε .* eⱼ)) / (2ε)
        end
        @test isapprox(dθ, dθ_fd; atol=0.02)
    end

    # Verify that the pullback works when the oracle uses a different precision than the parameters
    @testset "Mixed-precision rrule pullback builds cached state" begin
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]
        kw = (; max_iters=5000, tol=1e-4)

        (x_star, _), pb = rrule(solve, _f, ProbabilitySimplex(Float32(1)), x0, θ₀; grad=_∇f!, kw...)
        dθ = pb((2 .* x_star, nothing))[5]
        @test length(dθ) == 2
        @test all(isfinite, dθ)

        # Cross-check against Float64 oracle result
        (x_ref, _), pb_ref = rrule(solve, _f, ProbabilitySimplex(), x0, θ₀; grad=_∇f!, kw...)
        dθ_ref = pb_ref((2 .* x_ref, nothing))[5]
        @test isapprox(dθ, dθ_ref; atol=0.1)
    end

    # Verify that a zero upstream tangent produces all-zero (NoTangent) outputs
    @testset "ZeroTangent pullback returns all NoTangent" begin
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]

        (_, _), pb = rrule(solve, _f, ProbabilitySimplex(), x0, θ₀;
                           grad=_∇f!, max_iters=1000, tol=1e-4)
        tangents = pb((ChainRulesCore.ZeroTangent(), nothing))
        @test length(tangents) == 5
        @test all(t -> t isa NoTangent, tangents)
    end

    # Verify that differentiating through a parametric box constraint matches finite differences
    @testset "ParametricBox rrule (manual gradient)" begin
        n = 3
        _f_box(x, θ) = 0.5 * dot(x, x) - dot(θ[1:length(x)], x)
        _∇f_box!(g, x, θ) = (g .= x .- θ[1:length(x)])

        plmo = ParametricBox(θ -> θ[1:n], θ -> θ[n+1:2n])
        θ₀ = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        x0 = [0.5, 0.5, 0.5]
        kw = (; max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())

        (x_star, _), pb = rrule(solve, _f_box, plmo, x0, θ₀; grad=_∇f_box!, kw...)
        @test all(x_star .≈ 0.0)

        dθ = pb((ones(n), nothing))[5]
        @test length(dθ) == 2n
        @test all(isfinite, dθ)

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
        @test isapprox(dθ, dθ_fd; atol=0.03)
    end

    # Verify that parametric box differentiation works with automatic gradients
    @testset "ParametricBox rrule (auto gradient)" begin
        n = 2
        _f_box2(x, θ) = 0.5 * dot(x, x) - dot(θ[1:length(x)], x)
        _∇f_box2!(g, x, θ) = (g .= x .- θ[1:length(x)])
        plmo = ParametricBox(θ -> θ[1:n], θ -> θ[n+1:2n])
        θ₀ = [0.3, 0.7, 1.0, 1.0]
        x0 = [0.5, 0.5]
        kw = (; max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())

        (x_star, _), pb = rrule(solve, _f_box2, plmo, x0, θ₀; kw...)
        @test all(isfinite, x_star)
        dθ = pb((ones(n), nothing))[5]
        @test length(dθ) == 2n
        @test all(isfinite, dθ)

        # Cross-check against manual gradient variant
        (_, _), pb_m = rrule(solve, _f_box2, plmo, x0, θ₀; grad=_∇f_box2!, kw...)
        dθ_m = pb_m((ones(n), nothing))[5]
        @test isapprox(dθ, dθ_m; atol=0.01)
    end

    # Verify that constraint orthogonalization produces orthogonal vectors and correct null space projection
    @testset "Multi-constraint orthogonalization (_orthogonalize!)" begin
        a1 = [1.0, 1.0, 1.0]
        a2 = [1.0, 2.0, 1.0]
        a_vecs = [copy(a1), copy(a2)]
        a_norm_sqs = [dot(a1, a1), dot(a2, a2)]
        Marguerite._orthogonalize!(a_vecs, a_norm_sqs)

        @test abs(dot(a_vecs[1], a_vecs[2])) < 1e-12
        @test a_vecs[1] ≈ a1
        @test a_norm_sqs[1] ≈ dot(a_vecs[1], a_vecs[1])
        @test a_norm_sqs[2] ≈ dot(a_vecs[2], a_vecs[2])

        w = [1.0, 2.0, 3.0]
        out = similar(w)
        Marguerite._null_project!(out, w, a_vecs, a_norm_sqs)
        @test abs(dot(a1, out)) < 1e-10
        @test abs(dot(a2, out)) < 1e-10

        A = hcat(a1, a2)'
        P_exact = I - A' * inv(A * A') * A
        out_exact = P_exact * w
        @test out ≈ out_exact atol=1e-10
    end

    # Verify that the KKT adjoint solver handles multiple non-orthogonal equality constraints correctly
    @testset "KKT adjoint with multiple non-orthogonal equality constraints" begin
        n = 3
        _f_mc(x, θ) = 0.5 * dot(x, x) - dot(θ, x)

        eq_normals = [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0]]
        eq_rhs = [1.0, 1.5]
        as = Marguerite.ActiveConstraints{Float64}(Int[], Float64[], BitVector(), collect(1:n), eq_normals, eq_rhs)

        x_star = [0.2, 0.5, 0.3]
        μ_eq_true = [0.1, -0.05]
        θ₀ = x_star .- (μ_eq_true[1] .* eq_normals[1] .+ μ_eq_true[2] .* eq_normals[2])
        dx = [1.0, 0.0, 0.0]

        u, μ_bound, μ_eq, cg_result = Marguerite._kkt_adjoint_solve(
            _f_mc, Marguerite.SECOND_ORDER_BACKEND, x_star, θ₀, dx, as;
            cg_maxiter=100, cg_tol=1e-10, cg_λ=1e-6)

        @test cg_result.converged
        @test isempty(μ_bound)

        A = hcat(eq_normals...)'
        residual = u .+ A' * μ_eq .- dx
        @test norm(residual) < 1e-4
        @test abs(dot(eq_normals[1], u)) < 1e-6
        @test abs(dot(eq_normals[2], u)) < 1e-6

        K = [Matrix(1.0I, n, n) A'; A zeros(2, 2)]
        rhs = [dx; zeros(2)]
        sol = K \ rhs
        @test u ≈ sol[1:n] atol=1e-4
        @test μ_eq ≈ sol[n+1:end] atol=1e-4
    end

    # Verify that active set detection works correctly when the simplex budget is very large
    @testset "Active set tolerance with scaled budget" begin
        n = 100
        r_large = 1e6
        θ₀ = fill(r_large / n + 0.1, n)
        x0 = fill(r_large / n, n)

        lmo = Simplex(r_large)
        x_star, _ = solve(_f, lmo, x0, θ₀; grad=_∇f!, max_iters=5000, tol=1e-8)

        as = Marguerite.active_set(lmo, x_star)
        @test length(as.eq_normals) == 1
    end

    # Verify that custom oracles without active_set error unless assume_interior is set, and that wrappers match built-in results
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
        dθ_ref = bilevel_gradient(
            x -> sum((x .- [0.6, 0.4]).^2),
            _f, ProbabilitySimplex(), x0, θ₀;
            kw...)
        @test isapprox(dθ, dθ_ref; atol=0.1)

        (x_wrapped, _), pb_wrapped = rrule(solve, _f, wrapped_lmo, x0, θ₀; kw...)
        dθ_wrapped = pb_wrapped((2 .* x_wrapped, nothing))[5]
        (x_callable, _), pb_callable = rrule(solve, _f, callable_lmo, x0, θ₀; kw...)
        dθ_callable = pb_callable((2 .* x_callable, nothing))[5]
        (x_ref, _), pb_ref = rrule(solve, _f, ProbabilitySimplex(), x0, θ₀; kw...)
        dθ_ref_rrule = pb_ref((2 .* x_ref, nothing))[5]
        @test isapprox(dθ_wrapped, dθ_ref_rrule; atol=0.1)
        @test isapprox(dθ_callable, dθ_ref_rrule; atol=0.1)
    end

    # Verify that spectraplex differentiation at a rank-1 boundary matches finite differences
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

    # Verify that repeated pullback calls return independent copies of the parameter gradient
    @testset "Auto-gradient pullback returns owned theta cotangent" begin
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

    # Verify that differentiation is correct when the solution sits at a vertex of the simplex
    @testset "Boundary solution (vertex of simplex) -- KKT correctness" begin
        n = 2
        θ₀ = [10.0, 0.0]
        x0 = [0.5, 0.5]
        x_target = [0.8, 0.2]
        kw = (; max_iters=10000, tol=1e-6, step_rule=AdaptiveStepSize())

        (x_star, _), pb = rrule(solve, _f, ProbabilitySimplex(), x0, θ₀; grad=_∇f!, kw...)
        @test x_star[1] ≈ 1.0 atol=1e-3
        @test x_star[2] ≈ 0.0 atol=1e-3

        dθ = pb((2.0 .* (x_star .- x_target), nothing))[5]
        @test length(dθ) == n
        @test all(isfinite, dθ)

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

    # ── solution_jacobian() via direct reduced Hessian ────────────────────────

    # Verify that the solution Jacobian on a simplex matches finite differences
    @testset "solution_jacobian: simplex, manual grad, finite-diff match" begin
        n = 10
        # Use θ that gives an interior solution (all x*_i > 0) so FD is smooth
        θ = ones(n) ./ n .+ 0.01 .* [0.03, -0.02, 0.01, -0.01, 0.02, -0.03, 0.01, 0.0, -0.01, 0.02]
        x0 = fill(1.0/n, n)
        kw = (; max_iters=5000, tol=1e-10, step_rule=AdaptiveStepSize(), diff_lambda=1e-8)

        J, res = solution_jacobian(_f, ProbSimplex(1.0), x0, θ; grad=_∇f!, kw...)
        @test size(J) == (n, n)

        # Finite-difference Jacobian
        ε = 1e-6
        J_fd = zeros(n, n)
        for j in 1:n
            eⱼ = zeros(n); eⱼ[j] = ε
            x_plus, _ = solve(_f, ProbSimplex(1.0), x0, θ .+ eⱼ; grad=_∇f!, max_iters=5000, tol=1e-10, step_rule=AdaptiveStepSize())
            x_minus, _ = solve(_f, ProbSimplex(1.0), x0, θ .- eⱼ; grad=_∇f!, max_iters=5000, tol=1e-10, step_rule=AdaptiveStepSize())
            J_fd[:, j] .= (x_plus .- x_minus) ./ (2ε)
        end
        @test norm(J - J_fd, Inf) < 0.02
    end

    # Verify that auto-gradient and manual-gradient solution Jacobians agree
    @testset "solution_jacobian: simplex, auto grad matches manual" begin
        n = 10
        θ = ones(n) ./ n .+ 0.01 .* [0.03, -0.02, 0.01, -0.01, 0.02, -0.03, 0.01, 0.0, -0.01, 0.02]
        x0 = fill(1.0/n, n)
        kw = (; max_iters=5000, tol=1e-10, step_rule=AdaptiveStepSize())

        J_auto, _ = solution_jacobian(_f, ProbSimplex(1.0), x0, θ; kw...)
        J_manual, _ = solution_jacobian(_f, ProbSimplex(1.0), x0, θ; grad=_∇f!, kw...)
        @test isapprox(J_auto, J_manual; atol=1e-6)
    end

    # Verify that the solution Jacobian for a box-constrained interior solution matches finite differences
    @testset "solution_jacobian: box interior, finite-diff match" begin
        n = 5
        θ = 0.5 .* ones(n)  # solution will be in interior of [0,1]^n
        x0 = 0.5 .* ones(n)
        f_box(x, θ) = 0.5 * dot(x .- θ, x .- θ)
        ∇f_box!(g, x, θ) = (g .= x .- θ)
        kw = (; max_iters=5000, tol=1e-10, step_rule=AdaptiveStepSize(), diff_lambda=1e-8)

        J, _ = solution_jacobian(f_box, Box(zeros(n), ones(n)), x0, θ; grad=∇f_box!, kw...)
        @test size(J) == (n, n)

        ε = 1e-6
        J_fd = zeros(n, n)
        for j in 1:n
            eⱼ = zeros(n); eⱼ[j] = ε
            x_plus, _ = solve(f_box, Box(zeros(n), ones(n)), x0, θ .+ eⱼ; grad=∇f_box!, max_iters=5000, tol=1e-10, step_rule=AdaptiveStepSize())
            x_minus, _ = solve(f_box, Box(zeros(n), ones(n)), x0, θ .- eⱼ; grad=∇f_box!, max_iters=5000, tol=1e-10, step_rule=AdaptiveStepSize())
            J_fd[:, j] .= (x_plus .- x_minus) ./ (2ε)
        end
        @test isapprox(J, J_fd; atol=1e-3)
    end

    # Verify that the direct solution Jacobian matches the one reconstructed from pullback calls
    @testset "solution_jacobian: matches pullback approach" begin
        n = 8
        θ = ones(n) ./ n .+ 0.01 .* [0.03, -0.02, 0.01, -0.01, 0.02, -0.03, 0.01, 0.0]
        x0 = fill(1.0/n, n)
        kw = (; max_iters=5000, tol=1e-8, step_rule=AdaptiveStepSize())

        J_direct, _ = solution_jacobian(_f, ProbSimplex(1.0), x0, θ; grad=_∇f!, kw...)

        sr, pb = rrule(solve, _f, ProbSimplex(1.0), x0, θ; grad=_∇f!, kw...)
        J_pb = zeros(n, n)
        eᵢ = zeros(n)
        for i in 1:n
            fill!(eᵢ, 0.0); eᵢ[i] = 1.0
            tangents = pb((eᵢ, nothing))
            J_pb[i, :] .= tangents[5]
        end
        @test isapprox(J_direct, J_pb'; atol=1e-4)
    end

    # Verify that the in-place solution_jacobian! produces the same result as the allocating version
    @testset "solution_jacobian!: in-place matches out-of-place" begin
        n = 5
        θ = ones(n) ./ n .+ 0.01 .* [0.03, -0.02, 0.01, -0.01, 0.02]
        x0 = fill(1.0/n, n)
        kw = (; max_iters=5000, tol=1e-10, step_rule=AdaptiveStepSize(), diff_lambda=1e-8)

        J_alloc, res_alloc = solution_jacobian(_f, ProbSimplex(1.0), x0, θ; grad=_∇f!, kw...)

        J_pre = zeros(n, n)
        J_inplace, res_inplace = solution_jacobian!(J_pre, _f, ProbSimplex(1.0), x0, θ; grad=_∇f!, kw...)
        @test J_inplace === J_pre
        @test isapprox(J_alloc, J_pre; atol=1e-12)
    end

    # Verify that the solution Jacobian is zero when all variables are pinned at their bounds
    @testset "solution_jacobian: n_free == 0 (all at bounds)" begin
        n = 3
        # θ = [2, 2, 2]: solution of min ‖x - θ‖² on [0,1]^n is x* = [1,1,1] (all at upper bound)
        θ = fill(2.0, n)
        x0 = fill(0.5, n)
        f_box(x, θ) = 0.5 * dot(x .- θ, x .- θ)
        ∇f_box!(g, x, θ) = (g .= x .- θ)
        kw = (; max_iters=5000, tol=1e-10, step_rule=AdaptiveStepSize())

        J, _ = solution_jacobian(f_box, Box(zeros(n), ones(n)), x0, θ; grad=∇f_box!, kw...)
        @test size(J) == (n, n)
        @test norm(J) < 1e-10
    end

    # Verify that the spectraplex solution Jacobian at a rank-1 boundary matches finite differences
    @testset "solution_jacobian: Spectraplex boundary, finite-diff match" begin
        lmo_sp = Spectraplex(2)
        x0_sp = vec(Matrix(1.0I, 2, 2) ./ 2)
        θ_sp = [0.0]
        f_sp(x, θ) = dot([0.0, θ[1], θ[1], 1.0], x)
        ∇f_sp!(g, x, θ) = (g .= [0.0, θ[1], θ[1], 1.0])
        kw_sp = (; max_iters=4000, tol=1e-12, diff_lambda=0.0)

        J, _ = solution_jacobian(f_sp, lmo_sp, x0_sp, θ_sp; grad=∇f_sp!, kw_sp...)
        @test size(J) == (4, 1)

        ε = 1e-5
        J_fd = zeros(4, 1)
        x_p, _ = solve(f_sp, lmo_sp, x0_sp, θ_sp .+ [ε]; grad=∇f_sp!, kw_sp...)
        x_m, _ = solve(f_sp, lmo_sp, x0_sp, θ_sp .- [ε]; grad=∇f_sp!, kw_sp...)
        J_fd[:, 1] .= (x_p .- x_m) ./ (2ε)
        @test isapprox(J, J_fd; atol=2e-4)

        J_auto, _ = solution_jacobian(f_sp, lmo_sp, x0_sp, θ_sp; kw_sp...)
        @test isapprox(J_auto, J; atol=1e-6)
    end

    # Verify that the 3x3 spectraplex solution Jacobian with two parameters matches finite differences
    @testset "solution_jacobian: 3×3 Spectraplex rank-1 boundary" begin
        lmo_sp3 = Spectraplex(3)
        x0_sp3 = vec(Matrix(1.0I, 3, 3) ./ 3)
        θ_sp3 = [0.5, -0.3]
        # Objective with two parameters entering off-diagonal terms
        f_sp3(x, θ) = dot(vec([2.0 θ[1] 0.0; θ[1] 3.0 θ[2]; 0.0 θ[2] 1.0]), x)
        function ∇f_sp3!(g, x, θ)
            g .= vec([2.0 θ[1] 0.0; θ[1] 3.0 θ[2]; 0.0 θ[2] 1.0])
        end
        kw3 = (; max_iters=8000, tol=1e-12, diff_lambda=0.0)

        J3, res3 = solution_jacobian(f_sp3, lmo_sp3, x0_sp3, θ_sp3; grad=∇f_sp3!, kw3...)
        @test size(J3) == (9, 2)

        ε = 1e-5
        J3_fd = zeros(9, 2)
        for j in 1:2
            eⱼ = zeros(2); eⱼ[j] = ε
            x_p, _ = solve(f_sp3, lmo_sp3, x0_sp3, θ_sp3 .+ eⱼ; grad=∇f_sp3!, kw3...)
            x_m, _ = solve(f_sp3, lmo_sp3, x0_sp3, θ_sp3 .- eⱼ; grad=∇f_sp3!, kw3...)
            J3_fd[:, j] .= (x_p .- x_m) ./ (2ε)
        end
        @test isapprox(J3, J3_fd; atol=2e-4)
    end

    @testset "solution_jacobian!: DimensionMismatch on wrong size" begin
        n = 3
        θ = [0.4, 0.3, 0.3]
        x0 = fill(1.0/n, n)
        J_bad = zeros(n, n + 1)  # wrong column count
        @test_throws DimensionMismatch solution_jacobian!(J_bad, _f, ProbSimplex(1.0), x0, θ; grad=_∇f!)
    end

    # ── solution_jacobian with ParametricOracle ──────────────────────────

    # Shared ParametricBox setup (θ₁=0 puts x₁ at lower bound, x₂/x₃ free)
    _n_pbox = 3
    _f_pbox(x, θ) = 0.5 * dot(x, x) - dot(θ[1:_n_pbox], x)
    _∇f_pbox!(g, x, θ) = (g .= x .- θ[1:_n_pbox])
    _plmo_pbox = ParametricBox(θ -> θ[1:_n_pbox], θ -> θ[_n_pbox+1:2_n_pbox])
    _θ₀_pbox = [0.0, 0.6, 0.4, 1.0, 1.0, 1.0]
    _x0_pbox = fill(0.5, _n_pbox)
    _kw_pbox = (; max_iters=5000, tol=1e-10, step_rule=AdaptiveStepSize(), diff_lambda=1e-8)

    @testset "solution_jacobian: ParametricBox, finite-diff match" begin
        J, _ = solution_jacobian(_f_pbox, _plmo_pbox, _x0_pbox, _θ₀_pbox; grad=_∇f_pbox!, _kw_pbox...)
        @test size(J) == (_n_pbox, 2_n_pbox)

        ε = 1e-5
        m = 2_n_pbox
        J_fd = zeros(_n_pbox, m)
        for j in 1:m
            eⱼ = zeros(m); eⱼ[j] = ε
            x_plus, _ = solve(_f_pbox, _plmo_pbox, _x0_pbox, _θ₀_pbox .+ eⱼ; grad=_∇f_pbox!, _kw_pbox...)
            x_minus, _ = solve(_f_pbox, _plmo_pbox, _x0_pbox, _θ₀_pbox .- eⱼ; grad=_∇f_pbox!, _kw_pbox...)
            J_fd[:, j] .= (x_plus .- x_minus) ./ (2ε)
        end
        # Loose tolerance: x₁ sits exactly at the lower bound, so FD perturbations
        # cross the active set boundary and converge more slowly.
        @test isapprox(J, J_fd; atol=0.03)
    end

    @testset "solution_jacobian: ParametricBox, auto grad matches manual" begin
        J_manual, _ = solution_jacobian(_f_pbox, _plmo_pbox, _x0_pbox, _θ₀_pbox; grad=_∇f_pbox!, _kw_pbox...)
        J_auto, _ = solution_jacobian(_f_pbox, _plmo_pbox, _x0_pbox, _θ₀_pbox; _kw_pbox...)
        @test isapprox(J_auto, J_manual; atol=1e-6)
    end

    @testset "solution_jacobian: ParametricBox matches pullback" begin
        J_direct, _ = solution_jacobian(_f_pbox, _plmo_pbox, _x0_pbox, _θ₀_pbox; grad=_∇f_pbox!, _kw_pbox...)

        _, pb = rrule(solve, _f_pbox, _plmo_pbox, _x0_pbox, _θ₀_pbox; grad=_∇f_pbox!, _kw_pbox...)
        m = 2_n_pbox
        J_pb = zeros(_n_pbox, m)
        eᵢ = zeros(_n_pbox)
        for i in 1:_n_pbox
            fill!(eᵢ, 0.0); eᵢ[i] = 1.0
            tangents = pb((eᵢ, nothing))
            J_pb[i, :] .= tangents[5]
        end
        @test isapprox(J_direct, J_pb; atol=1e-4)
    end

    # Shared ParametricWeightedSimplex setup.
    # Use vertex-targeted objective (c=[3,0]) so FW converges fast from a
    # non-optimal starting point. x0=[0.3,0.3] ≠ x*=[1,0].
    _n_pws = 2
    _c_pws = [3.0, 0.0]
    _f_pws(x, θ) = 0.5 * dot(x, x) - dot(_c_pws, x)
    _∇f_pws!(g, x, θ) = (g .= x .- _c_pws)
    _plmo_pws = ParametricWeightedSimplex(θ -> [1.0, 1.0], θ -> θ[1], θ -> θ[2:3])
    _θ₀_pws = [1.0, 0.0, 0.0]
    _x0_pws = [0.3, 0.3]
    _kw_pws = (; max_iters=20000, tol=1e-10, step_rule=AdaptiveStepSize(), diff_lambda=1e-8)

    # Shared Simplex/ProbSimplex setup: θ = [r, c₁, c₂, c₃], f = 0.5‖x−c‖²
    _n_psim = 3
    _f_psim(x, θ) = 0.5 * dot(x .- θ[2:end], x .- θ[2:end])
    _∇f_psim!(g, x, θ) = (g .= x .- θ[2:end])
    _θ₀_psim = [1.0, 0.5, 0.3, 0.2]
    _x0_psim = fill(1.0/_n_psim, _n_psim)
    _kw_psim = (; max_iters=20000, tol=1e-10, step_rule=AdaptiveStepSize(), diff_lambda=1e-8)

    @testset "solution_jacobian: ParametricProbSimplex matches pullback" begin
        plmo = ParametricProbSimplex(θ -> θ[1])
        J_direct, _ = solution_jacobian(_f_psim, plmo, _x0_psim, _θ₀_psim; grad=_∇f_psim!, _kw_psim...)
        m = length(_θ₀_psim)
        @test size(J_direct) == (_n_psim, m)

        _, pb = rrule(solve, _f_psim, plmo, _x0_psim, _θ₀_psim; grad=_∇f_psim!, _kw_psim...)
        J_pb = zeros(_n_psim, m)
        eᵢ = zeros(_n_psim)
        for i in 1:_n_psim
            fill!(eᵢ, 0.0); eᵢ[i] = 1.0
            tangents = pb((eᵢ, nothing))
            J_pb[i, :] .= tangents[5]
        end
        @test isapprox(J_direct, J_pb; atol=1e-4)
    end

    @testset "solution_jacobian: ParametricSimplex matches pullback" begin
        plmo = ParametricSimplex(θ -> θ[1])
        J_direct, _ = solution_jacobian(_f_psim, plmo, _x0_psim, _θ₀_psim; grad=_∇f_psim!, _kw_psim...)
        m = length(_θ₀_psim)
        @test size(J_direct) == (_n_psim, m)

        _, pb = rrule(solve, _f_psim, plmo, _x0_psim, _θ₀_psim; grad=_∇f_psim!, _kw_psim...)
        J_pb = zeros(_n_psim, m)
        eᵢ = zeros(_n_psim)
        for i in 1:_n_psim
            fill!(eᵢ, 0.0); eᵢ[i] = 1.0
            tangents = pb((eᵢ, nothing))
            J_pb[i, :] .= tangents[5]
        end
        @test isapprox(J_direct, J_pb; atol=1e-4)
    end

    @testset "solution_jacobian: ParametricSimplex, auto grad matches manual" begin
        plmo = ParametricSimplex(θ -> θ[1])
        J_manual, _ = solution_jacobian(_f_psim, plmo, _x0_psim, _θ₀_psim; grad=_∇f_psim!, _kw_psim...)
        J_auto, _ = solution_jacobian(_f_psim, plmo, _x0_psim, _θ₀_psim; _kw_psim...)
        @test isapprox(J_auto, J_manual; atol=1e-6)
    end

    @testset "solution_jacobian: ParametricWeightedSimplex matches pullback" begin
        J_direct, _ = solution_jacobian(_f_pws, _plmo_pws, _x0_pws, _θ₀_pws; grad=_∇f_pws!, _kw_pws...)
        m = length(_θ₀_pws)
        @test size(J_direct) == (_n_pws, m)

        _, pb = rrule(solve, _f_pws, _plmo_pws, _x0_pws, _θ₀_pws; grad=_∇f_pws!, _kw_pws...)
        J_pb = zeros(_n_pws, m)
        eᵢ = zeros(_n_pws)
        for i in 1:_n_pws
            fill!(eᵢ, 0.0); eᵢ[i] = 1.0
            tangents = pb((eᵢ, nothing))
            J_pb[i, :] .= tangents[5]
        end
        @test isapprox(J_direct, J_pb; atol=1e-4)
    end

    @testset "solution_jacobian: ParametricWeightedSimplex, auto grad matches manual" begin
        J_manual, _ = solution_jacobian(_f_pws, _plmo_pws, _x0_pws, _θ₀_pws; grad=_∇f_pws!, _kw_pws...)
        J_auto, _ = solution_jacobian(_f_pws, _plmo_pws, _x0_pws, _θ₀_pws; _kw_pws...)
        @test isapprox(J_auto, J_manual; atol=1e-6)
    end

    @testset "solution_jacobian: ParametricProbSimplex, finite-diff match" begin
        n = 3
        f_psim(x, θ) = 0.5 * dot(x .- θ[2:end], x .- θ[2:end])
        ∇f_psim!(g, x, θ) = (g .= x .- θ[2:end])
        plmo = ParametricProbSimplex(θ -> θ[1])
        θ₀ = [1.0, 0.5, 0.3, 0.2]
        kw = (; max_iters=20000, tol=1e-10, step_rule=AdaptiveStepSize(), diff_lambda=1e-8)

        # x0 must lie on the equality simplex sum(x)=r(θ) for each perturbation
        x0_fn(θ) = fill(θ[1] / n, n)
        J, _ = solution_jacobian(f_psim, plmo, x0_fn(θ₀), θ₀; grad=∇f_psim!, kw...)
        m = length(θ₀)
        ε = 1e-5
        J_fd = zeros(n, m)
        for j in 1:m
            eⱼ = zeros(m); eⱼ[j] = ε
            θ_p = θ₀ .+ eⱼ; θ_m = θ₀ .- eⱼ
            x_plus, _ = solve(f_psim, plmo, x0_fn(θ_p), θ_p; grad=∇f_psim!, kw...)
            x_minus, _ = solve(f_psim, plmo, x0_fn(θ_m), θ_m; grad=∇f_psim!, kw...)
            J_fd[:, j] .= (x_plus .- x_minus) ./ (2ε)
        end
        @test isapprox(J, J_fd; atol=1e-3)
    end

    @testset "solution_jacobian: ParametricSimplex, finite-diff match" begin
        # Budget r=0.8 < sum(targets)=1.0 → budget constraint is clearly active
        n = 2
        f_csim(x, θ) = 0.5 * dot(x, x) - dot(θ[1:n], x)
        ∇f_csim!(g, x, θ) = (g .= x .- θ[1:n])
        plmo = ParametricSimplex(θ -> θ[end])
        θ₀ = [0.7, 0.3, 0.8]
        kw = (; max_iters=20000, tol=1e-10, step_rule=AdaptiveStepSize(), diff_lambda=1e-8)

        x0_fn(θ) = fill(θ[end] / n, n)
        J, _ = solution_jacobian(f_csim, plmo, x0_fn(θ₀), θ₀; grad=∇f_csim!, kw...)
        m = length(θ₀)
        ε = 1e-5
        J_fd = zeros(n, m)
        for j in 1:m
            eⱼ = zeros(m); eⱼ[j] = ε
            θ_p = θ₀ .+ eⱼ; θ_m = θ₀ .- eⱼ
            x_plus, _ = solve(f_csim, plmo, x0_fn(θ_p), θ_p; grad=∇f_csim!, kw...)
            x_minus, _ = solve(f_csim, plmo, x0_fn(θ_m), θ_m; grad=∇f_csim!, kw...)
            J_fd[:, j] .= (x_plus .- x_minus) ./ (2ε)
        end
        @test isapprox(J, J_fd; atol=1e-3)
    end

    @testset "solution_jacobian: ParametricWeightedSimplex, finite-diff match" begin
        # Use a vertex-targeted objective (c=[3,0]) so FW converges fast to a
        # vertex, avoiding the O(1/k) convergence issue on face-interior solutions.
        n = 2
        f_ws(x, θ) = 0.5 * dot(x, x) - dot(θ[1:n], x)
        ∇f_ws!(g, x, θ) = (g .= x .- θ[1:n])
        plmo_ws = ParametricWeightedSimplex(
            θ -> [1.0, 1.0], θ -> θ[n+1], θ -> θ[n+2:n+1+n])
        θ₀_ws = [3.0, 0.0, 1.0, 0.0, 0.0]
        x0_ws = [0.5, 0.5]
        kw_ws = (; max_iters=10000, tol=1e-6, step_rule=AdaptiveStepSize(), diff_lambda=1e-8)

        J, _ = solution_jacobian(f_ws, plmo_ws, x0_ws, θ₀_ws; grad=∇f_ws!, kw_ws...)
        m = length(θ₀_ws)
        ε = 1e-4
        J_fd = zeros(n, m)
        for j in 1:m
            eⱼ = zeros(m); eⱼ[j] = ε
            x_plus, _ = solve(f_ws, plmo_ws, x0_ws, θ₀_ws .+ eⱼ; grad=∇f_ws!, kw_ws...)
            x_minus, _ = solve(f_ws, plmo_ws, x0_ws, θ₀_ws .- eⱼ; grad=∇f_ws!, kw_ws...)
            J_fd[:, j] .= (x_plus .- x_minus) ./ (2ε)
        end
        @test isapprox(J, J_fd; atol=0.05)
    end

    @testset "solution_jacobian: ParametricBox, interior solution (n_bound=0)" begin
        # All θ values keep x* well inside the box → no active bounds
        n = 3
        f_int(x, θ) = 0.5 * dot(x, x) - dot(θ[1:n], x)
        ∇f_int!(g, x, θ) = (g .= x .- θ[1:n])
        plmo = ParametricBox(θ -> zeros(n), θ -> ones(n))
        θ₀ = [0.5, 0.3, 0.4]
        x0 = fill(0.5, n)
        kw = (; max_iters=5000, tol=1e-10, step_rule=AdaptiveStepSize(), diff_lambda=1e-8)

        J, _ = solution_jacobian(f_int, plmo, x0, θ₀; grad=∇f_int!, kw...)
        # x* = θ (unconstrained), so ∂x*/∂θ = I
        @test isapprox(J, Matrix(1.0I, n, n); atol=1e-4)
    end

    @testset "solution_jacobian: unsupported ParametricOracle throws ArgumentError" begin
        struct _TestParametricOracle <: Marguerite.ParametricOracle end
        Marguerite.materialize(::_TestParametricOracle, θ) = Box(0.0, 1.0)
        plmo = _TestParametricOracle()
        f_test(x, θ) = 0.5 * dot(x, x)
        x0 = [0.5, 0.5]
        θ₀ = [1.0]
        @test_throws ArgumentError solution_jacobian(f_test, plmo, x0, θ₀)
    end

    # ------------------------------------------------------------------
    # T1. Spectraplex pack/unpack roundtrip tests
    # ------------------------------------------------------------------

    # Verify that packing and unpacking spectraplex tangent coordinates recovers the original data
    @testset "Spectraplex pack/unpack roundtrip" begin
        for k in [1, 2, 3]
            # Check that trace zero pack then unpack recovers the original matrix
            @testset "trace zero k=$k" begin
                d = Marguerite._spectraplex_trace_zero_dim(k)
                d == 0 && continue
                # Random trace zero symmetric matrix
                M = randn(k, k)
                M = (M + M') / 2
                t = tr(M) / k
                for i in 1:k; M[i, i] -= t; end
                @test abs(tr(M)) < 1e-12
                z = zeros(d)
                Marguerite._spectraplex_pack_trace_zero!(z, M)
                M2 = zeros(k, k)
                Marguerite._spectraplex_unpack_trace_zero!(M2, z)
                @test isapprox(M, M2; atol=1e-12)
            end
        end

        # Check that compress then expand recovers the original tangent vector
        @testset "compress/expand roundtrip" begin
            n = 3
            # Rank-1 solution: U is 3×1, V_perp is 3×2
            v1 = normalize(randn(3))
            U = reshape(v1, 3, 1)
            V_perp = nullspace(U')
            rank = size(U, 2)
            nullity = size(V_perp, 2)
            d = Marguerite._spectraplex_tangent_dim(rank, nullity)
            z = randn(d)

            # Allocate buffers
            face_buf = zeros(rank, rank)
            mixed_buf = zeros(rank, nullity)
            tmp_face = zeros(n, rank)
            tmp_null = zeros(n, nullity)
            full_buf = zeros(n, n)
            cross_buf = zeros(n, n)

            out = zeros(n * n)
            Marguerite._spectraplex_expand!(out, z, U, V_perp,
                face_buf, mixed_buf, tmp_face, tmp_null, full_buf, cross_buf)

            z2 = zeros(d)
            Marguerite._spectraplex_compress!(z2, out, U, V_perp,
                tmp_face, tmp_null, face_buf, mixed_buf, full_buf)

            @test isapprox(z, z2; atol=1e-10)
        end
    end

    # ------------------------------------------------------------------
    # T2. Spectraplex mixed curvature term direct test
    # ------------------------------------------------------------------

    # Verify that the spectraplex mixed curvature correction computes the expected commutator
    @testset "Spectraplex mixed curvature term" begin
        rank, nullity = 1, 2
        G_uu = randn(rank, rank)
        G_uu = (G_uu + G_uu') / 2  # symmetric
        G_vv = randn(nullity, nullity)
        G_vv = (G_vv + G_vv') / 2
        B = randn(rank, nullity)

        # Build z with known face block (zero) and known mixed block B
        face_dim = Marguerite._spectraplex_trace_zero_dim(rank)
        d = Marguerite._spectraplex_tangent_dim(rank, nullity)
        z = zeros(d)
        p = face_dim + 1
        for j in 1:nullity
            for i in 1:rank
                z[p] = B[i, j]
                p += 1
            end
        end

        out = zeros(d)
        mixed_buf = zeros(rank, nullity)
        mixed_curv_buf = zeros(rank, nullity)
        Marguerite._spectraplex_add_mixed_curvature!(out, z, G_uu, G_vv, mixed_buf, mixed_curv_buf)

        # Expected: B*G_vv - G_uu*B, packed into the mixed block of out
        expected_mixed = B * G_vv - G_uu * B
        out_mixed = zeros(rank, nullity)
        p = face_dim + 1
        for j in 1:nullity
            for i in 1:rank
                out_mixed[i, j] = out[p]
                p += 1
            end
        end
        @test isapprox(out_mixed, expected_mixed; atol=1e-12)
    end

    # ------------------------------------------------------------------
    # T3. Spectraplex full-rank differentiation
    # ------------------------------------------------------------------

    # Verify that spectraplex differentiation works when the solution has full rank
    @testset "Spectraplex full-rank rrule" begin
        n = 2
        m = n * n
        # Use a strongly-convex quadratic where the spectraplex-projected solution
        # is full rank. θ scales a diagonal target; at θ=[1], solution ≈ I/2.
        D_target = Diagonal([0.6, 0.4])
        θ_fr = [1.0]
        f_full(x, θ) = begin
            X = reshape(x, n, n)
            target = θ[1] * D_target
            0.5 * sum((X .- target) .^ 2)
        end
        lmo_full = Spectraplex(n)
        x0_full = collect(vec(D_target))  # start at target (already feasible)
        kw_full = (tol=1e-6, max_iters=5000, step_rule=Marguerite.AdaptiveStepSize())

        sr, pb = rrule(solve, f_full, lmo_full, x0_full, θ_fr; kw_full...)
        x_star = sr.x

        # Verify full rank
        X = reshape(x_star, n, n)
        eigs = eigvals(Symmetric((X + X') / 2))
        @test all(eigs .> 1e-3)

        # Pullback
        dy = (ones(m), nothing)
        _, _, _, _, dθ = pb(dy)

        # Finite difference check
        ε = 1e-5
        x_p, _ = solve(f_full, lmo_full, x0_full, θ_fr .+ [ε]; kw_full...)
        x_m, _ = solve(f_full, lmo_full, x0_full, θ_fr .- [ε]; kw_full...)
        dθ_fd = [dot(ones(m), x_p .- x_m) / (2ε)]
        @test isapprox(dθ, dθ_fd; atol=5e-3)
    end

    # ------------------------------------------------------------------
    # T4. _factor_reduced_hessian fallback tests
    # ------------------------------------------------------------------

    # Verify that the reduced Hessian factorization uses Cholesky for positive-definite, LU for indefinite, and errors on singular
    @testset "_factor_reduced_hessian fallback" begin
        # PD matrix → Cholesky succeeds
        H_pd = [4.0 1.0; 1.0 3.0]
        factor_pd = Marguerite._factor_reduced_hessian(copy(H_pd), 0.0)
        @test factor_pd isa Cholesky

        # Indefinite matrix → LU fallback
        H_indef = [1.0 0.0; 0.0 -1.0]
        factor_lu = @test_warn "not positive definite" Marguerite._factor_reduced_hessian(copy(H_indef), 0.0)
        @test factor_lu isa LU

        # Singular matrix → error
        H_sing = [0.0 0.0; 0.0 0.0]
        @test_throws ErrorException Marguerite._factor_reduced_hessian(copy(H_sing), 0.0)
    end

end
