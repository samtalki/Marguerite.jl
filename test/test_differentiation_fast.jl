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

@testset "Differentiation (fast representative coverage)" begin

    # Coverage map:
    # - simplex rrule (manual + auto) with finite-difference anchors
    # - parametric-oracle rrule
    # - KKT/CG internals and active-set edge cases
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

    @testset "Mixed-precision rrule pullback builds cached state" begin
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]
        kw = (; max_iters=5000, tol=1e-4)

        (x_star, _), pb = rrule(solve, _f, ProbabilitySimplex(Float32(1)), x0, θ₀; grad=_∇f!, kw...)
        dθ = pb((2 .* x_star, nothing))[5]
        @test length(dθ) == 2
        @test all(isfinite, dθ)
    end

    @testset "ZeroTangent pullback returns all NoTangent" begin
        θ₀ = [0.7, 0.3]
        x0 = [0.5, 0.5]

        (_, _), pb = rrule(solve, _f, ProbabilitySimplex(), x0, θ₀;
                           grad=_∇f!, max_iters=1000, tol=1e-4)
        tangents = pb((ChainRulesCore.ZeroTangent(), nothing))
        @test length(tangents) == 5
        @test all(t -> t isa NoTangent, tangents)
    end

    @testset "ParametricBox rrule (manual gradient)" begin
        n = 3
        _f_box(x, θ) = 0.5 * dot(x, x) - dot(θ[1:length(x)], x)
        _∇f_box!(g, x, θ) = (g .= x .- θ[1:length(x)])

        plmo = ParametricBox(θ -> θ[1:n], θ -> θ[n+1:2n])
        θ₀ = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        x0 = [0.5, 0.5, 0.5]
        kw = (; max_iters=5000, tol=1e-6)

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

    @testset "ParametricBox rrule (auto gradient)" begin
        n = 2
        _f_box2(x, θ) = 0.5 * dot(x, x) - dot(θ[1:length(x)], x)
        plmo = ParametricBox(θ -> θ[1:n], θ -> θ[n+1:2n])
        θ₀ = [0.3, 0.7, 1.0, 1.0]
        x0 = [0.5, 0.5]
        kw = (; max_iters=5000, tol=1e-6)

        (x_star, _), pb = rrule(solve, _f_box2, plmo, x0, θ₀; kw...)
        @test all(isfinite, x_star)
        dθ = pb((ones(n), nothing))[5]
        @test length(dθ) == 2n
        @test all(isfinite, dθ)
    end

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

    @testset "Boundary solution (vertex of simplex) -- KKT correctness" begin
        n = 2
        θ₀ = [10.0, 0.0]
        x0 = [0.5, 0.5]
        x_target = [0.8, 0.2]
        kw = (; max_iters=10000, tol=1e-6)

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

    # ── jacobian() via direct reduced Hessian ────────────────────────

    @testset "jacobian: simplex, manual grad, finite-diff match" begin
        n = 10
        # Use θ that gives an interior solution (all x*_i > 0) so FD is smooth
        θ = ones(n) ./ n .+ 0.01 .* [0.03, -0.02, 0.01, -0.01, 0.02, -0.03, 0.01, 0.0, -0.01, 0.02]
        x0 = fill(1.0/n, n)
        kw = (; max_iters=5000, tol=1e-10, step_rule=AdaptiveStepSize(), diff_lambda=1e-8)

        J, res = jacobian(_f, ProbSimplex(1.0), x0, θ; grad=_∇f!, kw...)
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
        @test norm(J - J_fd, Inf) < 0.01
    end

    @testset "jacobian: simplex, auto grad matches manual" begin
        n = 10
        θ = ones(n) ./ n .+ 0.01 .* [0.03, -0.02, 0.01, -0.01, 0.02, -0.03, 0.01, 0.0, -0.01, 0.02]
        x0 = fill(1.0/n, n)
        kw = (; max_iters=5000, tol=1e-10, step_rule=AdaptiveStepSize())

        J_auto, _ = jacobian(_f, ProbSimplex(1.0), x0, θ; kw...)
        J_manual, _ = jacobian(_f, ProbSimplex(1.0), x0, θ; grad=_∇f!, kw...)
        @test isapprox(J_auto, J_manual; atol=1e-6)
    end

    @testset "jacobian: box interior, finite-diff match" begin
        n = 5
        θ = 0.5 .* ones(n)  # solution will be in interior of [0,1]^n
        x0 = 0.5 .* ones(n)
        f_box(x, θ) = 0.5 * dot(x .- θ, x .- θ)
        ∇f_box!(g, x, θ) = (g .= x .- θ)
        kw = (; max_iters=5000, tol=1e-10, step_rule=AdaptiveStepSize(), diff_lambda=1e-8)

        J, _ = jacobian(f_box, Box(zeros(n), ones(n)), x0, θ; grad=∇f_box!, kw...)
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

    @testset "jacobian: matches pullback approach" begin
        n = 8
        θ = ones(n) ./ n .+ 0.01 .* [0.03, -0.02, 0.01, -0.01, 0.02, -0.03, 0.01, 0.0]
        x0 = fill(1.0/n, n)
        kw = (; max_iters=5000, tol=1e-8, step_rule=AdaptiveStepSize())

        J_direct, _ = jacobian(_f, ProbSimplex(1.0), x0, θ; grad=_∇f!, kw...)

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

    @testset "jacobian!: in-place matches out-of-place" begin
        n = 5
        θ = ones(n) ./ n .+ 0.01 .* [0.03, -0.02, 0.01, -0.01, 0.02]
        x0 = fill(1.0/n, n)
        kw = (; max_iters=5000, tol=1e-10, step_rule=AdaptiveStepSize(), diff_lambda=1e-8)

        J_alloc, res_alloc = jacobian(_f, ProbSimplex(1.0), x0, θ; grad=_∇f!, kw...)

        J_pre = zeros(n, n)
        J_inplace, res_inplace = jacobian!(J_pre, _f, ProbSimplex(1.0), x0, θ; grad=_∇f!, kw...)
        @test J_inplace === J_pre
        @test isapprox(J_alloc, J_pre; atol=1e-12)
    end

    @testset "jacobian: n_free == 0 (all at bounds)" begin
        n = 3
        # θ = [2, 2, 2]: solution of min ‖x - θ‖² on [0,1]^n is x* = [1,1,1] (all at upper bound)
        θ = fill(2.0, n)
        x0 = fill(0.5, n)
        f_box(x, θ) = 0.5 * dot(x .- θ, x .- θ)
        ∇f_box!(g, x, θ) = (g .= x .- θ)
        kw = (; max_iters=5000, tol=1e-10, step_rule=AdaptiveStepSize())

        J, _ = jacobian(f_box, Box(zeros(n), ones(n)), x0, θ; grad=∇f_box!, kw...)
        @test size(J) == (n, n)
        @test norm(J) < 1e-10
    end

end
