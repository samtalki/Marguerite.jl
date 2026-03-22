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
using Random
using ChainRulesCore: ChainRulesCore, rrule, NoTangent
import DifferentiationInterface as DI

# Exhaustive bilevel coverage.
# Representative default-path checks live in test/test_bilevel_fast.jl.

@testset "Bilevel Optimization" begin
    Random.seed!(123)
    n = 5

    # Random PD Hessian (not identity -- makes x*(θ) nontrivial)
    A = randn(n, n)
    H = A'A + 0.5I

    # Shared objective/gradient — hoisted so all testsets share function types,
    # avoiding redundant rule compilation per anonymous closure.
    _f(x, θ) = 0.5 * dot(x, H * x) - dot(θ, x)
    _∇f!(g, x, θ) = (g .= H * x .- θ)

    # Identity-Hessian variant for clean FD checks
    _f_id(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
    _∇f_id!(g, x, θ) = (g .= x .- θ)

    # Embed θ on the diagonal of an n×n matrix (vectorized).
    function _diag_vec_sp(θ, n)
        t = zeros(eltype(θ), n * n)
        @inbounds for i in 1:n
            t[(i - 1) * n + i] = θ[i]
        end
        t
    end

    struct CallableProbSimplexBilevel end

    function (::CallableProbSimplexBilevel)(v::AbstractVector, g::AbstractVector)
        fill!(v, zero(eltype(v)))
        v[argmin(g)] = one(eltype(v))
        return v
    end

    function Marguerite.active_set(::CallableProbSimplexBilevel, x::AbstractVector{T}; tol::Real=1e-8) where T
        return Marguerite.active_set(ProbabilitySimplex(T(1)), x; tol=tol)
    end

    lmo = ProbabilitySimplex()
    x0 = fill(1.0 / n, n)
    solve_kw = (; max_iters=10_000, tol=1e-3)

    # Target on the simplex
    x_target = zeros(n)
    x_target[1] = 0.6; x_target[2] = 0.3; x_target[3] = 0.1
    outer_loss(x) = sum((x .- x_target).^2)

    # Bilevel step: solve inner, compute outer loss gradient via rrule
    function bilevel_step(θ)
        (x_star, result), pb = rrule(solve, _f, lmo, x0, θ; grad=_∇f!, solve_kw...)
        loss = sum((x_star .- x_target).^2)
        dx = 2.0 .* (x_star .- x_target)
        tangents = pb((dx, nothing))
        dθ = tangents[end]
        return x_star, loss, dθ
    end

    @testset "Bilevel convergence" begin
        θ = H * x_target  # warm start
        η = 0.1
        outer_iters = 80

        losses = Float64[]
        errors = Float64[]

        for k in 1:outer_iters
            x_star, loss, dθ = bilevel_step(θ)
            push!(losses, loss)
            push!(errors, norm(x_star .- x_target))
            θ .= θ .- η .* dθ
        end

        @test losses[end] < 1e-4
        x_final, _ = solve(_f, lmo, x0, θ; grad=_∇f!, solve_kw...)
        @test isapprox(x_final, x_target; atol=1e-2)
    end

    @testset "AD gradient matches finite differences" begin
        θ_test = [0.3, 0.25, 0.2, 0.15, 0.1]
        fd_kw = (; max_iters=10_000, tol=1e-4)  # reduced from 50k for test speed

        (x_ad, _), pb = rrule(solve, _f_id, lmo, x0, θ_test; grad=_∇f_id!, fd_kw...)
        dx = 2.0 .* (x_ad .- x_target)
        dθ_ad = pb((dx, nothing))[end]

        ε = 1e-3
        dθ_fd = zeros(n)
        for j in 1:n
            eⱼ = zeros(n); eⱼ[j] = 1.0
            x_plus, _ = solve(_f_id, lmo, x0, θ_test .+ ε .* eⱼ; grad=_∇f_id!, fd_kw...)
            x_minus, _ = solve(_f_id, lmo, x0, θ_test .- ε .* eⱼ; grad=_∇f_id!, fd_kw...)
            loss_plus = sum((x_plus .- x_target).^2)
            loss_minus = sum((x_minus .- x_target).^2)
            dθ_fd[j] = (loss_plus - loss_minus) / (2ε)
        end

        @test isapprox(dθ_ad, dθ_fd; atol=0.15)  # relaxed from 0.1 to match reduced iterations
    end

    @testset "bilevel_solve (manual gradient)" begin
        θ_test = H * x_target
        x_bs, dθ_bs, cg_bs = bilevel_solve(outer_loss, _f, lmo, x0, θ_test; grad=_∇f!, solve_kw...)
        @test cg_bs.converged

        x_rrule, _, dθ_rrule = bilevel_step(θ_test)
        @test isapprox(x_bs, x_rrule; atol=1e-6)
        @test isapprox(dθ_bs, dθ_rrule; atol=1e-4)
    end

    @testset "bilevel_solve (auto gradient)" begin
        θ_test = H * x_target
        x_bs, dθ_bs, _ = bilevel_solve(outer_loss, _f, lmo, x0, θ_test; solve_kw...)

        x_manual, dθ_manual, _ = bilevel_solve(outer_loss, _f, lmo, x0, θ_test; grad=_∇f!, solve_kw...)
        @test isapprox(x_bs, x_manual; atol=1e-6)
        @test isapprox(dθ_bs, dθ_manual; atol=1e-4)
    end

    @testset "bilevel_gradient matches finite differences" begin
        θ_test = [0.3, 0.25, 0.2, 0.15, 0.1]
        fd_kw = (; max_iters=10_000, tol=1e-4)  # reduced from 50k for test speed

        dθ_bg = bilevel_gradient(outer_loss, _f_id, lmo, x0, θ_test; grad=_∇f_id!, fd_kw...)

        ε = 1e-3
        dθ_fd = zeros(n)
        for j in 1:n
            eⱼ = zeros(n); eⱼ[j] = 1.0
            x_plus, _ = solve(_f_id, lmo, x0, θ_test .+ ε .* eⱼ; grad=_∇f_id!, fd_kw...)
            x_minus, _ = solve(_f_id, lmo, x0, θ_test .- ε .* eⱼ; grad=_∇f_id!, fd_kw...)
            dθ_fd[j] = (outer_loss(x_plus) - outer_loss(x_minus)) / (2ε)
        end

        @test isapprox(dθ_bg, dθ_fd; atol=0.15)  # relaxed from 0.1 to match reduced iterations
    end

    @testset "bilevel_gradient auto vs manual" begin
        θ_test = [0.3, 0.25, 0.2, 0.15, 0.1]
        fd_kw = (; max_iters=10_000, tol=1e-3)
        dθ_manual = bilevel_gradient(outer_loss, _f_id, lmo, x0, θ_test; grad=_∇f_id!, fd_kw...)
        dθ_auto = bilevel_gradient(outer_loss, _f_id, lmo, x0, θ_test; fd_kw...)
        @test isapprox(dθ_auto, dθ_manual; atol=1e-4)
    end

    @testset "Spectraplex bilevel matches mixed boundary finite differences" begin
        lmo_sp = Spectraplex(2)
        x0_sp = vec(Matrix(1.0I, 2, 2) ./ 2)
        θ_sp = [0.0]
        kw_sp = (; max_iters=4000, tol=1e-12, diff_lambda=0.0)

        _f_sp_boundary(x, θ) = dot([0.0, θ[1], θ[1], 1.0], x)
        _∇f_sp_boundary!(g, x, θ) = (g .= [0.0, θ[1], θ[1], 1.0])
        outer_loss_sp(x) = x[2]

        x_bs, dθ_bs, cg_bs = bilevel_solve(
            outer_loss_sp, _f_sp_boundary, lmo_sp, x0_sp, θ_sp;
            grad=_∇f_sp_boundary!, kw_sp...)
        @test cg_bs.converged
        @test x_bs ≈ vec([1.0 0.0; 0.0 0.0]) atol=1e-12

        dθ_manual = bilevel_gradient(
            outer_loss_sp, _f_sp_boundary, lmo_sp, x0_sp, θ_sp;
            grad=_∇f_sp_boundary!, kw_sp...)
        dθ_auto = bilevel_gradient(outer_loss_sp, _f_sp_boundary, lmo_sp, x0_sp, θ_sp; kw_sp...)

        ε = 1e-5
        L(θ_) = begin
            x_, _ = solve(_f_sp_boundary, lmo_sp, x0_sp, [θ_];
                          grad=_∇f_sp_boundary!, max_iters=4000, tol=1e-12)
            x_[2]
        end
        dθ_fd = (L(ε) - L(-ε)) / (2ε)

        @test dθ_bs ≈ [dθ_fd] atol=2e-4
        @test dθ_manual ≈ dθ_bs atol=1e-10
        @test dθ_auto ≈ [dθ_fd] atol=2e-4
    end

    @testset "Custom oracle differentiation requires active_set or explicit interior assumption" begin
        x0_plain = [0.5, 0.5]
        θ_plain = [0.7, 0.3]
        outer_loss_plain(x) = sum((x .- [0.6, 0.4]).^2)
        kw_plain = (; grad=_∇f_id!, max_iters=1000, tol=1e-3)

        plain_lmo(v, g) = (fill!(v, 0.0); i = argmin(g); v[i] = 1.0; v)
        callable_lmo = CallableProbSimplexBilevel()

        @test_throws ArgumentError bilevel_solve(
            outer_loss_plain, _f_id, plain_lmo, x0_plain, θ_plain; kw_plain...)

        x_bs, dθ_bs, cg_bs = @test_logs (:warn, r"assume_interior=true") bilevel_solve(
            outer_loss_plain, _f_id, plain_lmo, x0_plain, θ_plain;
            assume_interior=true, kw_plain...)
        @test cg_bs.converged
        @test all(isfinite, dθ_bs)

        @test_throws ArgumentError bilevel_gradient(
            outer_loss_plain, _f_id, plain_lmo, x0_plain, θ_plain; kw_plain...)

        dθ_bg = @test_logs (:warn, r"assume_interior=true") bilevel_gradient(
            outer_loss_plain, _f_id, plain_lmo, x0_plain, θ_plain;
            assume_interior=true, kw_plain...)
        @test isapprox(dθ_bg, dθ_bs; atol=1e-8)

        x_callable, dθ_callable, cg_callable = bilevel_solve(
            outer_loss_plain, _f_id, callable_lmo, x0_plain, θ_plain; kw_plain...)
        @test cg_callable.converged
        @test all(isfinite, dθ_callable)

        dθ_callable_bg = bilevel_gradient(
            outer_loss_plain, _f_id, callable_lmo, x0_plain, θ_plain; kw_plain...)
        @test isapprox(dθ_callable_bg, dθ_callable; atol=1e-8)

        x_ref, dθ_ref, _ = bilevel_solve(
            outer_loss_plain, _f_id, ProbabilitySimplex(), x0_plain, θ_plain; kw_plain...)
        @test isapprox(x_callable, x_ref; atol=1e-8)
        @test isapprox(dθ_callable, dθ_ref; atol=0.1)
    end

    @testset "bilevel_solve with default backends" begin
        outer_loss_sm(x) = sum((x .- [0.6, 0.4]).^2)
        default_kw = (; max_iters=10_000, tol=1e-3)

        x_def, dθ_def, cg_def = bilevel_solve(outer_loss_sm, _f_id, lmo,
                                               fill(0.5, 2), [0.7, 0.3]; grad=_∇f_id!, default_kw...)
        @test cg_def.converged
        @test all(isfinite, dθ_def)
    end

    @testset "bilevel_solve with custom CG params" begin
        θ_test = H * x_target

        x_bs, dθ_bs, cg_bs = bilevel_solve(outer_loss, _f, lmo, x0, θ_test;
                                    grad=_∇f!, solve_kw..., diff_cg_maxiter=100, diff_cg_tol=1e-8, diff_lambda=1e-3)
        @test cg_bs.converged
        @test all(isfinite, dθ_bs)
        @test length(dθ_bs) == n
    end

    # ------------------------------------------------------------------
    # ParametricOracle bilevel tests
    # ------------------------------------------------------------------

    @testset "bilevel_solve with ParametricBox (manual gradient)" begin
        n_box = 3
        _f_box(x, θ) = 0.5 * dot(x, x) - dot(θ[1:n_box], x)
        _∇f_box!(g, x, θ) = (g .= x .- θ[1:n_box])
        plmo = ParametricBox(θ -> θ[n_box+1:2n_box], θ -> θ[2n_box+1:3n_box])

        x_target_box = [0.3, 0.5, 0.2]
        outer_loss_box(x) = sum((x .- x_target_box).^2)

        # θ = [obj_params; lb; ub]
        θ_box = [x_target_box; zeros(n_box); ones(n_box)]
        x0_box = [0.5, 0.5, 0.5]
        box_kw = (; max_iters=20_000, tol=1e-3)

        x_bs, dθ_bs, cg_bs = bilevel_solve(outer_loss_box, _f_box, plmo, x0_box, θ_box; grad=_∇f_box!, box_kw...)
        @test all(isfinite, dθ_bs)
        @test length(dθ_bs) == 3n_box
    end

    @testset "bilevel_solve with ParametricBox (auto gradient)" begin
        n_box = 3
        _f_box_auto(x, θ) = 0.5 * dot(x, x) - dot(θ[1:n_box], x)
        plmo = ParametricBox(θ -> θ[n_box+1:2n_box], θ -> θ[2n_box+1:3n_box])

        x_target_box = [0.3, 0.5, 0.2]
        outer_loss_box(x) = sum((x .- x_target_box).^2)

        θ_box = [x_target_box; zeros(n_box); ones(n_box)]
        x0_box = [0.5, 0.5, 0.5]
        box_kw = (; max_iters=20_000, tol=1e-3)

        x_bs, dθ_bs, cg_bs = bilevel_solve(outer_loss_box, _f_box_auto, plmo, x0_box, θ_box; box_kw...)
        @test all(isfinite, dθ_bs)
        @test length(dθ_bs) == 3n_box
    end

    @testset "bilevel_gradient with ParametricBox" begin
        n_box = 2
        _f_bg(x, θ) = 0.5 * dot(x, x) - dot(θ[1:n_box], x)
        _∇f_bg!(g, x, θ) = (g .= x .- θ[1:n_box])
        plmo = ParametricBox(θ -> θ[n_box+1:2n_box], θ -> θ[2n_box+1:3n_box])

        x_target_bg = [0.3, 0.7]
        outer_loss_bg(x) = sum((x .- x_target_bg).^2)

        θ_bg = [x_target_bg; zeros(n_box); ones(n_box)]
        x0_bg = [0.5, 0.5]
        bg_kw = (; max_iters=20_000, tol=1e-3)

        dθ_manual = bilevel_gradient(outer_loss_bg, _f_bg, plmo, x0_bg, θ_bg; grad=_∇f_bg!, bg_kw...)
        dθ_auto = bilevel_gradient(outer_loss_bg, _f_bg, plmo, x0_bg, θ_bg; bg_kw...)
        @test isapprox(dθ_auto, dθ_manual; atol=1e-4)
    end

    @testset "bilevel_gradient with ParametricBox matches finite differences" begin
        n_fd = 2
        _f_fd(x, θ) = 0.5 * dot(x, x) - dot(θ[1:n_fd], x)
        _∇f_fd!(g, x, θ) = (g .= x .- θ[1:n_fd])
        plmo_fd = ParametricBox(θ -> θ[n_fd+1:2n_fd], θ -> θ[2n_fd+1:3n_fd])

        x_target_fd = [0.3, 0.7]
        outer_loss_fd(x) = sum((x .- x_target_fd).^2)

        θ_fd = [x_target_fd; zeros(n_fd); ones(n_fd)]
        x0_fd = [0.5, 0.5]
        fd_kw = (; max_iters=10_000, tol=1e-4)

        dθ_bg = bilevel_gradient(outer_loss_fd, _f_fd, plmo_fd, x0_fd, θ_fd; grad=_∇f_fd!, fd_kw...)

        ε = 1e-4
        m_fd = length(θ_fd)
        dθ_fd = zeros(m_fd)
        for j in 1:m_fd
            eⱼ = zeros(m_fd); eⱼ[j] = 1.0
            x_plus, _ = solve(_f_fd, plmo_fd, x0_fd, θ_fd .+ ε .* eⱼ; grad=_∇f_fd!, fd_kw...)
            x_minus, _ = solve(_f_fd, plmo_fd, x0_fd, θ_fd .- ε .* eⱼ; grad=_∇f_fd!, fd_kw...)
            dθ_fd[j] = (outer_loss_fd(x_plus) - outer_loss_fd(x_minus)) / (2ε)
        end

        @test isapprox(dθ_bg, dθ_fd; atol=0.15)
    end

    @testset "bilevel with Spectraplex" begin
        n_sp = 2
        _f_sp(x, θ) = 0.5 * dot(x, x) - dot(_diag_vec_sp(θ, n_sp), x)
        _∇f_sp!(g, x, θ) = (g .= x .- _diag_vec_sp(θ, n_sp))

        lmo_sp = Spectraplex(n_sp)
        x0_sp = vec(Matrix(1.0I, n_sp, n_sp) ./ n_sp)
        θ_sp = [2.0, 0.5]
        x_target_sp = vec([0.8 0.0; 0.0 0.2])
        outer_loss_sp(x) = sum((x .- x_target_sp).^2)
        sp_kw = (; max_iters=5000, tol=1e-6)

        # bilevel_solve manual gradient
        x_bs, dθ_bs, cg_bs = bilevel_solve(outer_loss_sp, _f_sp, lmo_sp, x0_sp, θ_sp; grad=_∇f_sp!, sp_kw...)
        @test cg_bs.converged
        @test all(isfinite, dθ_bs)

        # bilevel_gradient auto vs manual
        dθ_manual = bilevel_gradient(outer_loss_sp, _f_sp, lmo_sp, x0_sp, θ_sp; grad=_∇f_sp!, sp_kw...)
        dθ_auto = bilevel_gradient(outer_loss_sp, _f_sp, lmo_sp, x0_sp, θ_sp; sp_kw...)
        @test isapprox(dθ_auto, dθ_manual; atol=1e-4)

        # bilevel_gradient vs FD
        ε = 1e-3
        dθ_fd = zeros(2)
        for j in 1:2
            eⱼ = zeros(2); eⱼ[j] = 1.0
            x_plus, _ = solve(_f_sp, lmo_sp, x0_sp, θ_sp .+ ε .* eⱼ; grad=_∇f_sp!, sp_kw...)
            x_minus, _ = solve(_f_sp, lmo_sp, x0_sp, θ_sp .- ε .* eⱼ; grad=_∇f_sp!, sp_kw...)
            dθ_fd[j] = (outer_loss_sp(x_plus) - outer_loss_sp(x_minus)) / (2ε)
        end
        @test isapprox(dθ_manual, dθ_fd; atol=0.15)
    end

    @testset "bilevel with Spectraplex ignores antisymmetric parameter directions" begin
        n_sp = 2
        lmo_sp = Spectraplex(n_sp)
        x0_sp = vec(Matrix(1.0I, n_sp, n_sp) ./ n_sp)
        θ_sp = [1.0]
        sp_kw = (; max_iters=5000, tol=1e-6)

        _f_sp(x, θ) = 0.5 * dot(x, x) - θ[1] * (x[2] - x[3])
        function _∇f_sp!(g, x, θ)
            g .= x
            g[2] -= θ[1]
            g[3] += θ[1]
        end

        outer_loss_sp(x) = x[2] - x[3]

        dθ_manual = bilevel_gradient(outer_loss_sp, _f_sp, lmo_sp, x0_sp, θ_sp;
                                     grad=_∇f_sp!, sp_kw...)
        dθ_auto = bilevel_gradient(outer_loss_sp, _f_sp, lmo_sp, x0_sp, θ_sp; sp_kw...)

        @test dθ_manual ≈ zeros(1) atol=1e-8
        @test dθ_auto ≈ zeros(1) atol=1e-8
    end

    @testset "bilevel convergence with ParametricBox" begin
        n_box = 2
        _f_conv(x, θ) = 0.5 * dot(x, x) - dot(θ[1:n_box], x)
        _∇f_conv!(g, x, θ) = (g .= x .- θ[1:n_box])

        # Parameterize lower bound; upper bound fixed at 1
        plmo = ParametricBox(θ -> θ[n_box+1:2n_box], θ -> ones(n_box))

        x_target_conv = [0.3, 0.7]
        outer_loss_conv(x) = sum((x .- x_target_conv).^2)

        # Start with suboptimal lower bounds
        θ = [x_target_conv; [0.0, 0.0]]
        x0_conv = [0.5, 0.5]
        conv_kw = (; max_iters=20_000, tol=1e-3)
        η = 0.05

        losses = Float64[]
        for k in 1:50
            x_bs, dθ, _ = bilevel_solve(outer_loss_conv, _f_conv, plmo, x0_conv, θ; grad=_∇f_conv!, conv_kw...)
            push!(losses, outer_loss_conv(x_bs))
            θ = θ .- η .* dθ
        end

        @test losses[end] < 1e-4
    end

    @testset "bilevel_gradient with ParametricWeightedSimplex and θ-dependent α matches FD" begin
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
        outer_loss = x -> dot(dx, x)
        kw = (; max_iters=50_000, tol=5e-6)

        dθ = bilevel_gradient(outer_loss, _f_ws_varα, plmo, x0, θ₀; grad=_∇f_ws_varα!, kw...)

        ε = 1e-6
        dθ_fd = zeros(length(θ₀))
        for j in eachindex(θ₀)
            eⱼ = zeros(length(θ₀)); eⱼ[j] = 1.0
            x_plus, _ = solve(_f_ws_varα, plmo, x0, θ₀ .+ ε .* eⱼ; grad=_∇f_ws_varα!, kw...)
            x_minus, _ = solve(_f_ws_varα, plmo, x0, θ₀ .- ε .* eⱼ; grad=_∇f_ws_varα!, kw...)
            dθ_fd[j] = (outer_loss(x_plus) - outer_loss(x_minus)) / (2ε)
        end

        @test isapprox(dθ, dθ_fd; atol=3e-2)
    end
end
