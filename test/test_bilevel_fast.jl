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
using ChainRulesCore: rrule

@testset "Bilevel Optimization (fast representative coverage)" begin

    # Coverage map:
    # - simplex bilevel gradients and bilevel_solve (manual + auto)
    # - custom-oracle active_set fallback/error behavior
    # - parametric-oracle bilevel gradients
    # - Spectraplex bilevel boundary sensitivity and symmetry cases
    # Exhaustive variants remain in test/test_bilevel.jl.

    Random.seed!(123)
    n = 5

    A = randn(n, n)
    H = A' * A + 0.5I

    _f(x, θ) = 0.5 * dot(x, H * x) - dot(θ, x)
    _∇f!(g, x, θ) = (g .= H * x .- θ)

    _f_id(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
    _∇f_id!(g, x, θ) = (g .= x .- θ)

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

    x_target = zeros(n)
    x_target[1] = 0.6
    x_target[2] = 0.3
    x_target[3] = 0.1
    outer_loss(x) = sum((x .- x_target).^2)

    function bilevel_step(θ)
        (x_star, _), pb = rrule(solve, _f, lmo, x0, θ; grad=_∇f!, solve_kw...)
        loss = sum((x_star .- x_target).^2)
        dθ = pb((2.0 .* (x_star .- x_target), nothing))[end]
        return x_star, loss, dθ
    end

    @testset "AD gradient matches finite differences" begin
        θ_test = [0.3, 0.25, 0.2, 0.15, 0.1]
        fd_kw = (; max_iters=10_000, tol=1e-8, step_rule=AdaptiveStepSize())

        (x_ad, _), pb = rrule(solve, _f_id, lmo, x0, θ_test; grad=_∇f_id!, fd_kw...)
        dθ_ad = pb((2.0 .* (x_ad .- x_target), nothing))[end]

        ε = 1e-5
        dθ_fd = zeros(n)
        for j in 1:n
            eⱼ = zeros(n)
            eⱼ[j] = 1.0
            x_plus, _ = solve(_f_id, lmo, x0, θ_test .+ ε .* eⱼ; grad=_∇f_id!, fd_kw...)
            x_minus, _ = solve(_f_id, lmo, x0, θ_test .- ε .* eⱼ; grad=_∇f_id!, fd_kw...)
            dθ_fd[j] = (outer_loss(x_plus) - outer_loss(x_minus)) / (2ε)
        end

        @test isapprox(dθ_ad, dθ_fd; atol=0.05)
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

        dθ_callable_bg = bilevel_gradient(
            outer_loss_plain, _f_id, callable_lmo, x0_plain, θ_plain; kw_plain...)
        @test isapprox(dθ_callable_bg, dθ_callable; atol=1e-8)

        x_ref, dθ_ref, _ = bilevel_solve(
            outer_loss_plain, _f_id, ProbabilitySimplex(), x0_plain, θ_plain; kw_plain...)
        @test isapprox(x_callable, x_ref; atol=1e-8)
        @test isapprox(dθ_callable, dθ_ref; atol=0.1)
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
        dθ_fd = zeros(length(θ_fd))
        for j in eachindex(θ_fd)
            eⱼ = zeros(length(θ_fd))
            eⱼ[j] = 1.0
            x_plus, _ = solve(_f_fd, plmo_fd, x0_fd, θ_fd .+ ε .* eⱼ; grad=_∇f_fd!, fd_kw...)
            x_minus, _ = solve(_f_fd, plmo_fd, x0_fd, θ_fd .- ε .* eⱼ; grad=_∇f_fd!, fd_kw...)
            dθ_fd[j] = (outer_loss_fd(x_plus) - outer_loss_fd(x_minus)) / (2ε)
        end

        @test isapprox(dθ_bg, dθ_fd; atol=0.05)
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

    @testset "bilevel cross_deriv= kwarg matches auto" begin
        n_cd = 5
        f_cd(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
        ∇f_cd!(g, x, θ) = (g .= x .- θ)
        outer_cd(x) = sum(x)
        # ∂²f/∂θ∂x = -I, so -(∂²f/∂θ∂x)ᵀ u = u
        cd_fn(u, θ) = copy(u)
        lmo_cd = ProbSimplex(1.0)
        x0_cd = fill(1.0/n_cd, n_cd)
        θ_cd = 0.2 .* ones(n_cd) .+ 0.01 .* (1:n_cd)
        kw_cd = (; max_iters=5000, tol=1e-10)

        dθ_auto = bilevel_gradient(outer_cd, f_cd, lmo_cd, x0_cd, θ_cd; grad=∇f_cd!, kw_cd...)
        dθ_cd = bilevel_gradient(outer_cd, f_cd, lmo_cd, x0_cd, θ_cd;
                                 grad=∇f_cd!, cross_deriv=cd_fn, kw_cd...)
        @test isapprox(dθ_auto, dθ_cd; atol=1e-8)
    end

end
