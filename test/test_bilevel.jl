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
using Random
using ChainRulesCore: ChainRulesCore, rrule, NoTangent
import DifferentiationInterface as DI

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

    lmo = ProbabilitySimplex()
    x0 = fill(1.0 / n, n)
    solve_kw = (; max_iters=10_000, tol=1e-3)

    # Target on the simplex
    x_target = zeros(n)
    x_target[1] = 0.6; x_target[2] = 0.3; x_target[3] = 0.1
    outer_loss(x) = sum((x .- x_target).^2)

    # Bilevel step: solve inner, compute outer loss gradient via rrule
    function bilevel_step(θ)
        (x_star, result), pb = rrule(solve, _f, _∇f!, lmo, x0, θ; solve_kw...)
        loss = sum((x_star .- x_target).^2)
        x̄ = 2.0 .* (x_star .- x_target)
        tangents = pb((x̄, nothing))
        θ̄ = tangents[end]
        return x_star, loss, θ̄
    end

    @testset "Bilevel convergence" begin
        θ = H * x_target  # warm start
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
        x_final, _ = solve(_f, _∇f!, lmo, x0, θ; solve_kw...)
        @test isapprox(x_final, x_target; atol=1e-2)
    end

    @testset "AD gradient matches finite differences" begin
        θ_test = [0.3, 0.25, 0.2, 0.15, 0.1]
        fd_kw = (; max_iters=50_000, tol=1e-4)

        (x_ad, _), pb = rrule(solve, _f_id, _∇f_id!, lmo, x0, θ_test; fd_kw...)
        x̄ = 2.0 .* (x_ad .- x_target)
        θ̄_ad = pb((x̄, nothing))[end]

        ε = 1e-3
        θ̄_fd = zeros(n)
        for j in 1:n
            eⱼ = zeros(n); eⱼ[j] = 1.0
            x_plus, _ = solve(_f_id, _∇f_id!, lmo, x0, θ_test .+ ε .* eⱼ; fd_kw...)
            x_minus, _ = solve(_f_id, _∇f_id!, lmo, x0, θ_test .- ε .* eⱼ; fd_kw...)
            loss_plus = sum((x_plus .- x_target).^2)
            loss_minus = sum((x_minus .- x_target).^2)
            θ̄_fd[j] = (loss_plus - loss_minus) / (2ε)
        end

        @test isapprox(θ̄_ad, θ̄_fd; atol=0.1)
    end

    @testset "bilevel_solve (manual gradient)" begin
        θ_test = H * x_target
        x_bs, θ̄_bs, cg_bs = bilevel_solve(outer_loss, _f, _∇f!, lmo, x0, θ_test; solve_kw...)
        @test cg_bs.converged

        x_rrule, _, θ̄_rrule = bilevel_step(θ_test)
        @test isapprox(x_bs, x_rrule; atol=1e-6)
        @test isapprox(θ̄_bs, θ̄_rrule; atol=1e-4)
    end

    @testset "bilevel_solve (auto gradient)" begin
        θ_test = H * x_target
        x_bs, θ̄_bs, _ = bilevel_solve(outer_loss, _f, lmo, x0, θ_test; solve_kw...)

        x_manual, θ̄_manual, _ = bilevel_solve(outer_loss, _f, _∇f!, lmo, x0, θ_test; solve_kw...)
        @test isapprox(x_bs, x_manual; atol=1e-6)
        @test isapprox(θ̄_bs, θ̄_manual; atol=1e-4)
    end

    @testset "bilevel_gradient matches finite differences" begin
        θ_test = [0.3, 0.25, 0.2, 0.15, 0.1]
        fd_kw = (; max_iters=50_000, tol=1e-4)

        θ̄_bg = bilevel_gradient(outer_loss, _f_id, _∇f_id!, lmo, x0, θ_test; fd_kw...)

        ε = 1e-3
        θ̄_fd = zeros(n)
        for j in 1:n
            eⱼ = zeros(n); eⱼ[j] = 1.0
            x_plus, _ = solve(_f_id, _∇f_id!, lmo, x0, θ_test .+ ε .* eⱼ; fd_kw...)
            x_minus, _ = solve(_f_id, _∇f_id!, lmo, x0, θ_test .- ε .* eⱼ; fd_kw...)
            θ̄_fd[j] = (outer_loss(x_plus) - outer_loss(x_minus)) / (2ε)
        end

        @test isapprox(θ̄_bg, θ̄_fd; atol=0.1)
    end

    @testset "bilevel_gradient auto vs manual" begin
        θ_test = [0.3, 0.25, 0.2, 0.15, 0.1]
        fd_kw = (; max_iters=10_000, tol=1e-3)
        θ̄_manual = bilevel_gradient(outer_loss, _f_id, _∇f_id!, lmo, x0, θ_test; fd_kw...)
        θ̄_auto = bilevel_gradient(outer_loss, _f_id, lmo, x0, θ_test; fd_kw...)
        @test isapprox(θ̄_auto, θ̄_manual; atol=1e-4)
    end

    @testset "bilevel_solve with default backends" begin
        outer_loss_sm(x) = sum((x .- [0.6, 0.4]).^2)
        default_kw = (; max_iters=10_000, tol=1e-3)

        x_def, θ̄_def, cg_def = bilevel_solve(outer_loss_sm, _f_id, _∇f_id!, lmo,
                                               fill(0.5, 2), [0.7, 0.3]; default_kw...)
        @test cg_def.converged
        @test all(isfinite, θ̄_def)
    end

    @testset "bilevel_solve with custom CG params" begin
        θ_test = H * x_target

        x_bs, θ̄_bs, cg_bs = bilevel_solve(outer_loss, _f, _∇f!, lmo, x0, θ_test;
                                    solve_kw..., diff_cg_maxiter=100, diff_cg_tol=1e-8, diff_λ=1e-3)
        @test cg_bs.converged
        @test all(isfinite, θ̄_bs)
        @test length(θ̄_bs) == n
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

        x_bs, θ̄_bs, cg_bs = bilevel_solve(outer_loss_box, _f_box, _∇f_box!, plmo, x0_box, θ_box; box_kw...)
        @test all(isfinite, θ̄_bs)
        @test length(θ̄_bs) == 3n_box
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

        x_bs, θ̄_bs, cg_bs = bilevel_solve(outer_loss_box, _f_box_auto, plmo, x0_box, θ_box; box_kw...)
        @test all(isfinite, θ̄_bs)
        @test length(θ̄_bs) == 3n_box
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

        θ̄_manual = bilevel_gradient(outer_loss_bg, _f_bg, _∇f_bg!, plmo, x0_bg, θ_bg; bg_kw...)
        θ̄_auto = bilevel_gradient(outer_loss_bg, _f_bg, plmo, x0_bg, θ_bg; bg_kw...)
        @test isapprox(θ̄_auto, θ̄_manual; atol=1e-4)
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
        fd_kw = (; max_iters=50_000, tol=1e-4)

        θ̄_bg = bilevel_gradient(outer_loss_fd, _f_fd, _∇f_fd!, plmo_fd, x0_fd, θ_fd; fd_kw...)

        ε = 1e-4
        m_fd = length(θ_fd)
        θ̄_fd = zeros(m_fd)
        for j in 1:m_fd
            eⱼ = zeros(m_fd); eⱼ[j] = 1.0
            x_plus, _ = solve(_f_fd, _∇f_fd!, plmo_fd, x0_fd, θ_fd .+ ε .* eⱼ; fd_kw...)
            x_minus, _ = solve(_f_fd, _∇f_fd!, plmo_fd, x0_fd, θ_fd .- ε .* eⱼ; fd_kw...)
            θ̄_fd[j] = (outer_loss_fd(x_plus) - outer_loss_fd(x_minus)) / (2ε)
        end

        @test isapprox(θ̄_bg, θ̄_fd; atol=0.1)
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
            x_bs, θ̄, _ = bilevel_solve(outer_loss_conv, _f_conv, _∇f_conv!, plmo, x0_conv, θ; conv_kw...)
            push!(losses, outer_loss_conv(x_bs))
            θ = θ .- η .* θ̄
        end

        @test losses[end] < 1e-4
    end
end
