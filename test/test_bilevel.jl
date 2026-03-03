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
    solve_kw = (; max_iters=5000, tol=1e-4)

    # Target on the simplex
    x_target = zeros(n)
    x_target[1] = 0.6; x_target[2] = 0.3; x_target[3] = 0.1

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
        fd_kw = (; max_iters=20_000, tol=1e-6)

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
        outer_loss(x) = sum((x .- x_target).^2)

        θ_test = H * x_target
        x_bs, θ̄_bs, cg_bs = bilevel_solve(outer_loss, _f, _∇f!, lmo, x0, θ_test; solve_kw...)
        @test cg_bs.converged

        x_rrule, _, θ̄_rrule = bilevel_step(θ_test)
        @test isapprox(x_bs, x_rrule; atol=1e-6)
        @test isapprox(θ̄_bs, θ̄_rrule; atol=1e-4)
    end

    @testset "bilevel_solve (auto gradient)" begin
        outer_loss(x) = sum((x .- x_target).^2)

        θ_test = H * x_target
        x_bs, θ̄_bs, _ = bilevel_solve(outer_loss, _f, lmo, x0, θ_test; solve_kw...)

        x_manual, θ̄_manual, _ = bilevel_solve(outer_loss, _f, _∇f!, lmo, x0, θ_test; solve_kw...)
        @test isapprox(x_bs, x_manual; atol=1e-6)
        @test isapprox(θ̄_bs, θ̄_manual; atol=1e-4)
    end

    @testset "bilevel_gradient matches finite differences" begin
        θ_test = [0.3, 0.25, 0.2, 0.15, 0.1]
        outer_loss(x) = sum((x .- x_target).^2)
        fd_kw = (; max_iters=20_000, tol=1e-6)

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
        outer_loss(x) = sum((x .- x_target).^2)
        θ_test = [0.3, 0.25, 0.2, 0.15, 0.1]
        fd_kw = (; max_iters=5000, tol=1e-4)
        θ̄_manual = bilevel_gradient(outer_loss, _f_id, _∇f_id!, lmo, x0, θ_test; fd_kw...)
        θ̄_auto = bilevel_gradient(outer_loss, _f_id, lmo, x0, θ_test; fd_kw...)
        @test isapprox(θ̄_auto, θ̄_manual; atol=1e-4)
    end

    @testset "bilevel_solve with default backends" begin
        outer_loss_sm(x) = sum((x .- [0.6, 0.4]).^2)
        default_kw = (; max_iters=5000, tol=1e-4)

        x_def, θ̄_def, cg_def = bilevel_solve(outer_loss_sm, _f_id, _∇f_id!, lmo,
                                               fill(0.5, 2), [0.7, 0.3]; default_kw...)
        @test cg_def.converged
        @test all(isfinite, θ̄_def)
    end

    @testset "bilevel_solve with custom CG params" begin
        outer_loss(x) = sum((x .- x_target).^2)
        θ_test = H * x_target

        x_bs, θ̄_bs, cg_bs = bilevel_solve(outer_loss, _f, _∇f!, lmo, x0, θ_test;
                                    solve_kw..., diff_cg_maxiter=100, diff_cg_tol=1e-8, diff_λ=1e-3)
        @test cg_bs.converged
        @test all(isfinite, θ̄_bs)
        @test length(θ̄_bs) == n
    end
end
