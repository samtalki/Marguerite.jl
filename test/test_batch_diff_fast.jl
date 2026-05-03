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
using LinearAlgebra: dot, norm, I
using ChainRulesCore: rrule, NoTangent, Tangent

@testset "Batch Diff (fast)" begin

    # Interior solution on Box — fast convergence, simple active set
    n = 3
    B = 2
    H = Matrix{Float64}(3.0I, n, n)
    θ = [0.3, 0.6, 0.9]  # x* = θ/3 ∈ (0, 1) — interior

    f_per_col(x, t, b) = 0.5 * dot(x, H * x) - dot(t, x)
    grad_per_col!(g, x, t, b) = (g .= H * x .- t; g)
    expr = BatchedExpression(f_per_col, grad_per_col!)

    lmo = Box(0.0, 1.0)
    X0 = fill(0.5, n, B)
    cfg = BatchSolveConfig(max_iters=10000, tol=1e-6, step_rule=AdaptiveStepSize())

    @testset "rrule basic" begin
        (X_star, result), pb = rrule(batch_solve, expr, lmo, X0, θ; config=cfg)
        @test all(result.converged)

        dX = ones(n, B)
        tangents = pb(dX)
        @test length(tangents) == 5
        dθ = tangents[5]
        @test length(dθ) == n
        @test dθ isa AbstractVector
    end

    @testset "rrule finite difference check" begin
        (X_star, result), pb = rrule(batch_solve, expr, lmo, X0, θ; config=cfg)
        dX = randn(n, B)
        tangents = pb(dX)
        dθ = tangents[5]

        ε = 1e-5
        dθ_fd = zeros(n)
        for j in 1:n
            θ_p = copy(θ); θ_p[j] += ε
            θ_m = copy(θ); θ_m[j] -= ε
            X_p, _ = batch_solve(expr, lmo, X0, θ_p; config=cfg)
            X_m, _ = batch_solve(expr, lmo, X0, θ_m; config=cfg)
            dθ_fd[j] = sum(dX .* (X_p .- X_m)) / (2ε)
        end
        @test norm(dθ - dθ_fd) / max(1.0, norm(dθ_fd)) < 0.1
    end

    @testset "Zero tangent" begin
        (_, _), pb = rrule(batch_solve, expr, lmo, X0, θ; config=cfg)
        tangents = pb(NoTangent())
        @test all(t -> t isa NoTangent, tangents)
    end

    @testset "Tangent cotangent" begin
        (X_star, result), pb = rrule(batch_solve, expr, lmo, X0, θ; config=cfg)
        dX = randn(n, B)
        # raw matrix path
        dθ_raw = pb(dX)[5]
        # Tangent path — what real AD frameworks send
        BSR = typeof(BatchSolveResult(X_star, result))
        dθ_tan = pb(Tangent{BSR}(X=dX))[5]
        @test isapprox(dθ_raw, dθ_tan; atol=1e-12)
    end

    @testset "Boundary Box+θ" begin
        # θ chosen so x* sits on the upper bound for some entries.
        # x* = clamp(θ/3, 0, 1). With θ_j = 4.0, j=1: x*[1] = 1.0 (upper).
        n2 = 3; B2 = 2
        H2 = Matrix{Float64}(3.0I, n2, n2)
        θ_bnd = [4.0, 1.5, -2.0]  # x* = (1, 0.5, 0)
        f_b(x, t, b) = 0.5 * dot(x, H2 * x) - dot(t, x)
        g_b!(g, x, t, b) = (g .= H2 * x .- t; g)
        expr_b = BatchedExpression(f_b, g_b!)
        X0b = fill(0.5, n2, B2)
        lmob = Box(0.0, 1.0)

        (X_star, result), pb = rrule(batch_solve, expr_b, lmob, X0b, θ_bnd;
                                      config=BatchSolveConfig(max_iters=10000, tol=1e-7,
                                                              step_rule=AdaptiveStepSize()))
        @test all(result.converged)
        # x*[1] = 1.0 (active upper), x*[3] = 0.0 (active lower)
        @test isapprox(X_star[1, 1], 1.0; atol=1e-3)
        @test isapprox(X_star[3, 1], 0.0; atol=1e-3)

        dθ = pb(randn(n2, B2))[5]
        @test all(isfinite, dθ)
        # At a boundary, the cotangent should null out along directions blocked
        # by an active constraint. Here θ_1 only acts on x*[1] which is pinned,
        # so dx_1 along θ_1 perturbations is ≈ 0 → dθ[1] should be small.
        @test abs(dθ[1]) < 0.1 * sum(abs, dθ)
    end

    @testset "batch_solution_jacobian" begin
        J, sr = batch_solution_jacobian(expr, lmo, X0, θ; config=cfg)
        @test size(J) == (n * B, n)
        @test sr isa BatchSolveResult

        for b in 1:B
            f_b(x, θ_) = 0.5 * dot(x, H * x) - dot(θ_, x)
            grad_b!(g, x, θ_) = (g .= H * x .- θ_)
            J_b, _ = solution_jacobian(f_b, lmo, X0[:, b], θ;
                                        grad=grad_b!, max_iters=10000, tol=1e-6,
                                        step_rule=AdaptiveStepSize())
            J_batch_b = J[(b-1)*n+1 : b*n, :]
            @test norm(J_b - J_batch_b) < 1e-2
        end
    end
end
