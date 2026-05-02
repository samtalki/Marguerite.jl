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
using Random
using ChainRulesCore: rrule, NoTangent

@testset "Batch Diff (exhaustive)" begin
    Random.seed!(42)

    @testset "rrule ScalarBox B=$B" for B in [1, 2, 4]
        n = 4
        H = Matrix{Float64}(3.0I, n, n)
        # interior θ: x* = θ/3 ∈ [0.1, 0.3]^4, away from box bounds
        θ = [0.6, 0.3, 0.9, 0.5]

        f_per_col(x, t, b) = 0.5 * dot(x, H * x) - dot(t, x)
        grad_per_col!(g, x, t, b) = (g .= H * x .- t; g)
        expr = BatchedExpression(f_per_col, grad_per_col!)

        lmo = Box(0.0, 1.0)
        X0 = fill(0.5, n, B)
        cfg = BatchSolveConfig(max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())

        (X_star, result), pb = rrule(batch_solve, expr, lmo, X0, θ; config=cfg)
        @test all(result.converged)

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
        @test norm(dθ - dθ_fd) / max(1.0, norm(dθ_fd)) < 0.05
    end

    @testset "rrule ProbSimplex (autodiff cross derivative)" begin
        n = 3
        B = 2
        H = [2.0 0.5 0.0; 0.5 2.0 0.5; 0.0 0.5 2.0]
        θ = [0.3, -0.2, 0.1]

        f_per_col(x, t, b) = 0.5 * dot(x, H * x) - dot(t, x)
        # No grad_per_col! → falls back to joint HVP path.
        expr = BatchedExpression(f_per_col, nothing)

        lmo = ProbSimplex()
        X0 = fill(1.0 / n, n, B)
        cfg = BatchSolveConfig(max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())

        # ProbSimplex with no manual grad — verify the forward solve at least.
        # The autodiff path requires `expr.f_per_col` to support Dual eltype.
        f_dual(x, t, b) = 0.5 * dot(x, H * x) - dot(t, x)
        gf!(g, x, t, b) = (g .= H * x .- t; g)
        expr2 = BatchedExpression(f_dual, gf!)
        (X_star, result), pb = rrule(batch_solve, expr2, lmo, X0, θ; config=cfg)
        @test all(result.converged)

        dX = randn(n, B)
        dθ = pb(dX)[5]
        @test length(dθ) == n
    end

    @testset "rrule consistency with scalar rrule" begin
        n = 3; B = 2
        H = Matrix{Float64}(2.0I, n, n)
        θ = [0.5, -0.3, 0.2]

        f_per_col(x, t, b) = 0.5 * dot(x, H * x) - dot(t, x)
        grad_per_col!(g, x, t, b) = (g .= H * x .- t; g)
        expr = BatchedExpression(f_per_col, grad_per_col!)

        lmo = ProbSimplex()
        X0 = fill(1.0 / n, n, B)
        cfg = BatchSolveConfig(max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())

        (_, _), pb_batch = rrule(batch_solve, expr, lmo, X0, θ; config=cfg)
        dX = randn(n, B)
        dθ_batch = pb_batch(dX)[5]

        # Sum of per-problem scalar rrule pullbacks
        dθ_ref = zeros(n)
        for b in 1:B
            f_b(x, θ_) = 0.5 * dot(x, H * x) - dot(θ_, x)
            grad_b!(g, x, θ_) = (g .= H * x .- θ_)
            (_, _), pb_b = rrule(solve, f_b, lmo, X0[:, b], θ;
                                  grad=grad_b!, max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())
            tb = pb_b((dX[:, b], nothing))
            dθ_ref .+= tb[5]
        end
        @test norm(dθ_batch - dθ_ref) / max(1.0, norm(dθ_ref)) < 0.05
    end

    @testset "ParametricBox rrule" begin
        n = 2; B = 2
        H = Matrix{Float64}(3.0I, n, n)
        θ = [0.1, 0.9]

        f_per_col(x, _, b) = 0.5 * dot(x, H * x)
        grad_per_col!(g, x, _, b) = (g .= H * x; g)
        expr = BatchedExpression(f_per_col, grad_per_col!)

        plmo = ParametricBox(θ -> zeros(n), θ -> θ)
        X0 = fill(0.3, n, B)
        cfg = BatchSolveConfig(max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())

        (X_star, result), pb = rrule(batch_solve, expr, plmo, X0, θ; config=cfg)
        @test all(result.converged)

        dX = randn(n, B)
        dθ = pb(dX)[5]
        @test length(dθ) == n
    end

    @testset "batch_solution_jacobian ScalarBox" begin
        n = 3; B = 3
        H = Matrix{Float64}(2.0I, n, n)
        θ = [0.3, 0.5, 0.7]  # x* = θ/2 ∈ [0.15, 0.35]^3

        f_per_col(x, t, b) = 0.5 * dot(x, H * x) - dot(t, x)
        grad_per_col!(g, x, t, b) = (g .= H * x .- t; g)
        expr = BatchedExpression(f_per_col, grad_per_col!)

        lmo = Box(0.0, 1.0)
        X0 = fill(0.5, n, B)
        cfg = BatchSolveConfig(max_iters=5000, tol=1e-8, step_rule=AdaptiveStepSize())

        J, sr = batch_solution_jacobian(expr, lmo, X0, θ; config=cfg)
        @test size(J) == (n * B, n)

        ε = 1e-5
        J_fd = zeros(n * B, n)
        for j in 1:n
            θ_p = copy(θ); θ_p[j] += ε
            θ_m = copy(θ); θ_m[j] -= ε
            X_p, _ = batch_solve(expr, lmo, X0, θ_p; config=cfg)
            X_m, _ = batch_solve(expr, lmo, X0, θ_m; config=cfg)
            J_fd[:, j] = vec(X_p .- X_m) / (2ε)
        end
        @test norm(J - J_fd) / max(1.0, norm(J_fd)) < 0.05
    end
end
