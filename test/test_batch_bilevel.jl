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

@testset "Batch Bilevel (exhaustive)" begin

    @testset "ScalarBox, B=$B" for B in [1, 2, 4, 8]
        n = 4
        H = Matrix{Float64}(3.0I, n, n)
        θ = randn(n)

        inner_f(x, t, b) = 0.5 * dot(x, H * x) - dot(t, x)
        inner_grad!(g, x, t, b) = (g .= H * x .- t; g)
        outer_f(x, _, b) = sum(x .^ 2)
        outer_grad!(g, x, _, b) = (g .= 2 .* x; g)
        inner = BatchedExpression(inner_f, inner_grad!)
        outer = BatchedExpression(outer_f, outer_grad!)

        lmo = Box(0.0, 1.0)
        X0 = fill(0.5, n, B)
        cfg = BatchSolveConfig(max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())

        X, dθ, cg = batch_bilevel_solve(outer, inner, lmo, X0, θ; config=cfg)
        @test length(dθ) == n
        @test all(c -> c.converged, cg)

        # Finite difference check
        ε = 1e-5
        dθ_fd = zeros(n)
        for j in 1:n
            θ_p = copy(θ); θ_p[j] += ε
            θ_m = copy(θ); θ_m[j] -= ε
            X_p, _ = batch_solve(inner, lmo, X0, θ_p; config=cfg)
            X_m, _ = batch_solve(inner, lmo, X0, θ_m; config=cfg)
            obj_p = sum(outer_f(view(X_p, :, b), nothing, b) for b in 1:B)
            obj_m = sum(outer_f(view(X_m, :, b), nothing, b) for b in 1:B)
            dθ_fd[j] = (obj_p - obj_m) / (2ε)
        end
        @test norm(dθ - dθ_fd) / max(1.0, norm(dθ_fd)) < 0.05
    end

    @testset "ProbSimplex with auto gradient" begin
        n = 3; B = 2
        H = [2.0 0.5 0.0; 0.5 2.0 0.5; 0.0 0.5 2.0]
        θ = [0.3, -0.2, 0.1]

        inner_f(x, t, b) = 0.5 * dot(x, H * x) - dot(t, x)
        inner_grad!(g, x, t, b) = (g .= H * x .- t; g)
        outer_f(x, _, b) = 0.5 * dot(x, x)
        # No outer grad — auto-diff path
        inner = BatchedExpression(inner_f, inner_grad!)
        outer = BatchedExpression(outer_f, nothing)

        lmo = ProbSimplex()
        X0 = fill(1.0 / n, n, B)
        cfg = BatchSolveConfig(max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())

        X, dθ, cg = batch_bilevel_solve(outer, inner, lmo, X0, θ; config=cfg)
        @test length(dθ) == n
        @test all(c -> c.converged, cg)
    end

    @testset "Manual cross_hvp on inner" begin
        n = 3; B = 2
        H = Matrix{Float64}(2.0I, n, n)
        θ = [0.5, -0.3, 0.2]

        inner_f(x, t, b) = 0.5 * dot(x, H * x) - dot(t, x)
        inner_grad!(g, x, t, b) = (g .= H * x .- t; g)
        # cross_hvp: -(∂²f/∂x∂θ)' u = u (since ∂∇ₓf/∂θ = -I)
        cross_hvp(out, x, t, u, b) = (out .= u; out)
        outer_f(x, _, b) = sum(x .^ 2)
        outer_grad!(g, x, _, b) = (g .= 2 .* x; g)

        inner_auto = BatchedExpression(inner_f, inner_grad!)
        inner_manual = BatchedExpression(inner_f, inner_grad!, cross_hvp)
        outer = BatchedExpression(outer_f, outer_grad!)

        lmo = ProbSimplex()
        X0 = fill(1.0 / n, n, B)
        cfg = BatchSolveConfig(max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())

        _, dθ_auto, _ = batch_bilevel_solve(outer, inner_auto, lmo, X0, θ; config=cfg)
        _, dθ_man, _ = batch_bilevel_solve(outer, inner_manual, lmo, X0, θ; config=cfg)
        @test norm(dθ_auto - dθ_man) < 1e-4
    end

    @testset "ParametricBox" begin
        n = 2; B = 2
        H = Matrix{Float64}(3.0I, n, n)
        θ = [0.1, 0.9]

        inner_f(x, _, b) = 0.5 * dot(x, H * x)
        inner_grad!(g, x, _, b) = (g .= H * x; g)
        outer_f(x, _, b) = sum(x)
        outer_grad!(g, x, _, b) = (fill!(g, 1.0); g)
        inner = BatchedExpression(inner_f, inner_grad!)
        outer = BatchedExpression(outer_f, outer_grad!)

        plmo = ParametricBox(θ -> zeros(n), θ -> θ)
        X0 = fill(0.5, n, B)
        cfg = BatchSolveConfig(max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())

        X, dθ, cg = batch_bilevel_solve(outer, inner, plmo, X0, θ; config=cfg)
        @test length(dθ) == n
    end
end
