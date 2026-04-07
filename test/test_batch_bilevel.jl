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

        inner_batch(X, θ) = [0.5 * dot(X[:, b], H * X[:, b]) - dot(θ, X[:, b]) for b in 1:B]
        grad_batch!(G, X, θ) = (G .= H * X .- θ)
        outer_batch(X) = [sum(X[:, b] .^ 2) for b in 1:B]

        lmo = Box(0.0, 1.0)
        X0 = fill(0.5, n, B)

        X, dθ, cg = batch_bilevel_solve(outer_batch, inner_batch, lmo, X0, θ;
                                          grad_batch=grad_batch!, max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())
        @test length(dθ) == n
        @test all(c -> c.converged, cg)

        # Finite difference check
        ε = 1e-5
        dθ_fd = zeros(n)
        for j in 1:n
            θ_p = copy(θ); θ_p[j] += ε
            θ_m = copy(θ); θ_m[j] -= ε
            X_p, _ = batch_solve(inner_batch, lmo, X0, θ_p; grad_batch=grad_batch!, max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())
            X_m, _ = batch_solve(inner_batch, lmo, X0, θ_m; grad_batch=grad_batch!, max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())
            obj_p = sum(outer_batch(X_p))
            obj_m = sum(outer_batch(X_m))
            dθ_fd[j] = (obj_p - obj_m) / (2ε)
        end
        @test norm(dθ - dθ_fd) / max(1.0, norm(dθ_fd)) < 0.05
    end

    @testset "ProbSimplex with auto gradient" begin
        n = 3
        B = 2
        H = [2.0 0.5 0.0; 0.5 2.0 0.5; 0.0 0.5 2.0]
        θ = [0.3, -0.2, 0.1]

        inner_batch(X, θ) = [0.5 * dot(X[:, b], H * X[:, b]) - dot(θ, X[:, b]) for b in 1:B]
        outer_batch(X) = [0.5 * dot(X[:, b], X[:, b]) for b in 1:B]

        lmo = ProbSimplex()
        X0 = fill(1.0 / n, n, B)

        X, dθ, cg = batch_bilevel_solve(outer_batch, inner_batch, lmo, X0, θ;
                                          max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())
        @test length(dθ) == n
        @test all(c -> c.converged, cg)
    end

    @testset "Manual cross_deriv_batch" begin
        n = 3
        B = 2
        H = Matrix{Float64}(2.0I, n, n)
        θ = [0.5, -0.3, 0.2]

        inner_batch(X, θ) = [0.5 * dot(X[:, b], H * X[:, b]) - dot(θ, X[:, b]) for b in 1:B]
        grad_batch!(G, X, θ) = (G .= H * X .- θ)
        outer_batch(X) = [sum(X[:, b] .^ 2) for b in 1:B]
        # Cross-derivative: -(∂²f/∂θ∂x)ᵀu = u (since ∂∇ₓf/∂θ = -I)
        cross_deriv(u, θ, b) = copy(u)

        lmo = ProbSimplex()
        X0 = fill(1.0 / n, n, B)

        X_auto, dθ_auto, _ = batch_bilevel_solve(outer_batch, inner_batch, lmo, X0, θ;
                                                    grad_batch=grad_batch!, max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())
        X_man, dθ_man, _ = batch_bilevel_solve(outer_batch, inner_batch, lmo, X0, θ;
                                                 grad_batch=grad_batch!, cross_deriv_batch=cross_deriv,
                                                 max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())
        @test norm(dθ_auto - dθ_man) < 1e-4
    end

    @testset "ParametricBox" begin
        n = 2
        B = 2
        H = Matrix{Float64}(3.0I, n, n)
        θ = [0.1, 0.9]

        inner_batch(X, θ) = [0.5 * dot(X[:, b], H * X[:, b]) for b in 1:B]
        grad_batch!(G, X, θ) = (G .= H * X)
        outer_batch(X) = [sum(X[:, b]) for b in 1:B]

        plmo = ParametricBox(θ -> zeros(n), θ -> θ)
        X0 = fill(0.5, n, B)

        X, dθ, cg = batch_bilevel_solve(outer_batch, inner_batch, plmo, X0, θ;
                                          grad_batch=grad_batch!, max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())
        @test length(dθ) == n
    end
end
