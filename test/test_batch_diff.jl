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
using ChainRulesCore: rrule, NoTangent

@testset "Batch Diff (exhaustive)" begin

    @testset "rrule ScalarBox B=$B" for B in [1, 2, 4]
        n = 4
        H = Matrix{Float64}(3.0I, n, n)
        θ = randn(n)

        f_batch(X, θ) = [0.5 * dot(X[:, b], H * X[:, b]) - dot(θ, X[:, b]) for b in 1:B]
        grad_batch!(G, X, θ) = (G .= H * X .- θ)

        lmo = Box(0.0, 1.0)
        X0 = fill(0.5, n, B)

        (X_star, result), pb = rrule(batch_solve, f_batch, lmo, X0, θ;
                                      grad_batch=grad_batch!, max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())
        @test all(result.converged)

        dX = randn(n, B)
        tangents = pb(dX)
        dθ = tangents[5]

        # Finite difference check
        ε = 1e-5
        dθ_fd = zeros(n)
        for j in 1:n
            θ_p = copy(θ); θ_p[j] += ε
            θ_m = copy(θ); θ_m[j] -= ε
            X_p, _ = batch_solve(f_batch, lmo, X0, θ_p; grad_batch=grad_batch!, max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())
            X_m, _ = batch_solve(f_batch, lmo, X0, θ_m; grad_batch=grad_batch!, max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())
            dθ_fd[j] = sum(dX .* (X_p .- X_m)) / (2ε)
        end
        @test norm(dθ - dθ_fd) / max(1.0, norm(dθ_fd)) < 0.05
    end

    @testset "rrule ProbSimplex auto gradient" begin
        n = 3
        B = 2
        H = [2.0 0.5 0.0; 0.5 2.0 0.5; 0.0 0.5 2.0]
        θ = [0.3, -0.2, 0.1]

        f_batch(X, θ) = [0.5 * dot(X[:, b], H * X[:, b]) - dot(θ, X[:, b]) for b in 1:B]

        lmo = ProbSimplex()
        X0 = fill(1.0 / n, n, B)

        (X_star, result), pb = rrule(batch_solve, f_batch, lmo, X0, θ;
                                      max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())
        @test all(result.converged)

        dX = randn(n, B)
        tangents = pb(dX)
        dθ = tangents[5]
        @test length(dθ) == n
    end

    @testset "rrule consistency with scalar rrule" begin
        n = 3
        B = 2
        H = Matrix{Float64}(2.0I, n, n)
        θ = [0.5, -0.3, 0.2]

        f_batch(X, θ) = [0.5 * dot(X[:, b], H * X[:, b]) - dot(θ, X[:, b]) for b in 1:B]
        grad_batch!(G, X, θ) = (G .= H * X .- θ)

        lmo = ProbSimplex()
        X0 = fill(1.0 / n, n, B)

        (X_star_batch, _), pb_batch = rrule(batch_solve, f_batch, lmo, X0, θ;
                                             grad_batch=grad_batch!, max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())

        dX = randn(n, B)
        tangents_batch = pb_batch(dX)
        dθ_batch = tangents_batch[5]

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
        n = 2
        B = 2
        H = Matrix{Float64}(3.0I, n, n)
        θ = [0.1, 0.9]

        f_batch(X, θ) = [0.5 * dot(X[:, b], H * X[:, b]) for b in 1:B]
        grad_batch!(G, X, θ) = (G .= H * X)

        plmo = ParametricBox(θ -> zeros(n), θ -> θ)
        X0 = fill(0.3, n, B)

        (X_star, result), pb = rrule(batch_solve, f_batch, plmo, X0, θ;
                                      grad_batch=grad_batch!, max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())
        @test all(result.converged)

        dX = randn(n, B)
        tangents = pb(dX)
        dθ = tangents[5]
        @test length(dθ) == n
    end

    @testset "batch_solution_jacobian ScalarBox" begin
        n = 3
        B = 3
        H = Matrix{Float64}(2.0I, n, n)
        θ = randn(n)

        f_batch(X, θ) = [0.5 * dot(X[:, b], H * X[:, b]) - dot(θ, X[:, b]) for b in 1:B]
        grad_batch!(G, X, θ) = (G .= H * X .- θ)

        lmo = Box(0.0, 1.0)
        X0 = fill(0.5, n, B)

        J, sr = batch_solution_jacobian(f_batch, lmo, X0, θ;
                                          grad_batch=grad_batch!, max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())
        @test size(J) == (n * B, n)

        # Finite difference check on Jacobian
        ε = 1e-5
        J_fd = zeros(n * B, n)
        for j in 1:n
            θ_p = copy(θ); θ_p[j] += ε
            θ_m = copy(θ); θ_m[j] -= ε
            X_p, _ = batch_solve(f_batch, lmo, X0, θ_p; grad_batch=grad_batch!, max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())
            X_m, _ = batch_solve(f_batch, lmo, X0, θ_m; grad_batch=grad_batch!, max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())
            J_fd[:, j] = vec(X_p .- X_m) / (2ε)
        end
        @test norm(J - J_fd) / max(1.0, norm(J_fd)) < 0.05
    end
end
