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

@testset "Batch Diff (fast)" begin

    # Interior solution on ScalarBox — fast convergence, simple active set
    n = 3
    B = 2
    H = Matrix{Float64}(3.0I, n, n)
    θ = [0.3, 0.6, 0.9]  # x* = θ/3 ∈ (0, 1) — interior

    f_batch(X, θ) = [0.5 * dot(X[:, b], H * X[:, b]) - dot(θ, X[:, b]) for b in 1:B]
    grad_batch!(G, X, θ) = (G .= H * X .- θ)

    lmo = Box(0.0, 1.0)
    X0 = fill(0.5, n, B)
    kw = (; grad_batch=grad_batch!, max_iters=10000, tol=1e-6,
            step_rule=AdaptiveStepSize())

    @testset "rrule basic" begin
        (X_star, result), pb = rrule(batch_solve, f_batch, lmo, X0, θ; kw...)
        @test all(result.converged)

        dX = ones(n, B)
        tangents = pb(dX)
        @test length(tangents) == 5
        dθ = tangents[5]
        @test length(dθ) == n
        @test dθ isa AbstractVector
    end

    @testset "rrule finite difference check" begin
        (X_star, result), pb = rrule(batch_solve, f_batch, lmo, X0, θ; kw...)
        dX = randn(n, B)
        tangents = pb(dX)
        dθ = tangents[5]

        ε = 1e-5
        dθ_fd = zeros(n)
        for j in 1:n
            θ_p = copy(θ); θ_p[j] += ε
            θ_m = copy(θ); θ_m[j] -= ε
            X_p, _ = batch_solve(f_batch, lmo, X0, θ_p; kw...)
            X_m, _ = batch_solve(f_batch, lmo, X0, θ_m; kw...)
            dθ_fd[j] = sum(dX .* (X_p .- X_m)) / (2ε)
        end
        @test norm(dθ - dθ_fd) / max(1.0, norm(dθ_fd)) < 0.1
    end

    @testset "Zero tangent" begin
        (_, _), pb = rrule(batch_solve, f_batch, lmo, X0, θ; kw...)
        tangents = pb(NoTangent())
        @test all(t -> t isa NoTangent, tangents)
    end

    @testset "batch_solution_jacobian" begin
        J, sr = batch_solution_jacobian(f_batch, lmo, X0, θ; kw...)
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
