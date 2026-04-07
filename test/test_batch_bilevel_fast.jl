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

@testset "Batch Bilevel (fast)" begin

    n = 3
    B = 2
    H = Matrix{Float64}(2.0I, n, n)
    θ = [0.5, -0.3, 0.2]

    inner_batch(X, θ) = [0.5 * dot(X[:, b], H * X[:, b]) - dot(θ, X[:, b]) for b in 1:B]
    grad_batch!(G, X, θ) = (G .= H * X .- θ)
    outer_batch(X) = [sum(X[:, b] .^ 2) for b in 1:B]

    lmo = ProbSimplex()
    X0 = fill(1.0 / n, n, B)

    @testset "Basic correctness" begin
        X, dθ, cg_results = batch_bilevel_solve(outer_batch, inner_batch, lmo, X0, θ;
                                                  grad_batch=grad_batch!,
                                                  max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())
        @test size(X) == (n, B)
        @test length(dθ) == n
        @test length(cg_results) == B
        @test all(c -> c.converged, cg_results)

        # Compare against B independent scalar bilevel_solve calls
        dθ_ref = zeros(n)
        for b in 1:B
            x0_b = X0[:, b]
            outer_b(x) = sum(x .^ 2)
            inner_b(x, θ_) = 0.5 * dot(x, H * x) - dot(θ_, x)
            grad_b!(g, x, θ_) = (g .= H * x .- θ_)
            _, dθ_b, _ = bilevel_solve(outer_b, inner_b, lmo, x0_b, θ;
                                        grad=grad_b!, max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())
            dθ_ref .+= dθ_b
        end
        @test norm(dθ - dθ_ref) < 1e-3
    end

    @testset "batch_bilevel_gradient convenience" begin
        dθ = batch_bilevel_gradient(outer_batch, inner_batch, lmo, X0, θ;
                                     grad_batch=grad_batch!, max_iters=5000, tol=1e-6, step_rule=AdaptiveStepSize())
        @test length(dθ) == n
    end

    @testset "Tuple unpacking" begin
        result = batch_bilevel_solve(outer_batch, inner_batch, lmo, X0, θ;
                                      grad_batch=grad_batch!, max_iters=1000)
        X, dθ, cg = result
        @test X isa Matrix
        @test dθ isa AbstractVector
        @test cg isa Vector{<:CGResult}
    end

    @testset "Show methods" begin
        X, dθ, cg = batch_bilevel_solve(outer_batch, inner_batch, lmo, X0, θ;
                                          grad_batch=grad_batch!, max_iters=100)
        result = BatchBilevelResult(X, dθ, cg)
        s = sprint(show, result)
        @test contains(s, "BatchBilevelResult")
        sp = sprint(show, MIME("text/plain"), result)
        @test contains(sp, "CG:")
    end
end
