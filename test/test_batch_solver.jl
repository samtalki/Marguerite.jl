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
using LinearAlgebra: dot, norm
using Random: Xoshiro

@testset "Batch Solver" begin

    # Shared QP: min 0.5 x'Hx + c'x  s.t. x in C
    H = [4.0 1.0; 1.0 3.0]

    @testset "Basic correctness: ScalarBox" begin
        B = 4
        n = 2
        C = [0.5 -0.3 0.1 -0.2; -0.1 0.2 -0.4 0.3]
        f_batch(X) = [0.5 * dot(X[:, b], H * X[:, b]) + dot(C[:, b], X[:, b]) for b in 1:B]
        grad_batch!(G, X) = (G .= H * X .+ C)

        X0 = fill(0.5, n, B)
        lmo = Box(0.0, 1.0)

        X, result = batch_solve(f_batch, lmo, X0; grad_batch=grad_batch!, max_iters=10000, tol=1e-3)
        @test all(result.converged)
        @test size(X) == (n, B)

        # Verify against scalar solve
        for b in 1:B
            fb(x) = 0.5 * dot(x, H * x) + dot(C[:, b], x)
            gradb!(g, x) = (g .= H * x .+ C[:, b])
            x_ref, _ = solve(fb, lmo, X0[:, b]; grad=gradb!, max_iters=10000, tol=1e-3)
            @test norm(X[:, b] - x_ref) < 1e-2
        end
    end

    @testset "Basic correctness: ProbSimplex" begin
        B = 3
        n = 2
        C = [0.1 -0.2 0.3; -0.1 0.1 -0.1]
        f_batch(X) = [0.5 * dot(X[:, b], H * X[:, b]) + dot(C[:, b], X[:, b]) for b in 1:B]
        grad_batch!(G, X) = (G .= H * X .+ C)

        X0 = fill(0.5, n, B)
        lmo = ProbSimplex()

        X, result = batch_solve(f_batch, lmo, X0; grad_batch=grad_batch!, max_iters=10000, tol=1e-3)
        @test all(result.converged)
        for b in 1:B
            @test sum(X[:, b]) ≈ 1.0 atol=1e-4
            @test all(X[:, b] .>= -1e-8)
        end
    end

    @testset "Basic correctness: Knapsack" begin
        B = 2
        n = 4
        C = [0.5 -0.3; -0.2 0.1; 0.4 -0.5; -0.1 0.2]
        f_batch(X) = [0.5 * sum(X[:, b] .^ 2) + dot(C[:, b], X[:, b]) for b in 1:B]
        grad_batch!(G, X) = (G .= X .+ C)

        X0 = zeros(n, B)
        lmo = Knapsack(2, n)

        X, result = batch_solve(f_batch, lmo, X0; grad_batch=grad_batch!, max_iters=10000, tol=1e-3)
        @test all(result.converged)
        for b in 1:B
            @test sum(X[:, b]) <= 2.0 + 1e-4
            @test all(X[:, b] .>= -1e-8)
            @test all(X[:, b] .<= 1.0 + 1e-8)
        end
    end

    @testset "Convergence masking" begin
        n = 2
        B = 2
        # Simple quadratic on ProbSimplex — both start at [0.5, 0.5]
        H_easy = [2.0 0.0; 0.0 2.0]
        f_batch(X) = [0.5 * dot(X[:, b], H_easy * X[:, b]) for b in 1:B]
        grad_batch!(G, X) = (G .= H_easy * X)

        X0 = fill(0.5, n, B)
        lmo = ProbSimplex()
        X, result = batch_solve(f_batch, lmo, X0; grad_batch=grad_batch!, max_iters=10000, tol=1e-3)

        @test all(result.converged)
        # Solutions should be near [0.5, 0.5] (symmetric quadratic on simplex)
        @test norm(X[:, 1] - [0.5, 0.5]) < 0.1
    end

    @testset "Single-problem batch matches scalar solve" begin
        n = 3
        H3 = [3.0 0.5 0.0; 0.5 2.0 0.5; 0.0 0.5 1.0]
        f_batch(X) = [0.5 * dot(X[:, 1], H3 * X[:, 1])]
        grad_batch!(G, X) = (G .= H3 * X)
        gradf!(g, x) = (g .= H3 * x)

        x0 = [1/3, 1/3, 1/3]
        lmo = ProbSimplex()

        x_scalar, _ = solve(x -> 0.5 * dot(x, H3 * x), lmo, x0; grad=gradf!, max_iters=5000, tol=1e-4)
        X_batch, _ = batch_solve(f_batch, lmo, reshape(x0, 3, 1); grad_batch=grad_batch!, max_iters=5000, tol=1e-4)

        @test norm(X_batch[:, 1] - x_scalar) < 1e-3
    end

    @testset "Zero iterations" begin
        n = 2
        B = 2
        X0 = fill(0.5, n, B)
        f_batch(X) = [sum(X[:, b]) for b in 1:B]
        grad_batch!(G, X) = fill!(G, 1.0)

        X, result = batch_solve(f_batch, Box(0.0, 1.0), X0; grad_batch=grad_batch!, max_iters=0)
        @test result.iterations == 0
        @test X ≈ X0
    end

    @testset "Tuple unpacking" begin
        n = 2
        B = 2
        X0 = fill(0.5, n, B)
        f_batch(X) = [0.5 * sum(X[:, b] .^ 2) for b in 1:B]
        grad_batch!(G, X) = (G .= X)

        X, result = batch_solve(f_batch, Box(0.0, 1.0), X0; grad_batch=grad_batch!, max_iters=100)
        @test X isa Matrix
        @test result isa BatchResult
    end

    @testset "grad_batch required" begin
        X0 = fill(0.5, 2, 2)
        @test_throws ArgumentError batch_solve(X -> [0.0, 0.0], Box(0.0, 1.0), X0)
    end

    @testset "Show methods" begin
        c = BatchCache{Float64}(3, 4)
        @test contains(sprint(show, c), "BatchCache")
        @test contains(sprint(show, c), "n=3")
        @test contains(sprint(show, c), "B=4")

        r = BatchResult([0.1, 0.2], [1e-5, 1e-4], 100, BitVector([true, true]), [0, 0])
        @test contains(sprint(show, r), "BatchResult")
        @test contains(sprint(show, r), "converged=2/2")

        sr = BatchSolveResult(ones(3, 2), r)
        @test contains(sprint(show, sr), "BatchSolveResult")
        sp = sprint(show, MIME("text/plain"), sr)
        @test contains(sp, "(3, 2)")
    end
end
