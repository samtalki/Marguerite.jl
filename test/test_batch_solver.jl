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
using KernelAbstractions: KernelAbstractions

@testset "Batch Solver" begin

    H = [4.0 1.0; 1.0 3.0]

    @testset "Basic correctness: ScalarBox" begin
        B = 4
        n = 2
        C = [0.5 -0.3 0.1 -0.2; -0.1 0.2 -0.4 0.3]
        f_per_col(x, _, b) = 0.5 * dot(x, H * x) + dot(view(C, :, b), x)
        grad_per_col!(g, x, _, b) = (g .= H * x .+ view(C, :, b); g)
        expr = BatchedExpression(f_per_col, grad_per_col!)

        X0 = fill(0.5, n, B)
        lmo = Box(0.0, 1.0)

        X, result = batch_solve(expr, lmo, X0; max_iters=10000, tol=1e-3)
        @test all(result.converged)
        @test size(X) == (n, B)

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
        f_per_col(x, _, b) = 0.5 * dot(x, H * x) + dot(view(C, :, b), x)
        grad_per_col!(g, x, _, b) = (g .= H * x .+ view(C, :, b); g)
        expr = BatchedExpression(f_per_col, grad_per_col!)

        X0 = fill(0.5, n, B)
        lmo = ProbSimplex()

        X, result = batch_solve(expr, lmo, X0; max_iters=10000, tol=1e-3)
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
        f_per_col(x, _, b) = 0.5 * sum(x .^ 2) + dot(view(C, :, b), x)
        grad_per_col!(g, x, _, b) = (g .= x .+ view(C, :, b); g)
        expr = BatchedExpression(f_per_col, grad_per_col!)

        X0 = zeros(n, B)
        lmo = Knapsack(2, n)

        X, result = batch_solve(expr, lmo, X0; max_iters=10000, tol=1e-3)
        @test all(result.converged)
        for b in 1:B
            @test sum(X[:, b]) <= 2.0 + 1e-4
            @test all(X[:, b] .>= -1e-8)
            @test all(X[:, b] .<= 1.0 + 1e-8)
        end
    end

    @testset "Convergence on symmetric quadratic" begin
        n = 2
        B = 2
        H_easy = [2.0 0.0; 0.0 2.0]
        f_per_col(x, _, b) = 0.5 * dot(x, H_easy * x)
        grad_per_col!(g, x, _, b) = (g .= H_easy * x; g)
        expr = BatchedExpression(f_per_col, grad_per_col!)

        X0 = fill(0.5, n, B)
        lmo = ProbSimplex()
        X, result = batch_solve(expr, lmo, X0; max_iters=10000, tol=1e-3)
        @test all(result.converged)
        @test norm(X[:, 1] - [0.5, 0.5]) < 0.1
    end

    @testset "Single-problem batch matches scalar solve" begin
        n = 3
        H3 = [3.0 0.5 0.0; 0.5 2.0 0.5; 0.0 0.5 1.0]
        f_per_col(x, _, b) = 0.5 * dot(x, H3 * x)
        grad_per_col!(g, x, _, b) = (g .= H3 * x; g)
        expr = BatchedExpression(f_per_col, grad_per_col!)
        gradf!(g, x) = (g .= H3 * x)

        x0 = [1/3, 1/3, 1/3]
        lmo = ProbSimplex()

        x_scalar, _ = solve(x -> 0.5 * dot(x, H3 * x), lmo, x0; grad=gradf!, max_iters=5000, tol=1e-4)
        X_batch, _ = batch_solve(expr, lmo, reshape(x0, 3, 1); max_iters=5000, tol=1e-4)

        @test norm(X_batch[:, 1] - x_scalar) < 1e-3
    end

    @testset "Zero iterations" begin
        n = 2; B = 2
        X0 = fill(0.5, n, B)
        f_per_col(x, _, b) = sum(x)
        grad_per_col!(g, x, _, b) = (fill!(g, 1.0); g)
        expr = BatchedExpression(f_per_col, grad_per_col!)

        X, result = batch_solve(expr, Box(0.0, 1.0), X0; max_iters=0)
        @test result.iterations == 0
        @test X ≈ X0
    end

    @testset "Tuple unpacking" begin
        n = 2; B = 2
        X0 = fill(0.5, n, B)
        f_per_col(x, _, b) = 0.5 * sum(x .^ 2)
        grad_per_col!(g, x, _, b) = (g .= x; g)
        expr = BatchedExpression(f_per_col, grad_per_col!)

        X, result = batch_solve(expr, Box(0.0, 1.0), X0; max_iters=100)
        @test X isa Matrix
        @test result isa BatchResult
    end

    @testset "BatchSolveConfig overrides via kwargs" begin
        n = 2; B = 2
        X0 = fill(0.5, n, B)
        f_per_col(x, _, b) = 0.5 * sum(x .^ 2)
        grad_per_col!(g, x, _, b) = (g .= x; g)
        expr = BatchedExpression(f_per_col, grad_per_col!)

        cfg = BatchSolveConfig(max_iters=10, tol=1e-1)
        X, result = batch_solve(expr, Box(0.0, 1.0), X0; config=cfg)
        @test result.iterations <= 10

        # per-call kwarg should override config
        X2, result2 = batch_solve(expr, Box(0.0, 1.0), X0; config=cfg, max_iters=5)
        @test result2.iterations <= 5
    end

    @testset "BatchCache reuse resets state" begin
        n = 3; B = 2
        H_easy = [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0]
        f_per_col(x, _, b) = 0.5 * dot(x, H_easy * x)
        grad_per_col!(g, x, _, b) = (g .= H_easy * x; g)
        expr = BatchedExpression(f_per_col, grad_per_col!)

        X0 = fill(1.0/n, n, B)
        cache = BatchCache(X0)

        X1, r1 = batch_solve(expr, ProbSimplex(), X0; cache=cache, max_iters=200, tol=1e-3)
        # second run with same cache: state should be reset
        X2, r2 = batch_solve(expr, ProbSimplex(), X0; cache=cache, max_iters=200, tol=1e-3)
        @test all(r1.converged)
        @test all(r2.converged)
        @test all(cache.active .== false)  # both runs end with all problems converged
        @test all(cache.discards .>= 0)
        @test X1 ≈ X2
    end

    @testset "Cache validation" begin
        n = 2; B = 2
        f_per_col(x, _, b) = sum(x)
        grad_per_col!(g, x, _, b) = (fill!(g, 1.0); g)
        expr = BatchedExpression(f_per_col, grad_per_col!)
        X0 = fill(0.5, n, B)

        # Matching shape and backend: no error
        cache = BatchCache(X0)
        X, _ = batch_solve(expr, Box(0.0, 1.0), X0; cache=cache, max_iters=10)
        @test KernelAbstractions.get_backend(cache.gradient) === KernelAbstractions.get_backend(X)

        # Wrong n: cache built for (n+2, B)
        cache_wrong_n = BatchCache(zeros(n + 2, B))
        @test_throws DimensionMismatch batch_solve(expr, Box(0.0, 1.0), X0;
                                                    cache=cache_wrong_n, max_iters=10)

        # Wrong B: cache built for (n, B+1)
        cache_wrong_B = BatchCache(zeros(n, B + 1))
        @test_throws DimensionMismatch batch_solve(expr, Box(0.0, 1.0), X0;
                                                    cache=cache_wrong_B, max_iters=10)
    end

    @testset "step_rule plumbing" begin
        H_easy = [2.0 0.0; 0.0 2.0]
        f_per_col(x, _, b) = 0.5 * dot(x, H_easy * x)
        grad_per_col!(g, x, _, b) = (g .= H_easy * x; g)
        expr = BatchedExpression(f_per_col, grad_per_col!)
        # Start at vertices so the FW gap is nonzero and the step rule is exercised
        X0 = [1.0 0.0; 0.0 1.0]

        # Custom callable: confirm batch_solve actually invokes it
        seen = Int[]
        custom_rule = t -> (push!(seen, t); 0.1 / (t + 1))
        batch_solve(expr, ProbSimplex(), X0; step_rule=custom_rule, max_iters=5, tol=1e-12)
        @test !isempty(seen)
        @test seen == collect(0:length(seen)-1)

        # Default vs explicit MonotonicStepSize: same trajectory
        X1, _ = batch_solve(expr, ProbSimplex(), X0; max_iters=20, tol=1e-12)
        X2, _ = batch_solve(expr, ProbSimplex(), X0;
                            step_rule=MonotonicStepSize(), max_iters=20, tol=1e-12)
        @test X1 ≈ X2
    end

    @testset "Float32 forward solves" begin
        @testset "ScalarBox (F32)" begin
            B = 4; n = 2
            H32 = Float32[4.0 1.0; 1.0 3.0]
            C32 = Float32[0.5 -0.3 0.1 -0.2; -0.1 0.2 -0.4 0.3]
            f_per_col(x, _, b) = 0.5f0 * dot(x, H32 * x) + dot(view(C32, :, b), x)
            grad_per_col!(g, x, _, b) = (g .= H32 * x .+ view(C32, :, b); g)
            expr = BatchedExpression(f_per_col, grad_per_col!)

            X0 = fill(0.5f0, n, B)
            X, result = batch_solve(expr, Box(0.0f0, 1.0f0), X0;
                                    max_iters=10000, tol=1.0f-3)
            @test eltype(X) === Float32
            @test all(result.converged)
        end

        @testset "ProbSimplex (F32)" begin
            B = 3; n = 2
            H32 = Float32[4.0 1.0; 1.0 3.0]
            C32 = Float32[0.1 -0.2 0.3; -0.1 0.1 -0.1]
            f_per_col(x, _, b) = 0.5f0 * dot(x, H32 * x) + dot(view(C32, :, b), x)
            grad_per_col!(g, x, _, b) = (g .= H32 * x .+ view(C32, :, b); g)
            expr = BatchedExpression(f_per_col, grad_per_col!)

            X0 = fill(0.5f0, n, B)
            X, result = batch_solve(expr, ProbSimplex(), X0;
                                    max_iters=10000, tol=1.0f-3)
            @test eltype(X) === Float32
            @test all(result.converged)
            for b in 1:B
                @test sum(X[:, b]) ≈ 1.0f0 atol=1.0f-3
            end
        end
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
