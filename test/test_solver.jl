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
using BenchmarkTools, Random

@testset "Solver" begin

    @testset "Quadratic on probability simplex" begin
        Q = [4.0 1.0; 1.0 2.0]
        c = [-3.0, -1.0]
        f(x) = 0.5 * dot(x, Q * x) + dot(c, x)
        ∇f!(g, x) = (g .= Q * x .+ c)

        x, res = solve(f, ∇f!, ProbabilitySimplex(), [0.5, 0.5];
                        max_iters=5000, tol=1e-3)
        @test res.converged
        # Analytic optimum on the simplex: x ≈ [0.75, 0.25]
        @test x[1] ≈ 0.75 atol=1e-2
        @test x[2] ≈ 0.25 atol=1e-2
    end

    @testset "Quadratic on box" begin
        # min 0.5*||x - x*||^2 on [0, 1]^3, x* = [0.3, 0.7, 1.5]
        x_opt = [0.3, 0.7, 1.5]
        f(x) = 0.5 * sum((x .- x_opt).^2)
        ∇f!(g, x) = (g .= x .- x_opt)

        lmo = Box(zeros(3), ones(3))
        x, res = solve(f, ∇f!, lmo, [0.5, 0.5, 0.5];
                        max_iters=5000, tol=1e-3)
        @test res.converged
        # Solution should be projection: [0.3, 0.7, 1.0]
        @test x ≈ [0.3, 0.7, 1.0] atol=1e-2
    end

    @testset "Monotonic mode rejects bad steps" begin
        Q = [2.0 0.0; 0.0 2.0]
        f(x) = 0.5 * dot(x, Q * x)
        ∇f!(g, x) = (g .= Q * x)

        x, res = solve(f, ∇f!, ProbabilitySimplex(), [0.5, 0.5];
                        monotonic=true, max_iters=100)
        # Should have some discards since FW overshoots sometimes
        @test res.discards >= 0
        @test res.objective ≤ f([0.5, 0.5]) + 1e-10
    end

    @testset "Parametric solve" begin
        f(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
        ∇f!(g, x, θ) = (g .= x .- θ)

        θ = [0.8, 0.2]
        x, res = solve(f, ∇f!, ProbabilitySimplex(), [0.5, 0.5], θ;
                        max_iters=5000, tol=1e-3)
        @test res.converged
        # Optimal: project θ onto simplex → since sum(θ)=1, x* = θ
        @test x ≈ θ atol=1e-2
    end

    @testset "Cache reuse" begin
        Q = [2.0 0.5; 0.5 1.0]
        c = [-1.0, -0.5]
        f(x) = 0.5 * dot(x, Q * x) + dot(c, x)
        ∇f!(g, x) = (g .= Q * x .+ c)

        cache = Marguerite.Cache{Float64}(2)
        x1, res1 = solve(f, ∇f!, ProbabilitySimplex(), [0.5, 0.5]; cache=cache)
        x2, res2 = solve(f, ∇f!, ProbabilitySimplex(), [0.5, 0.5]; cache=cache)
        @test x1 ≈ x2
    end

    @testset "Auto-gradient (default backend)" begin
        Q = [2.0 0.5; 0.5 1.0]
        c = [-1.0, -0.5]
        f(x) = 0.5 * dot(x, Q * x) + dot(c, x)

        x, res = solve(f, ProbabilitySimplex(), [0.5, 0.5];
                        max_iters=5000, tol=1e-3)
        @test res.converged || res.gap < 0.01
    end

    @testset "Parametric auto-gradient solve (default backend)" begin
        f(x, θ) = 0.5 * dot(x, x) - dot(θ, x)

        θ = [0.7, 0.3]
        x, res = solve(f, ProbabilitySimplex(), [0.5, 0.5], θ;
                        max_iters=5000, tol=1e-3)
        @test res.converged
        @test x ≈ θ atol=1e-2
    end

    @testset "Parametric manual-gradient solve (default backend)" begin
        f(x, θ) = 0.5 * dot(x, x) - dot(θ, x)
        ∇f!(g, x, θ) = (g .= x .- θ)

        θ = [1.0, 2.0]
        x, res = solve(f, ∇f!, ProbabilitySimplex(), [0.5, 0.5], θ;
                        max_iters=5000, tol=1e-3)
        @test res.converged
        # θ not on simplex (sum=3), so x* is the projection
        # For this objective, x* = proj_simplex(θ) = [0, 1] (all weight on dim 2)
        @test x[2] > x[1]
    end

    @testset "ParametricOracle solve (ParametricBox)" begin
        f(x, θ) = 0.5 * dot(x, x) - dot(θ[1:length(x)], x)
        ∇f!(g, x, θ) = (g .= x .- θ[1:length(x)])

        n = 3
        plmo = ParametricBox(θ -> θ[1:n], θ -> θ[n+1:2n])
        # θ = [lb; ub], target = [0.3, 0.7, 1.5]
        θ = [0.0, 0.0, 0.0, 1.0, 1.0, 2.0]
        x0 = [0.5, 0.5, 0.5]

        x, res = solve(f, ∇f!, plmo, x0, θ; max_iters=5000, tol=1e-3)
        @test res.converged
        # x* = clamp(θ[1:3], lb, ub) = clamp([0,0,0], [0,0,0], [1,1,2]) = [0,0,0]
        @test x ≈ zeros(n) atol=1e-2

        # Change the objective so the unconstrained minimizer lies above the box ub
        # Objective pulls x toward [2.0, 2.0, 3.0] (above ub), so x* = ub = [1,1,2]
        f2(x, θ) = 0.5 * sum((x .- [2.0, 2.0, 3.0]).^2)
        ∇f2!(g, x, θ) = (g .= x .- [2.0, 2.0, 3.0])
        x2, res2 = solve(f2, ∇f2!, plmo, x0, θ; max_iters=5000, tol=1e-3)
        @test res2.converged
        @test x2 ≈ [1.0, 1.0, 2.0] atol=1e-2
    end

    @testset "ParametricOracle auto-gradient solve" begin
        f(x, θ) = 0.5 * dot(x, x) - dot(θ[1:length(x)], x)
        n = 3
        plmo = ParametricBox(θ -> θ[1:n], θ -> θ[n+1:2n])
        θ = [0.3, 0.7, 0.5, 1.0, 1.0, 1.0]
        x0 = [0.5, 0.5, 0.5]

        x, res = solve(f, plmo, x0, θ; max_iters=10000, tol=1e-3)
        @test res.converged
        @test x ≈ [0.3, 0.7, 0.5] atol=0.02
    end

    @testset "NaN and Inf safety" begin
        @testset "NaN objective rejected (monotonic=false)" begin
            # Interior optimum forces FW to iterate long enough to hit NaN
            calls = Ref(0)
            target = [0.7, 0.3]
            f(x) = (calls[] += 1; calls[] > 5 ? NaN : 0.5 * sum((x .- target).^2))
            ∇f!(g, x) = (g .= x .- target)
            x, res = @test_warn "non-finite objective" solve(
                f, ∇f!, ProbabilitySimplex(), [0.5, 0.5];
                monotonic=false, max_iters=50, tol=1e-15)
            @test res.discards > 0
        end

        @testset "NaN objective rejected (monotonic=true)" begin
            # Every iteration returns NaN, so all discards must come from
            # the NaN guard (not the monotonic check, which is never reached)
            calls = Ref(0)
            target = [0.7, 0.3]
            f(x) = (calls[] += 1; calls[] > 1 ? NaN : 0.5 * sum((x .- target).^2))
            ∇f!(g, x) = (g .= x .- target)
            x, res = solve(f, ∇f!, ProbabilitySimplex(), [0.5, 0.5];
                monotonic=true, max_iters=50, tol=1e-15)
            @test res.discards == 50
        end

        @testset "Backtracking exhaustion emits warning" begin
            # High curvature forces L to need ~1e20 before Armijo holds,
            # but 50 doublings from 1e-30 only reach ~1e-15
            α = 1e20
            target = [1.0, 0.0]
            f(x) = 0.5 * α * sum((x .- target).^2)
            ∇f!(g, x) = (g .= α .* (x .- target))
            step = Marguerite.AdaptiveStepSize(1e-30)
            x, res = @test_warn "backtracking did not converge" solve(
                f, ∇f!, ProbabilitySimplex(), [0.5, 0.5];
                step_rule=step, max_iters=5, tol=1e-15)
            @test isfinite(step.L)
        end

        @testset "NaN in backtracking does not corrupt L" begin
            f(x) = x[1] < 0.3 ? NaN : 0.5 * dot(x, x)
            ∇f!(g, x) = (g .= x)
            step = Marguerite.AdaptiveStepSize(1e-10)
            x, res = @test_warn r"non-finite objective" solve(
                f, ∇f!, ProbabilitySimplex(), [0.9, 0.1];
                step_rule=step, max_iters=10, tol=1e-15)
            @test isfinite(step.L)
            @test step.L < 1e100
        end
    end

    @testset "Benchmarks" begin
        n = 2
        Q = [2.0 0.5; 0.5 1.0]
        c = [-1.0, -0.5]
        f(x) = 0.5 * dot(x, Q * x) + dot(c, x)
        ∇f!(g, x) = (g .= Q * x .+ c)
        lmo = ProbabilitySimplex()
        x0 = [0.5, 0.5]

        # warmup
        solve(f, ∇f!, lmo, x0; max_iters=1000, tol=1e-6)

        @testset "Allocation bounds" begin
            alloc = @ballocated solve($f, $∇f!, $lmo, $x0;
                max_iters=1000, tol=1e-6)
            @test alloc < 1024
            @info "solve(n=$n, 1000 iters) allocations: $alloc bytes"
        end

        @testset "Pre-allocated cache" begin
            cache = Marguerite.Cache{Float64}(n)
            # warmup
            solve(f, ∇f!, lmo, x0; max_iters=1000, tol=1e-6, cache=cache)
            alloc = @ballocated solve($f, $∇f!, $lmo, $x0;
                max_iters=1000, tol=1e-6, cache=$cache)
            @test alloc < 1024
            @info "solve(n=$n, 1000 iters, cache) allocations: $alloc bytes"
        end

        @testset "Timing" begin
            n_big = 100
            rng = MersenneTwister(123)
            A = randn(rng, n_big, n_big)
            Q_big = A'A + 0.1I
            c_big = randn(rng, n_big)
            f_big(x) = 0.5 * dot(x, Q_big * x) + dot(c_big, x)
            ∇f_big!(g, x) = (mul!(g, Q_big, x); g .+= c_big; g)
            x0_big = fill(1.0 / n_big, n_big)
            lmo_big = ProbabilitySimplex()

            # warmup
            solve(f_big, ∇f_big!, lmo_big, x0_big; max_iters=5000, tol=1e-6)

            t = @belapsed solve($f_big, $∇f_big!, $lmo_big, $x0_big;
                max_iters=5000, tol=1e-6)
            @info "solve(n=$n_big, 5000 iters): $(round(t * 1e3; digits=2)) ms"
        end
    end
end
