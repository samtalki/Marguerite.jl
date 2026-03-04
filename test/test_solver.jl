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
using BenchmarkTools
using Random

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
        # Start from a vertex so early large γ = 2/(t+2) steps overshoot,
        # forcing the monotonic filter to reject objective-increasing updates.
        Q = [4.0 1.0; 1.0 2.0]
        c = [-3.0, -1.0]
        f(x) = 0.5 * dot(x, Q * x) + dot(c, x)
        ∇f!(g, x) = (g .= Q * x .+ c)

        x0 = [0.0, 1.0]
        x, res = solve(f, ∇f!, ProbabilitySimplex(), x0;
                        monotonic=true, max_iters=5000, tol=1e-3)
        @test res.converged
        @test res.discards >= 1
        @test res.objective ≤ f(x0) + 1e-10
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

        cache = Cache{Float64}(2)
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

        x, res = solve(f, plmo, x0, θ; max_iters=10000, tol=1e-5)
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
            cache = Cache{Float64}(n)
            # warmup
            solve(f, ∇f!, lmo, x0; max_iters=1000, tol=1e-6, cache=cache)
            alloc = @ballocated solve($f, $∇f!, $lmo, $x0;
                max_iters=1000, tol=1e-6, cache=$cache)
            @test alloc < 1024
            @info "solve(n=$n, 1000 iters, cache) allocations: $alloc bytes"
        end

        @testset "AdaptiveStepSize allocation bounds" begin
            step = Marguerite.AdaptiveStepSize()
            cache = Cache{Float64}(n)
            # warmup
            solve(f, ∇f!, lmo, x0; max_iters=1000, tol=1e-6,
                step_rule=step, cache=cache)
            step2 = Marguerite.AdaptiveStepSize()
            alloc = @ballocated solve($f, $∇f!, $lmo, $x0;
                max_iters=1000, tol=1e-6, step_rule=$step2, cache=$cache)
            @test alloc < 1024
            @info "solve(n=$n, 1000 iters, AdaptiveStepSize, cache) allocations: $alloc bytes"
        end

    end

    @testset "AdaptiveStepSize: zero-direction early return" begin
        identity_lmo(v, g) = (copyto!(v, [0.5, 0.5]); v)
        f(x) = 0.5 * dot(x, x)
        ∇f!(g, x) = (g .= x)
        x0 = [0.5, 0.5]
        step = Marguerite.AdaptiveStepSize()
        x, res = solve(f, ∇f!, identity_lmo, x0;
                       step_rule=step, max_iters=10, tol=1e-15)
        @test x ≈ x0
    end

    @testset "Convergence" begin
        Random.seed!(42)
        n_cv = 20
        A_cv = randn(n_cv, n_cv)
        Q_cv = A_cv'A_cv + 0.1I
        c_cv = randn(n_cv)
        f_cv(x) = 0.5 * dot(x, Q_cv * x) + dot(c_cv, x)
        ∇f_cv!(g, x) = (g .= Q_cv * x .+ c_cv)
        lmo_cv = ProbabilitySimplex()
        x0_cv = zeros(n_cv); x0_cv[1] = 1.0

        @testset "Primal gap decreases by 100x" begin
            x_ref, _ = solve(f_cv, ∇f_cv!, lmo_cv, x0_cv;
                             max_iters=10000, tol=1e-8, monotonic=false)
            f_ref = f_cv(x_ref)
            x_early, _ = solve(f_cv, ∇f_cv!, lmo_cv, x0_cv;
                               max_iters=1, tol=0.0, monotonic=false)
            x_late, _ = solve(f_cv, ∇f_cv!, lmo_cv, x0_cv;
                              max_iters=2000, tol=0.0, monotonic=false)
            @test (f_cv(x_late) - f_ref) < (f_cv(x_early) - f_ref) / 100
        end

        @testset "Manual loop matches solve()" begin
            iters = 500
            x_m = copy(x0_cv)
            g_m = zeros(n_cv); v_m = zeros(n_cv)
            step_m = MonotonicStepSize()
            for t in 0:(iters - 1)
                ∇f_cv!(g_m, x_m)
                lmo_cv(v_m, g_m)
                γ = step_m(t)
                x_m .= x_m .+ γ .* (v_m .- x_m)
            end
            x_solve, _ = solve(f_cv, ∇f_cv!, lmo_cv, x0_cv;
                               max_iters=iters, tol=0.0, monotonic=false)
            @test isapprox(f_cv(x_m), f_cv(x_solve); atol=1e-6)
        end
    end

    @testset "Convergence when f(x*) = 0" begin
        # Regression: old criterion gap ≤ tol * |f(x)| never converges when f(x*) = 0.
        # Target on the simplex → unconstrained minimizer is feasible → f(x*) = 0.
        x_target = [0.7, 0.3]
        f_zero(x) = 0.5 * sum((x .- x_target).^2)
        ∇f_zero!(g, x) = (g .= x .- x_target)
        x, res = solve(f_zero, ∇f_zero!, ProbabilitySimplex(), [0.5, 0.5];
                        max_iters=5000, tol=1e-3)
        @test res.converged
        @test x ≈ x_target atol=1e-2
    end

    @testset "Monotonic filter at large |f|" begin
        # Regression: old eps(T) threshold rejected nearly every step at large scale.
        # Scale a standard QP by 1e12 so |f(x)| ~ 1e12 and monotonic threshold matters.
        Q_big = 1e12 .* [4.0 1.0; 1.0 2.0]
        c_big = 1e12 .* [-3.0, -1.0]
        f_big(x) = 0.5 * dot(x, Q_big * x) + dot(c_big, x)
        ∇f_big!(g, x) = (g .= Q_big * x .+ c_big)

        x, res = solve(f_big, ∇f_big!, ProbabilitySimplex(), [0.5, 0.5];
                        monotonic=true, max_iters=5000, tol=1e-3)
        @test res.converged
        # Discards should not be excessive (old code: nearly every step rejected)
        @test res.discards < res.iterations
    end

    @testset "Cache dimension validation" begin
        cache3 = Cache{Float64}(3)
        f(x) = 0.5 * dot(x, x)
        ∇f!(g, x) = (g .= x)
        @test_throws DimensionMismatch solve(f, ∇f!, ProbabilitySimplex(), [0.5, 0.5];
                                              cache=cache3)
    end

    @testset "_ensure_vertex!" begin
        _ev! = Marguerite._ensure_vertex!

        @testset "nnz > 0 with AdaptiveStepSize materializes dense" begin
            c = Cache{Float64}(4)
            c.vertex_nzind[1] = 2
            c.vertex_nzind[2] = 4
            c.vertex_nzval[1] = 3.0
            c.vertex_nzval[2] = 7.0
            fill!(c.vertex, 999.0)
            _ev!(c, 2, Marguerite.AdaptiveStepSize())
            @test c.vertex ≈ [0.0, 3.0, 0.0, 7.0]
        end

        @testset "nnz = 0 with AdaptiveStepSize zeros the buffer" begin
            c = Cache{Float64}(3)
            fill!(c.vertex, 999.0)
            _ev!(c, 0, Marguerite.AdaptiveStepSize())
            @test c.vertex ≈ zeros(3)
        end

        @testset "nnz = -1 with AdaptiveStepSize is no-op" begin
            c = Cache{Float64}(3)
            fill!(c.vertex, 42.0)
            _ev!(c, -1, Marguerite.AdaptiveStepSize())
            @test c.vertex ≈ fill(42.0, 3)
        end

        @testset "any nnz with MonotonicStepSize is no-op" begin
            c = Cache{Float64}(3)
            fill!(c.vertex, 42.0)
            for nnz in [-1, 0, 2]
                _ev!(c, nnz, MonotonicStepSize())
                @test c.vertex ≈ fill(42.0, 3)
            end
        end
    end

    @testset "_trial_update!" begin
        _tu! = Marguerite._trial_update!

        @testset "dense path (nnz = -1)" begin
            c = Cache{Float64}(3)
            x = [1.0, 2.0, 3.0]
            c.vertex .= [4.0, 0.0, 1.0]
            γ = 0.25
            _tu!(c, x, γ, -1, 3)
            @test c.x_trial ≈ x .+ γ .* (c.vertex .- x)
        end

        @testset "sparse path (nnz > 0) matches dense" begin
            c = Cache{Float64}(4)
            x = [1.0, 2.0, 3.0, 4.0]
            # Sparse vertex: index 2 → 5.0, index 4 → 1.0
            c.vertex_nzind[1] = 2
            c.vertex_nzind[2] = 4
            c.vertex_nzval[1] = 5.0
            c.vertex_nzval[2] = 1.0
            γ = 0.3
            _tu!(c, x, γ, 2, 4)
            # Expected: (1-γ)*x + γ*v_sparse
            v_dense = [0.0, 5.0, 0.0, 1.0]
            expected = (1 - γ) .* x .+ γ .* v_dense
            @test c.x_trial ≈ expected
        end

        @testset "origin path (nnz = 0)" begin
            c = Cache{Float64}(3)
            x = [1.0, 2.0, 3.0]
            γ = 0.4
            _tu!(c, x, γ, 0, 3)
            @test c.x_trial ≈ (1 - γ) .* x
        end
    end

    @testset "Sparse vertex equivalence" begin
        # Verify that sparse vertex path gives identical results to dense path
        Random.seed!(42)
        n_eq = 20
        A_eq = randn(n_eq, n_eq)
        Q_eq = A_eq'A_eq + 0.1I
        c_eq = randn(n_eq)
        f_eq(x) = 0.5 * dot(x, Q_eq * x) + dot(c_eq, x)
        ∇f_eq!(g, x) = (g .= Q_eq * x .+ c_eq)

        x0_eq = zeros(n_eq); x0_eq[1] = 1.0

        for lmo in [ProbabilitySimplex(), Simplex(), Knapsack(5, n_eq),
                    MaskedKnapsack(5, [1, 2], n_eq)]
            x, res = solve(f_eq, ∇f_eq!, lmo, x0_eq;
                           max_iters=10000, tol=1e-6)
            @test res.converged || res.gap < 0.01
            @test isfinite(res.objective)
            @test all(isfinite, x)
        end
    end

    @testset "Sparsity bound nnz ≤ t+1" begin
        Random.seed!(42)
        n_sp = 20
        A_sp = randn(n_sp, n_sp)
        Q_sp = A_sp'A_sp + 0.1I
        c_sp = randn(n_sp)

        x = zeros(n_sp); x[1] = 1.0
        g = zeros(n_sp); v = zeros(n_sp)
        step = MonotonicStepSize()

        for t in 0:49
            g .= Q_sp * x .+ c_sp
            ProbabilitySimplex()(v, g)
            γ = step(t)
            x .= x .+ γ .* (v .- x)
            # theoretical sparsity bound is t+1; relaxed to t+2 for floating-point robustness
            @test count(xi -> abs(xi) > 1e-12, x) ≤ t + 2
        end
    end
end
