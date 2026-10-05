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
using LinearAlgebra
using Random

# Frank-Wolfe gap at x, computed from scratch with a dense vertex.
function _reference_gap(∇f!, lmo, x)
    g = similar(x)
    v = similar(x)
    ∇f!(g, x)
    lmo(v, g)
    return dot(g, x .- v)
end

@testset "Solver options" begin

    # Convex quadratic over the probability simplex, shared by the tests below.
    rng = Random.MersenneTwister(7)
    n = 20
    A = randn(rng, n, n)
    Q = A'A + 0.1I
    q = randn(rng, n)
    f(x) = 0.5 * dot(x, Q * x) + dot(q, x)
    ∇f!(g, x) = (mul!(g, Q, x); g .+= q; g)
    lmo = ProbSimplex()
    x0 = zeros(n); x0[1] = 1.0

    x_ref, res_ref = solve(f, lmo, x0; grad=∇f!, max_iters=100_000, tol=1e-12,
                           step_rule=AdaptiveStepSize(1.0))
    f_ref = res_ref.objective   # f_ref ≥ min f, so any valid lower bound is ≤ f_ref

    @testset "Callbacks and stopping" begin

        # The callback runs once per iteration and t counts completed iterations
        @testset "callback called once per iteration" begin
            for (max_iters, tol) in ((40, 0.0), (10_000, 1e-3))
                ts = Int[]
                cb = state -> (push!(ts, state.t); nothing)
                _, res = solve(f, lmo, x0; grad=∇f!, max_iters=max_iters, tol=tol, callback=cb)
                @test length(ts) == res.iterations
                @test ts == collect(1:res.iterations)
            end
            # Converged at the start point: no iteration, no call
            calls = Ref(0)
            xs = fill(1.0 / n, n)
            _, res0 = solve(x -> 0.0, lmo, xs; grad=(g, x) -> fill!(g, 0.0),
                            callback=_ -> (calls[] += 1; false))
            @test res0.iterations == 0
            @test calls[] == 0
            @test res0.converged
        end

        # The state carries the iterate, its objective and its gap
        @testset "callback state is consistent" begin
            ok = Ref(true)
            seen_reject = Ref(false)
            cb = function (state)
                ok[] &= state.obj ≈ f(state.x)
                ok[] &= isapprox(state.gap, _reference_gap(∇f!, lmo, state.x); rtol=1e-10, atol=1e-12)
                ok[] &= state.lower_bound ≤ state.obj
                ok[] &= state.elapsed ≥ 0
                ok[] &= 0 ≤ state.γ ≤ 1
                seen_reject[] |= !state.accepted
                return false
            end
            _, res = solve(f, lmo, x0; grad=∇f!, max_iters=200, tol=0.0, callback=cb)
            @test ok[]
            @test res.iterations == 200
            # Starting at a vertex, the open-loop step overshoots at least once
            @test seen_reject[] == (res.discards > 0)
        end

        # Returning true stops the run after the current iteration
        @testset "early stop via callback" begin
            x, res = solve(f, lmo, x0; grad=∇f!, max_iters=1000, tol=0.0,
                           callback=state -> state.t ≥ 5)
            @test res.iterations == 5
            @test !res.converged
            @test res.objective ≈ f(x)
            @test res.gap ≈ _reference_gap(∇f!, lmo, x) rtol=1e-10
        end

        # Only `true` stops; other return values are ignored
        @testset "non-Bool return values do not stop" begin
            _, res = solve(f, lmo, x0; grad=∇f!, max_iters=30, tol=0.0,
                           callback=_ -> nothing)
            @test res.iterations == 30
            _, res2 = solve(f, lmo, x0; grad=∇f!, max_iters=30, tol=0.0,
                            callback=_ -> 1)
            @test res2.iterations == 30
        end

        # A slow objective with a short time limit stops well before max_iters
        @testset "time_limit honoured" begin
            slow_f(x) = (Libc.systemsleep(0.002); f(x))
            limit = 0.05
            _, res = solve(slow_f, lmo, x0; grad=∇f!, max_iters=1_000_000, tol=0.0,
                           time_limit=limit)
            @test !res.converged
            @test res.iterations < 1_000_000
            @test res.elapsed ≥ limit
            @test res.elapsed < limit + 1.0
            # A zero limit stops before the first iteration
            _, res0 = solve(f, lmo, x0; grad=∇f!, max_iters=100, tol=0.0, time_limit=0)
            @test res0.iterations == 0
            @test res0.gap ≈ _reference_gap(∇f!, lmo, x0)
        end

        # rel_tol stops on gap ≤ rel_tol·|f| even when tol is zero
        @testset "rel_tol" begin
            x, res = solve(f, lmo, x0; grad=∇f!, max_iters=100_000, tol=0.0, rel_tol=0.05)
            @test res.converged
            @test res.gap ≤ 0.05 * abs(res.objective)
            @test res.gap ≈ _reference_gap(∇f!, lmo, x) rtol=1e-10
            _, res_tight = solve(f, lmo, x0; grad=∇f!, max_iters=100_000, tol=0.0, rel_tol=0.01)
            @test res_tight.converged
            @test res_tight.gap ≤ 0.01 * abs(res_tight.objective)
            @test res_tight.iterations > res.iterations
        end

        # On a max_iters exit the gap belongs to the returned iterate
        @testset "max_iters gap matches the returned x" begin
            for iters in (1, 2, 7, 50), mono in (true, false)
                x, res = solve(f, lmo, x0; grad=∇f!, max_iters=iters, tol=0.0, monotonic=mono)
                @test res.iterations == iters
                @test res.objective ≈ f(x)
                @test res.gap ≈ _reference_gap(∇f!, lmo, x) rtol=1e-10 atol=1e-14
            end
        end

        # The callback and stopping keywords also reach the parametric method
        @testset "parametric solve forwards the keywords" begin
            fθ(x, θ) = 0.5 * dot(x, Q * x) + dot(θ, x)
            ∇fθ!(g, x, θ) = (mul!(g, Q, x); g .+= θ; g)
            calls = Ref(0)
            _, res = solve(fθ, lmo, x0, q; grad=∇fθ!, max_iters=25, tol=0.0,
                           callback=_ -> (calls[] += 1; false))
            @test calls[] == res.iterations == 25
            _, res2 = solve(fθ, lmo, x0, q; grad=∇fθ!, tol=0.0, rel_tol=0.05)
            @test res2.converged
            @test res2.gap ≤ 0.05 * abs(res2.objective)
        end
    end

    @testset "Lower bound" begin

        # lower_bound ≤ objective, non-decreasing, and below the optimal value
        @testset "certificate on a convex quadratic over the simplex" begin
            for rule in (MonotonicStepSize(), AdaptiveStepSize(1.0))
                lbs = Float64[]
                objs = Float64[]
                cb = state -> (push!(lbs, state.lower_bound); push!(objs, state.obj); false)
                _, res = solve(f, lmo, x0; grad=∇f!, max_iters=300, tol=0.0,
                               step_rule=rule, callback=cb)
                @test all(lbs .≤ objs)
                @test issorted(lbs)
                @test all(lbs .≤ f_ref)
                @test res.lower_bound == lbs[end]
                @test res.lower_bound ≤ res.objective
                @test res.lower_bound ≥ res.objective - res.gap
                @test res.lower_bound ≤ f_ref
                @test res.elapsed > 0
            end
        end

        # The five-argument constructor fills the new fields
        @testset "Result five-argument constructor" begin
            r = Marguerite.Result(2.0, 0.5, 3, false, 1)
            @test r.lower_bound == 1.5
            @test r.elapsed == 0.0
            r_inf = Marguerite.Result(2.0, Inf, 0, false, 0)
            @test r_inf.lower_bound == -Inf
        end

        # Float32 iterates keep a Float32 lower bound
        @testset "Float32" begin
            Q32 = Float32.(Q); q32 = Float32.(q)
            f32(x) = 0.5f0 * dot(x, Q32 * x) + dot(q32, x)
            ∇f32!(g, x) = (mul!(g, Q32, x); g .+= q32; g)
            x32 = zeros(Float32, n); x32[1] = 1
            _, res = solve(f32, ProbSimplex(1.0f0), x32; grad=∇f32!, max_iters=100, tol=0.0)
            @test res.lower_bound isa Float32
            @test res.lower_bound ≤ res.objective
        end
    end
end
