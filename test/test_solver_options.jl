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
using BenchmarkTools

# A user-defined oracle that opts in to the pairwise variant by defining
# away_vertex! and pairwise_max_step.
struct _WrappedKnapsack <: Marguerite.AbstractOracle
    inner::MaskedKnapsack
end
(w::_WrappedKnapsack)(v, g) = w.inner(v, g)
Marguerite.away_vertex!(w::_WrappedKnapsack, c::Cache, x, g) = away_vertex!(w.inner, c, x, g)
Marguerite.pairwise_max_step(w::_WrappedKnapsack, x, d) = pairwise_max_step(w.inner, x, d)

# A user oracle whose largest pairwise step is always tiny, so every pairwise
# iteration must fall back to a Frank-Wolfe step.
struct _TinyStepKnapsack <: Marguerite.AbstractOracle
    inner::MaskedKnapsack
end
(w::_TinyStepKnapsack)(v, g) = w.inner(v, g)
Marguerite.away_vertex!(w::_TinyStepKnapsack, c::Cache, x, g) = away_vertex!(w.inner, c, x, g)
Marguerite.pairwise_max_step(w::_TinyStepKnapsack, x, d) = min(pairwise_max_step(w.inner, x, d), 1e-12)

# A user oracle with an inexact away vertex: the FW vertex with its selected
# optional coordinate of largest gradient swapped for the unselected one of
# smallest gradient, so the pairwise gap is the small difference of the two.
struct _SwapAwayKnapsack <: Marguerite.AbstractOracle
    inner::MaskedKnapsack
end
(w::_SwapAwayKnapsack)(v, g) = w.inner(v, g)
function Marguerite.away_vertex!(w::_SwapAwayKnapsack, c::Cache, x, g)
    v = c.away_vertex
    w.inner(v, g)
    free = findall(.!w.inner.is_masked)
    sel = filter(e -> v[e] == 1, free)
    uns = filter(e -> v[e] == 0, free)
    v[sel[argmax(g[sel])]] = 0
    v[uns[argmin(g[uns])]] = 1
    return v
end
Marguerite.pairwise_max_step(w::_SwapAwayKnapsack, x, d) = pairwise_max_step(w.inner, x, d)

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

    @testset "ShortStep" begin

        # With L equal to the curvature along d, one short step is the exact line minimizer
        @testset "exact minimizer along d on a quadratic" begin
            H = [3.0 0.5 0.0; 0.5 2.0 0.3; 0.0 0.3 1.5]
            c = [0.2, -0.4, 0.1]
            fq(x) = 0.5 * dot(x, H * x) + dot(c, x)
            ∇fq!(g, x) = (mul!(g, H, x); g .+= c; g)
            xs = [1.0, 0.0, 0.0]
            g0 = H * xs .+ c
            v = zeros(3); v[argmin(g0)] = 1.0
            d = v .- xs
            L_d = dot(d, H * d) / dot(d, d)
            γ_star = -dot(g0, d) / dot(d, H * d)
            @test 0 < γ_star < 1   # the clamp is inactive
            γs = Float64[]
            x1, res = solve(fq, ProbSimplex(), xs; grad=∇fq!, max_iters=1, tol=0.0,
                            step_rule=ShortStep(L_d), callback=s -> (push!(γs, s.γ); false))
            @test γs[1] ≈ γ_star rtol=1e-12
            @test x1 ≈ xs .+ γ_star .* d rtol=1e-12
            # The directional derivative vanishes at the new point
            @test abs(dot(H * x1 .+ c, d)) < 1e-12
        end

        # A large L shortens the step; the clamp keeps γ in [0, 1]
        @testset "clamp and scaling" begin
            γ_small = Float64[]; γ_large = Float64[]
            solve(f, lmo, x0; grad=∇f!, max_iters=1, tol=0.0, step_rule=ShortStep(1e-8),
                  callback=s -> (push!(γ_small, s.γ); false))
            solve(f, lmo, x0; grad=∇f!, max_iters=1, tol=0.0, step_rule=ShortStep(1e8),
                  callback=s -> (push!(γ_large, s.γ); false))
            @test γ_small[1] == 1.0
            @test 0 < γ_large[1] < 1e-6
        end

        # With the global constant L = λmax(H) every step decreases f, the first
        # step equals AdaptiveStepSize's first step from the same L, and both
        # reach the minimizer, which lies inside the simplex.
        @testset "agrees with AdaptiveStepSize" begin
            rngi = Random.MersenneTwister(11)
            ni = 10
            Bi = randn(rngi, ni, ni); Hi = Bi'Bi / ni + I
            ci = rand(rngi, ni) .+ 0.5; ci ./= sum(ci)
            fi(x) = 0.5 * dot(x .- ci, Hi * (x .- ci))
            ∇fi!(g, x) = (g .= Hi * (x .- ci); g)
            xi0 = zeros(ni); xi0[1] = 1.0
            L = eigmax(Symmetric(Hi))
            objs = Float64[]; γ_short = Float64[]
            x_s, res_s = solve(fi, ProbSimplex(), xi0; grad=∇fi!, max_iters=20_000, tol=1e-8,
                               monotonic=false, step_rule=ShortStep(L),
                               callback=s -> (push!(objs, s.obj); push!(γ_short, s.γ); false))
            @test res_s.converged
            @test res_s.discards == 0
            @test issorted(objs; rev=true)
            γ_adapt = Float64[]
            x_a, res_a = solve(fi, ProbSimplex(), xi0; grad=∇fi!, max_iters=20_000, tol=1e-8,
                               step_rule=AdaptiveStepSize(L),
                               callback=s -> (push!(γ_adapt, s.γ); false))
            @test res_a.converged
            @test γ_short[1] == γ_adapt[1]
            @test res_s.objective ≈ res_a.objective atol=1e-8
            @test x_s ≈ ci atol=1e-6
            @test x_a ≈ ci atol=1e-6
        end

        # The sparse vertex path (Knapsack) gives the same iterates as a dense oracle
        @testset "sparse and dense vertices agree" begin
            m = 30
            rngk = Random.MersenneTwister(3)
            B = randn(rngk, m, m); Hk = B'B / m + I
            ck = randn(rngk, m)
            fk(x) = 0.5 * dot(x, Hk * x) + dot(ck, x)
            ∇fk!(g, x) = (mul!(g, Hk, x); g .+= ck; g)
            ks = Knapsack(4, m)
            dense_lmo(v, g) = ks(v, g)   # plain function: dense vertex path
            L = eigmax(Symmetric(Hk))
            xk0 = zeros(m)
            x_sp, r_sp = solve(fk, ks, xk0; grad=∇fk!, max_iters=200, tol=0.0, step_rule=ShortStep(L))
            x_de, r_de = solve(fk, dense_lmo, xk0; grad=∇fk!, max_iters=200, tol=0.0, step_rule=ShortStep(L))
            @test x_sp ≈ x_de rtol=1e-12
            @test r_sp.objective ≈ r_de.objective rtol=1e-12
        end

        @testset "constructor, show and unsupported paths" begin
            @test ShortStep(2).L === 2.0
            @test ShortStep(2.0f0).L === 2.0f0
            @test_throws ArgumentError ShortStep(0.0)
            @test_throws ArgumentError ShortStep(-1.0)
            @test_throws ArgumentError ShortStep(Inf)
            @test sprint(show, ShortStep(2.5)) == "ShortStep(L=2.5)"
            expr = BatchedExpression((x, _, _) -> sum(abs2, x), (g, x, _, _) -> (g .= 2 .* x; g))
            @test_throws ArgumentError batch_solve(expr, ProbSimplex(), fill(0.5, 2, 3);
                                                   step_rule=ShortStep(2.0), max_iters=5)
        end
    end

    @testset "SecantLineSearch" begin
        _search! = Marguerite._secant_search!

        # One-dimensional checks of the trial sequence along d = 1 from x = 0
        @testset "trial sequence in one dimension" begin
            # φ(γ) = (γ - 0.3)²: trial 1 at γ = 1 overshoots, the secant lands on 0.3
            fq(x) = (x[1] - 0.3)^2
            ∇fq!(g, x) = (g[1] = 2 * (x[1] - 0.3); g)
            function setup(grad!)
                c = Cache{Float64}(1)
                x = [0.0]
                grad!(c.gradient, x)
                c.direction[1] = 1.0
                return c, x
            end
            c, x = setup(∇fq!)
            rule = SecantLineSearch()
            rule.γ_prev = 0.5
            γ, ot, ready = _search!(rule, 1, fq, ∇fq!, x, c, fq(x), 1.0, c.gradient[1], 1.0)
            @test γ ≈ 0.3 rtol=1e-14
            @test ot ≈ 0.0 atol=1e-28
            @test !ready
            @test c.x_trial[1] ≈ 0.3 rtol=1e-14
            @test rule.γ_prev == γ

            # γ_max caps trial 1; the secant still finds 0.3
            c, x = setup(∇fq!)
            rule.γ_prev = 0.5
            γ, _, _ = _search!(rule, 1, fq, ∇fq!, x, c, fq(x), 1.0, c.gradient[1], 0.4)
            @test γ ≈ 0.3 rtol=1e-14

            # Trial 1 short of the minimizer: accepted with its gradient
            c, x = setup(∇fq!)
            rule.γ_prev = 0.1
            γ, ot, ready = _search!(rule, 1, fq, ∇fq!, x, c, fq(x), 1.0, c.gradient[1], 1.0)
            @test γ == 0.2
            @test ready
            @test ot ≈ fq([0.2])
            @test c.gradient_trial[1] ≈ 2 * (0.2 - 0.3)

            # The first iteration resets γ_prev to γ0, so trial 1 is at min(2γ0, 1) = 1
            c, x = setup(∇fq!)
            rule.γ_prev = 1e-3
            fcount = Ref(0)
            fq_counted(x) = (fcount[] += 1; fq(x))
            γ, _, _ = _search!(rule, 0, fq_counted, ∇fq!, x, c, fq(x), 1.0, c.gradient[1], 1.0)
            @test γ ≈ 0.3 rtol=1e-14
            @test fcount[] == 2   # trial 1 and the secant point

            # φ'(γ) = 1 - 2e^{-kγ} is concave, so the secant point overshoots and f
            # rises there; halvings follow while f rises.
            k = 50.0
            fe(x) = x[1] - (2 / k) * (1 - exp(-k * x[1]))
            ∇fe!(g, x) = (g[1] = 1 - 2 * exp(-k * x[1]); g)
            # Trials: 1 (γ=1), secant (0.5), halvings 0.25, 0.125, 0.0625, 0.03125
            c, x = setup(∇fe!)
            r6 = SecantLineSearch(max_trials=6); r6.γ_prev = 0.5
            γ, ot, ready = _search!(r6, 1, fe, ∇fe!, x, c, fe(x), 1.0, c.gradient[1], 1.0)
            @test γ == 0.03125
            @test ot < fe(x)
            @test !ready
            # With the default four trials every trial fails: open-loop fallback 2/(t+2)
            c, x = setup(∇fe!)
            r4 = SecantLineSearch(); r4.γ_prev = 0.5
            γ, ot, ready = _search!(r4, 1, fe, ∇fe!, x, c, fe(x), 1.0, c.gradient[1], 1.0)
            @test γ == 2 / 3
            @test ot === nothing
            @test !ready
            @test r4.γ_prev == 2 / 3
            # A single trial falls back as soon as trial 1 overshoots
            c, x = setup(∇fq!)
            r1 = SecantLineSearch(1); r1.γ_prev = 0.5
            γ, ot, _ = _search!(r1, 6, fq, ∇fq!, x, c, fq(x), 1.0, c.gradient[1], 1.0)
            @test γ == 0.25
            @test ot === nothing
        end

        # On a quadratic over the simplex the first secant lands on the exact
        # minimizer along d, at two objective evaluations and two gradients.
        @testset "one secant step is exact on a quadratic" begin
            H = [3.0 0.5 0.0; 0.5 2.0 0.3; 0.0 0.3 1.5]
            cq = [0.2, -0.4, 0.1]
            nf = Ref(0); ng = Ref(0)
            fq(x) = (nf[] += 1; 0.5 * dot(x, H * x) + dot(cq, x))
            ∇fq!(g, x) = (ng[] += 1; mul!(g, H, x); g .+= cq; g)
            xs = [1.0, 0.0, 0.0]
            g0 = H * xs .+ cq
            v = zeros(3); v[argmin(g0)] = 1.0
            d = v .- xs
            γ_star = -dot(g0, d) / dot(d, H * d)
            @test 0 < γ_star < 1   # trial 1 at γ = 1 overshoots
            γs = Float64[]
            x1, res = solve(fq, ProbSimplex(), xs; grad=∇fq!, max_iters=1, tol=0.0,
                            step_rule=SecantLineSearch(), callback=s -> (push!(γs, s.γ); false))
            @test γs[1] ≈ γ_star rtol=1e-12
            @test x1 ≈ xs .+ γ_star .* d rtol=1e-12
            @test nf[] == 1 + 2   # start, trial 1, secant point
            @test ng[] == 1 + 2   # start, trial 1, accepted point
        end

        # When trial 1 is accepted its gradient is reused: one gradient per iteration
        @testset "accepted trial 1 reuses its gradient" begin
            ng = Ref(0)
            cl = [0.0, -1.0, -2.0]
            fl(x) = dot(cl, x) + 1e-3 * 0.5 * dot(x, x)
            ∇fl!(g, x) = (ng[] += 1; g .= cl .+ 1e-3 .* x; g)
            γs = Float64[]
            _, res = solve(fl, ProbSimplex(), [1.0, 0.0, 0.0]; grad=∇fl!, max_iters=1, tol=0.0,
                           step_rule=SecantLineSearch(), callback=s -> (push!(γs, s.γ); false))
            @test γs[1] == 1.0
            @test ng[] == 2   # start and trial 1, none after acceptance
        end

        # With exact secant steps the objective decreases monotonically without the
        # monotone guard, and the iterates reach the interior minimizer.
        @testset "monotone on the simplex quadratic" begin
            for (fun, grad, xstart) in ((f, ∇f!, x0),)
                objs = Float64[]
                _, res = solve(fun, lmo, xstart; grad=grad, max_iters=500, tol=0.0,
                               monotonic=false, step_rule=SecantLineSearch(),
                               callback=s -> (push!(objs, s.obj); false))
                @test res.discards == 0
                @test issorted(objs; rev=true)
                @test res.objective < fun(xstart)
                @test res.lower_bound ≤ f_ref
            end
            rngi = Random.MersenneTwister(11)
            ni = 10
            Bi = randn(rngi, ni, ni); Hi = Bi'Bi / ni + I
            ci = rand(rngi, ni) .+ 0.5; ci ./= sum(ci)
            fi(x) = 0.5 * dot(x .- ci, Hi * (x .- ci))
            ∇fi!(g, x) = (g .= Hi * (x .- ci); g)
            xi0 = zeros(ni); xi0[1] = 1.0
            objs = Float64[]
            x_sec, res_sec = solve(fi, ProbSimplex(), xi0; grad=∇fi!, max_iters=20_000, tol=1e-8,
                                   monotonic=false, step_rule=SecantLineSearch(),
                                   callback=s -> (push!(objs, s.obj); false))
            @test res_sec.converged
            @test res_sec.discards == 0
            @test issorted(objs; rev=true)
            @test x_sec ≈ ci atol=1e-6
            _, res_ol = solve(fi, ProbSimplex(), xi0; grad=∇fi!, max_iters=20_000, tol=1e-8)
            @test res_sec.iterations < res_ol.iterations
        end

        # A step rule that returns its objective value is not re-evaluated by the
        # solver: AdaptiveStepSize with an L that never backtracks costs one f per iteration.
        @testset "no duplicate objective evaluations" begin
            nf = Ref(0)
            fc(x) = (nf[] += 1; f(x))
            L = eigmax(Symmetric(Q))
            _, res = solve(fc, lmo, x0; grad=∇f!, max_iters=40, tol=0.0,
                           step_rule=AdaptiveStepSize(L; eta=1.0))
            @test res.iterations == 40
            @test nf[] == 1 + 40
        end

        # A supplied cache is used as is: the solve allocates the iterate copy and
        # small bookkeeping, not a second cache (n = 1000, so one Cache is ~57 kB).
        @testset "supplied caches are not duplicated" begin
            nc = 1000
            fz(x) = 0.5 * dot(x, x)
            ∇fz!(g, x) = (g .= x; g)
            xz = zeros(nc); xz[1] = 1.0
            cache = Cache{Float64}(nc)
            solve(fz, ProbSimplex(), xz; grad=∇fz!, max_iters=50, tol=0.0, cache=cache)
            alloc = @allocated solve(fz, ProbSimplex(), xz; grad=∇fz!, max_iters=50, tol=0.0, cache=cache)
            @test alloc < 2 * 8 * nc
            expr = BatchedExpression((x, _, _) -> 0.5 * dot(x, x), (g, x, _, _) -> (g .= x; g))
            X0 = zeros(nc, 4); X0[1, :] .= 1.0
            bc = BatchCache(X0)
            batch_solve(expr, ProbSimplex(), X0; max_iters=50, tol=0.0, cache=bc)
            balloc = @allocated batch_solve(expr, ProbSimplex(), X0; max_iters=50, tol=0.0, cache=bc)
            @test balloc < (@allocated BatchCache(X0))
        end

        # Reusing one rule object gives identical solves (γ_prev resets at t = 0)
        @testset "state resets between solves" begin
            rule = SecantLineSearch()
            x_a, r_a = solve(f, lmo, x0; grad=∇f!, max_iters=50, tol=0.0, step_rule=rule)
            x_b, r_b = solve(f, lmo, x0; grad=∇f!, max_iters=50, tol=0.0, step_rule=rule)
            @test x_a == x_b
            @test r_a.objective == r_b.objective
        end

        @testset "constructor, show and unsupported paths" begin
            @test SecantLineSearch().max_trials == 4
            @test SecantLineSearch(max_trials=7).max_trials == 7
            @test SecantLineSearch(2).max_trials == 2
            @test SecantLineSearch(γ0=0.25).γ_prev == 0.25
            @test_throws ArgumentError SecantLineSearch(0)
            @test_throws ArgumentError SecantLineSearch(γ0=0.0)
            @test_throws ArgumentError SecantLineSearch(γ0=1.5)
            @test sprint(show, SecantLineSearch()) == "SecantLineSearch(max_trials=4)"
            expr = BatchedExpression((x, _, _) -> sum(abs2, x), (g, x, _, _) -> (g .= 2 .* x; g))
            @test_throws ArgumentError batch_solve(expr, ProbSimplex(), fill(0.5, 2, 3);
                                                   step_rule=SecantLineSearch(), max_iters=5)
        end

        # Float32 iterates work with the Float64 rule state
        @testset "Float32" begin
            Q32 = Float32.(Q); q32 = Float32.(q)
            f32(x) = 0.5f0 * dot(x, Q32 * x) + dot(q32, x)
            ∇f32!(g, x) = (mul!(g, Q32, x); g .+= q32; g)
            x32 = zeros(Float32, n); x32[1] = 1
            x, res = solve(f32, ProbSimplex(1.0f0), x32; grad=∇f32!, max_iters=100, tol=0.0,
                           step_rule=SecantLineSearch())
            @test eltype(x) == Float32
            @test res.objective isa Float32
            @test res.objective < f32(x32)
        end
    end

    @testset "Pairwise variant" begin

        # Strongly convex quadratic whose gradient is negative on [0, 1]^m, so the
        # optimum lies on the face Σx = budget with coordinates at 0, at 1 and in between.
        rngp = Random.MersenneTwister(1)
        mp = 40
        masked = collect(1:5)
        kp = 10
        budget = kp + length(masked)
        ap = 0.5 .+ rand(rngp, mp)
        Bp = rand(rngp, mp, mp) ./ mp
        Hp = Diagonal(ap) + 0.05 * Bp'Bp
        cp = 1.0 .+ 2 .* rand(rngp, mp)
        fp(x) = 0.5 * dot(x .- cp, Hp * (x .- cp))
        ∇fp!(g, x) = (g .= Hp * (x .- cp); g)
        lmop = MaskedKnapsack(budget, masked, mp)
        xv = zeros(mp); xv[masked] .= 1.0                                   # vertex start
        xu = zeros(mp); xu[masked] .= 1.0; xu[6:end] .= kp / (mp - 5)       # on the face
        Lp = opnorm(Matrix(Hp))
        rules() = (MonotonicStepSize(), AdaptiveStepSize(1.0), ShortStep(Lp), SecantLineSearch())

        # Brute-force reference for the away vertex on the masked knapsack
        function ref_away(lmo, x, g; atol=sqrt(eps()))
            m = length(x)
            v = zeros(m)
            v[lmo.is_masked] .= 1.0
            ones_ = [e for e in 1:m if !lmo.is_masked[e] && x[e] ≥ 1 - atol]
            v[ones_] .= 1.0
            bud = lmo.k + lmo.n_masked   # not `budget`: that would rebind the enclosing local
            tight = sum(x) ≥ bud - max(m * eps(), atol) * max(1, bud)
            cand = [e for e in 1:m if !lmo.is_masked[e] && atol < x[e] < 1 - atol && (tight || g[e] > 0)]
            sort!(cand; lt=(i, j) -> g[i] > g[j] || (g[i] == g[j] && i < j))
            v[cand[1:min(length(cand), max(lmo.k - length(ones_), 0))]] .= 1.0
            return v
        end

        @testset "away_vertex! on the masked knapsack" begin
            lmo6 = MaskedKnapsack(3, [1], 6)
            c6 = Cache{Float64}(6)
            # Budget tight: the two fractional coordinates of largest gradient
            x = [1.0, 0.5, 0.5, 0.999, 0.001, 0.0]
            g = [0.0, -1.5, -1.5, -1.001, 1.001, 0.0]
            @test away_vertex!(lmo6, c6, x, g) === c6.away_vertex
            @test c6.away_vertex == [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
            # A coordinate at 1 is part of the face and stays in the away vertex,
            # whatever its gradient; zeros stay out
            x = [1.0, 1.0, 0.5, 0.5, 0.0, 0.0]
            g = [0.0, -9.0, 1.0, 2.0, 5.0, 5.0]
            away_vertex!(lmo6, c6, x, g)
            @test c6.away_vertex == [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
            # Budget slack: only fractional coordinates with positive gradient
            x = [1.0, 0.5, 0.2, 0.3, 0.0, 0.0]
            g = [0.0, -1.0, 3.0, -2.0, 5.0, 5.0]
            away_vertex!(lmo6, c6, x, g)
            @test c6.away_vertex == [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            g = [0.0, -1.0, -3.0, -2.0, 5.0, 5.0]
            away_vertex!(lmo6, c6, x, g)
            @test c6.away_vertex == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # Ties go to the smaller index
            x = [1.0, 0.5, 0.5, 0.5, 0.5, 0.0]
            g = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
            away_vertex!(lmo6, c6, x, g)
            @test c6.away_vertex == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

            # Randomized: matches the reference, lies in the face of x, and
            # ⟨g, v⁻⟩ ≥ ⟨g, x⟩; the pairwise direction has a positive feasible step
            rng_r = Random.MersenneTwister(99)
            bad = 0
            for trial in 1:500
                m = rand(rng_r, 5:120)
                nm = rand(rng_r, 0:min(10, m - 1))
                msk = randperm(rng_r, m)[1:nm]
                k = rand(rng_r, 1:(m - nm))
                lmo_r = MaskedKnapsack(k + nm, msk, m)
                free = setdiff(1:m, msk)
                x = zeros(m); x[msk] .= 1.0
                # random point of the polytope: mix of vertices with some coordinates at 0 and 1
                for _ in 1:3
                    v = zeros(m); v[msk] .= 1.0
                    v[free[randperm(rng_r, length(free))[1:k]]] .= 1.0
                    w = rand(rng_r) < 0.3 ? 1.0 : rand(rng_r)
                    x .= (1 - w) .* x .+ w .* v
                end
                rand(rng_r) < 0.3 && (x[free] .*= rand(rng_r))     # sometimes leave the budget slack
                g = trial % 4 == 0 ? Float64.(rand(rng_r, -2:2, m)) : randn(rng_r, m)
                c = Cache{Float64}(m)
                vm = copy(away_vertex!(lmo_r, c, x, g))
                ok = vm == ref_away(lmo_r, x, g)
                ok &= all(vm[msk] .== 1) && all(vm[x .== 0] .== 0) && all(vm[x .≥ 1] .== 1)
                ok &= sum(vm) ≤ k + nm
                ok &= dot(g, vm) ≥ dot(g, x) - 1e-12
                vp = zeros(m); lmo_r(vp, g)
                d = vp .- vm
                γmax = pairwise_max_step(lmo_r, x, d)
                ok &= any(!=(0), d) ? γmax > 0 : γmax == 0
                xs = x .+ γmax .* d
                ok &= all(xs .≥ -1e-12) && all(xs .≤ 1 + 1e-12) && sum(xs) ≤ k + nm + 1e-9
                bad += !ok
            end
            @test bad == 0

            # Zero allocations
            x = copy(xu); g = randn(Random.MersenneTwister(4), mp)
            cpw = Cache{Float64}(mp)
            away_vertex!(lmop, cpw, x, g)
            @test (@ballocations away_vertex!($lmop, $cpw, $x, $g)) == 0
        end

        @testset "pairwise_max_step" begin
            lmo6 = MaskedKnapsack(3, [1], 6)
            x = [1.0, 0.5, 0.5, 0.999, 0.001, 0.0]
            @test pairwise_max_step(lmo6, x, [0.0, 1.0, 1.0, -1.0, -1.0, 0.0]) ≈ 0.001
            @test pairwise_max_step(lmo6, x, [0.0, 1.0, 0.0, -1.0, 0.0, 0.0]) ≈ 0.5
            @test pairwise_max_step(lmo6, x, zeros(6)) == 0.0
            # With budget slack, a mass-adding direction is capped by the slack
            x = [1.0, 0.25, 0.25, 0.0, 0.0, 0.0]
            @test pairwise_max_step(lmo6, x, [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]) ≈ 0.75
            @test pairwise_max_step(lmo6, x, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]) ≈ 0.75
        end

        # Every iterate stays in the polytope; from the face the budget is preserved
        @testset "iterates are feasible and the budget is preserved" begin
            for xs in (xv, xu), rule in rules()
                sums = Float64[]
                viol = Ref(0.0)
                mask_ok = Ref(true)
                cb = function (s)
                    push!(sums, sum(s.x))
                    viol[] = max(viol[], -minimum(s.x), maximum(s.x) - 1)
                    mask_ok[] &= all(s.x[masked] .== 1)
                    return false
                end
                _, res = solve(fp, lmop, xs; grad=∇fp!, max_iters=150, tol=0.0,
                               step_rule=rule, variant=:pairwise, callback=cb)
                @test viol[] ≤ 1e-12
                @test mask_ok[]
                @test all(sums .≤ budget + 1e-9)
                # From the face, or after a full first step from the vertex,
                # every later iterate keeps Σx = budget
                @test maximum(abs.(sums .- budget)) ≤ 1e-9
                # Runs that reach the optimum have a gap at rounding level, which
                # can be slightly negative, so allow rounding in f
                @test res.lower_bound ≤ res.objective + 1e-12 * max(1, abs(res.objective))
            end
        end

        # Same step rule, at most the same iteration count: pairwise ends with a
        # much smaller gap (it may even reach a zero gap and stop early)
        @testset "lower gap than plain Frank-Wolfe" begin
            for xs in (xv, xu)
                _, r_fw = solve(fp, lmop, xs; grad=∇fp!, max_iters=300, tol=0.0,
                                step_rule=AdaptiveStepSize(1.0))
                _, r_pw = solve(fp, lmop, xs; grad=∇fp!, max_iters=300, tol=0.0,
                                step_rule=AdaptiveStepSize(1.0), variant=:pairwise)
                @test r_fw.iterations == 300
                @test r_pw.iterations ≤ 300
                @test r_pw.gap < r_fw.gap / 100
                @test r_pw.objective ≤ r_fw.objective
                @test r_pw.drop_steps > 0
                @test r_fw.drop_steps == 0
            end
            # The short step converges outright
            x_pw, r_pw = solve(fp, lmop, xu; grad=∇fp!, max_iters=1000, tol=1e-10,
                               step_rule=ShortStep(Lp), variant=:pairwise)
            @test r_pw.converged
            _, r_fw = solve(fp, lmop, xu; grad=∇fp!, max_iters=r_pw.iterations, tol=1e-10,
                            step_rule=ShortStep(Lp))
            @test !r_fw.converged
            @test r_pw.objective ≤ r_fw.objective
        end

        # When the budget is slack at the optimum, pairwise reaches the same point as FW
        @testset "slack budget at the optimum" begin
            cs = 0.6 .* rand(Random.MersenneTwister(8), mp) .- 0.1
            fs(x) = 0.5 * dot(x .- cs, Hp * (x .- cs))
            ∇fs!(g, x) = (g .= Hp * (x .- cs); g)
            _, r_ref = solve(fs, lmop, xv; grad=∇fs!, max_iters=50_000, tol=1e-12,
                             step_rule=AdaptiveStepSize(1.0))
            x_pw, r_pw = solve(fs, lmop, xv; grad=∇fs!, max_iters=5000, tol=1e-10,
                               step_rule=ShortStep(Lp), variant=:pairwise)
            @test r_pw.converged
            @test sum(x_pw) < budget - 1
            @test r_pw.objective ≈ r_ref.objective rtol=1e-9
            @test r_pw.lower_bound ≤ r_ref.objective + 1e-12
            # A step size far below 1 leaves the budget slack after the first step;
            # mass-adding pairwise steps then reach the face
            sums = Float64[]
            _, r_slow = solve(fp, lmop, xv; grad=∇fp!, max_iters=5000, tol=1e-8,
                              step_rule=ShortStep(50 * Lp), variant=:pairwise,
                              callback=s -> (push!(sums, sum(s.x)); false))
            @test sums[1] < budget - 1
            @test sums[end] ≈ budget atol=1e-9
            @test r_slow.converged
        end

        # A step capped by γ_max drops a coordinate to exactly 0 and is counted
        @testset "drop steps" begin
            lmo6 = MaskedKnapsack(3, [1], 6)
            x0d = [1.0, 0.5, 0.5, 0.999, 0.001, 0.0]
            cd_ = [0.0, 2.0, 2.0, 2.0, -1.0, 0.0]
            fd(x) = 0.5 * sum(abs2, x .- cd_)
            ∇fd!(g, x) = (g .= x .- cd_; g)
            x1, r1 = solve(fd, lmo6, x0d; grad=∇fd!, max_iters=1, tol=0.0,
                           step_rule=ShortStep(1e-6), variant=:pairwise)
            @test r1.drop_steps == 1
            @test x1[5] == 0.0
            @test x1 ≈ [1.0, 0.501, 0.501, 0.998, 0.0, 0.0]
            @test sum(x1) ≈ 3.0
            # The same step with a short step size is not a drop step
            _, r2 = solve(fd, lmo6, x0d; grad=∇fd!, max_iters=1, tol=0.0,
                          step_rule=ShortStep(1e6), variant=:pairwise)
            @test r2.drop_steps == 0
        end

        # Rounding residues: values within atol of a bound count as at the bound
        @testset "away_vertex! tolerances" begin
            lmo6 = MaskedKnapsack(3, [1], 6)
            c6 = Cache{Float64}(6)
            # x₂ is a rounding residue and x₄ is one ulp below 1; the budget is tight
            x = [1.0, 1e-17, 0.5, prevfloat(1.0), 0.5, 0.0]
            g = [0.0, 9.0, 1.0, -9.0, 2.0, 0.0]
            away_vertex!(lmo6, c6, x, g)
            @test c6.away_vertex == [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
            # With atol = 0 the residue enters the support and x₄ counts as fractional
            away_vertex!(lmo6, c6, x, g; atol=0.0)
            @test c6.away_vertex == [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
            # The pairwise step from the default away vertex is not capped by the residue
            vp = zeros(6); lmo6(vp, -abs.(g) .- 1)   # any FW vertex
            away_vertex!(lmo6, c6, x, g)
            d = vp .- c6.away_vertex
            @test pairwise_max_step(lmo6, x, d) ≥ sqrt(eps())
            # A budget slack below atol·budget counts as tight
            xs = [1.0, 0.5, 0.5 - 1e-9, 0.5, 0.5, 0.0]
            @test sum(xs) < 3
            away_vertex!(lmo6, c6, xs, -ones(6))
            @test sum(c6.away_vertex) == 3
        end

        # Regression: near-equal coordinates leave rounding residues after a
        # capped step. Group A (0.25 - i·2⁻⁵⁵, large gradient) is removed by the
        # first, capped step, which leaves A at 0..49·2⁻⁵⁵ and B one rounding
        # error below 1. Without tolerances every later step is capped at 2⁻⁵⁵
        # and the gap stays at 3.125; with them the solve converges at step 2.
        nA, nB, nC = 50, 30, 40
        ms = 1 + nA + nB + nC
        IA = 2:(1 + nA); IB = (2 + nA):(1 + nA + nB); IC = (2 + nA + nB):ms
        lmos = MaskedKnapsack(51, [1], ms)
        cs_ = zeros(ms); cs_[IA] .= -10.0; cs_[IB] .= 10.0; cs_[IC] .= 0.5
        fst(x) = 0.5 * sum(abs2, x .- cs_)
        ∇fst!(g, x) = (g .= x .- cs_; g)
        xst = zeros(ms); xst[1] = 1.0
        xst[IA] .= [0.25 - i * 2.0^-55 for i in 0:(nA - 1)]
        xst[IB] .= 0.75; xst[IC] .= 0.375

        @testset "rounding residues do not stall the pairwise variant" begin
            γs = Float64[]; gaps = Float64[]
            x, r = solve(fst, lmos, xst; grad=∇fst!, max_iters=60, tol=1e-12,
                         step_rule=AdaptiveStepSize(1.0), variant=:pairwise,
                         callback=s -> (push!(γs, s.γ); push!(gaps, s.gap); false))
            @test r.converged
            @test r.iterations ≤ 5
            @test count(γ -> 0 < γ < 1e-12, γs) == 0
            @test gaps[end] < 1e-9 * gaps[1]
            @test all(x[IA] .== 0)
            @test x[IC] ≈ fill(0.5, nC)
            @test sum(x) ≤ 51 + 1e-12
            # Without tolerances the residues are visited one per iteration, but
            # the backtracking keeps L bounded, so the solve still converges
            rule0 = AdaptiveStepSize(1.0)
            γ0s = Float64[]; L0s = Float64[]
            _, r0 = solve(fst, lmos, xst; grad=∇fst!, max_iters=120, tol=1e-12,
                          step_rule=rule0, variant=:pairwise, pairwise_atol=0,
                          callback=s -> (push!(γ0s, s.γ); push!(L0s, rule0.L); false))
            @test count(γ -> 0 < γ < 1e-12, γ0s) > 40
            @test maximum(L0s) ≤ 1.0
            @test r0.converged
        end

        # The capped step snaps the residues of group A to exactly 0 before f is
        # evaluated, and every evaluated point is the point that is kept
        @testset "snapping after a pairwise step" begin
            seen = Float64[]
            fseen(x) = (push!(seen, sum(x)); fst(x))
            x1, r1 = solve(fseen, lmos, xst; grad=∇fst!, max_iters=1, tol=0.0,
                           step_rule=ShortStep(1e-6), variant=:pairwise)
            @test r1.drop_steps == 1
            @test all(x1[IA] .== 0)
            @test r1.objective == fst(x1)
            # with atol = 0 the residues survive
            x0r, _ = solve(fst, lmos, xst; grad=∇fst!, max_iters=1, tol=0.0,
                           step_rule=ShortStep(1e-6), variant=:pairwise, pairwise_atol=0)
            @test count(>(0), x0r[IA]) == nA - 1
            @test maximum(x0r[IA]) < 1e-14
        end

        # When the largest feasible pairwise step is below atol, the iteration
        # takes a Frank-Wolfe step and counts it
        @testset "fallback to Frank-Wolfe steps" begin
            for rule in (AdaptiveStepSize(1.0), ShortStep(Lp))
                x_fb, r_fb = solve(fp, _TinyStepKnapsack(lmop), xu; grad=∇fp!, max_iters=50, tol=0.0,
                                   step_rule=rule, variant=:pairwise)
                @test r_fb.fallback_steps == r_fb.iterations - r_fb.discards
                @test r_fb.fallback_steps > 0
                @test r_fb.drop_steps == 0
                fresh = rule isa AdaptiveStepSize ? AdaptiveStepSize(1.0) : rule
                x_fw, r_fw = solve(fp, lmop, xu; grad=∇fp!, max_iters=50, tol=0.0, step_rule=fresh)
                @test r_fb.objective ≈ r_fw.objective rtol=1e-12
                @test x_fb ≈ x_fw rtol=1e-12
            end
            # Plain Frank-Wolfe and a regular pairwise run report no fallbacks
            _, r_pw = solve(fp, lmop, xu; grad=∇fp!, max_iters=50, tol=0.0,
                            step_rule=AdaptiveStepSize(1.0), variant=:pairwise)
            @test r_pw.fallback_steps == 0
            @test contains(sprint(show, MIME("text/plain"),
                                  Marguerite.Result(1.0, 0.1, 5, false, 0, 0.9, 0.0, 0, 3)), "fallback steps: 3")
            @test_throws ArgumentError solve(fp, lmop, xu; grad=∇fp!, max_iters=5,
                                             variant=:pairwise, pairwise_atol=-1)
        end

        # The pairwise step is replaced by a FW step when its gap is below half
        # the FW gap (here an inexact away vertex makes the pairwise gap tiny)
        @testset "pairwise gap guard" begin
            x_g, r_g = solve(fp, _SwapAwayKnapsack(lmop), xu; grad=∇fp!, max_iters=40, tol=0.0,
                             step_rule=AdaptiveStepSize(1.0), variant=:pairwise)
            @test r_g.fallback_steps == r_g.iterations - r_g.discards
            x_fw, r_fw = solve(fp, lmop, xu; grad=∇fp!, max_iters=40, tol=0.0,
                               step_rule=AdaptiveStepSize(1.0))
            @test x_g ≈ x_fw rtol=1e-12
            # The exact away vertex never triggers it on this problem
            _, r_x = solve(fp, lmop, xu; grad=∇fp!, max_iters=40, tol=0.0,
                           step_rule=AdaptiveStepSize(1.0), variant=:pairwise)
            @test r_x.fallback_steps == 0
        end

        @testset "keywords, extension and errors" begin
            # Forwarded through the parametric method
            fθ(x, θ) = 0.5 * dot(x .- θ, Hp * (x .- θ))
            ∇fθ!(g, x, θ) = (g .= Hp * (x .- θ); g)
            _, rθ = solve(fθ, lmop, xu, cp; grad=∇fθ!, max_iters=300, tol=0.0,
                          step_rule=AdaptiveStepSize(1.0), variant=:pairwise)
            _, r3 = solve(fp, lmop, xu; grad=∇fp!, max_iters=300, tol=0.0,
                          step_rule=AdaptiveStepSize(1.0), variant=:pairwise)
            @test rθ.objective ≈ r3.objective
            @test rθ.drop_steps == r3.drop_steps
            # A user oracle opts in by defining the two methods (see _WrappedKnapsack)
            x_w, r_w = solve(fp, _WrappedKnapsack(lmop), xu; grad=∇fp!, max_iters=300, tol=0.0,
                             step_rule=AdaptiveStepSize(1.0), variant=:pairwise)
            @test r_w.objective ≈ r3.objective
            # Oracles without the pairwise methods and unknown variants are rejected
            @test_throws ArgumentError solve(f, lmo, x0; grad=∇f!, max_iters=5, variant=:pairwise)
            @test_throws ArgumentError solve(f, (v, g) -> lmo(v, g), x0; grad=∇f!, max_iters=5,
                                             variant=:pairwise)
            @test_throws ArgumentError solve(fp, lmop, xu; grad=∇fp!, max_iters=5, variant=:away)
            # show prints the drop count
            @test contains(sprint(show, MIME("text/plain"), r3), "drop steps:")
        end
    end

    @testset "AdaptiveStepSize robustness" begin
        _bt! = Marguerite._backtrack!
        x = fill(0.1, 10)
        dir = fill(1.0, 10)
        buf = zeros(10)
        trial! = Marguerite._AlongTrial(x, dir)

        # A predicted decrease below the rounding level of f neither raises nor
        # relaxes L: f rises by an ulp-scale amount, which the allowance absorbs
        obj = 1.0e3
        f_flat(y) = obj * (1 + 2eps())
        rule = AdaptiveStepSize(1.0)
        γ, ot = _bt!(rule, f_flat, x, 10.0, -1e-20, 1.0, obj, buf, trial!)
        @test γ > 0
        @test ot == f_flat(buf)
        @test rule.L == 1.0
        # An informative accepted step relaxes L by η as before
        f_quad(y) = obj - 0.1 * sum(y .- x)   # linear decrease along dir
        rule = AdaptiveStepSize(1.0)
        γ, ot = _bt!(rule, f_quad, x, 10.0, -1.0, 1.0, obj, buf, trial!)
        @test γ == 0.1
        @test rule.L == 0.5

        # A search that fails all its trials keeps at most η^10 of its growth
        f_bad(y) = obj + 1.0
        rule = AdaptiveStepSize(1.0)
        γ, ot = @test_logs (:warn, r"did not converge") _bt!(rule, f_bad, x, 10.0, -1.0, 1.0, obj, buf, trial!)
        @test γ == 0
        @test ot == obj
        @test buf == x
        @test rule.L == 2.0^10 / 2

        # One iteration whose objective values are all spuriously large makes
        # that search fail. L grows by at most η^10 and the solve recovers: it
        # converges within a few iterations of the undisturbed run. (Keeping the
        # full η^50 growth, as before, left the run unconverged after 5000.)
        rngr = Random.MersenneTwister(11)
        nr = 10
        Br = randn(rngr, nr, nr); Hr = Br'Br / nr + I
        cr = rand(rngr, nr) .+ 0.5; cr ./= sum(cr)
        glitch = Ref(false)
        f_glitch(x) = 0.5 * dot(x .- cr, Hr * (x .- cr)) + (glitch[] ? 1e6 : 0.0)
        ∇fr!(g, x) = (g .= Hr * (x .- cr); g)
        xr0 = zeros(nr); xr0[1] = 1.0
        _, r_ref = solve(f_glitch, ProbSimplex(), xr0; grad=∇fr!, max_iters=5000, tol=1e-8,
                         step_rule=AdaptiveStepSize(1.0))
        @test r_ref.converged
        rule = AdaptiveStepSize(1.0)
        Ls = Float64[]
        _, rr = @test_logs (:warn, r"did not converge") match_mode=:any solve(
            f_glitch, ProbSimplex(), xr0; grad=∇fr!, max_iters=5000, tol=1e-8, step_rule=rule,
            callback=s -> (push!(Ls, rule.L); glitch[] = (s.t == 500); false))
        @test rr.converged
        @test Ls[501] ≤ Ls[500] * 2.0^10      # the failed search (iteration 501)
        @test maximum(Ls) ≤ maximum(Ls[1:500]) * 2.0^10
        @test rr.iterations ≤ r_ref.iterations + 30
    end

    @testset "Fused value and gradient" begin
        # Same iterates as separate callbacks, one fg call per objective evaluation,
        # for every step rule and both variants
        rngf = Random.MersenneTwister(1)
        mf = 30
        Hf = Diagonal(0.5 .+ rand(rngf, mf)) + 0.05 * (rand(rngf, mf, mf) ./ mf)' * (rand(rngf, mf, mf) ./ mf)
        cf = 1.0 .+ 2 .* rand(rngf, mf)
        fk(x) = 0.5 * dot(x .- cf, Hf * (x .- cf))
        ∇fk!(g, x) = (g .= Hf * (x .- cf); g)
        lk = MaskedKnapsack(12, collect(1:4), mf)
        xk = zeros(mf); xk[1:4] .= 1.0
        for (o, xs, fun, grad, var) in ((lmo, x0, f, ∇f!, :fw), (lk, xk, fk, ∇fk!, :fw), (lk, xk, fk, ∇fk!, :pairwise))
            for mk in (() -> MonotonicStepSize(), () -> AdaptiveStepSize(1.0),
                       () -> ShortStep(50.0), () -> SecantLineSearch())
                nf = Ref(0); ng = Ref(0); nfg = Ref(0)
                fc(x) = (nf[] += 1; fun(x))
                gc!(g, x) = (ng[] += 1; grad(g, x))
                fgc(g, x) = (nfg[] += 1; grad(g, x); fun(x))
                x_s, r_s = solve(fc, o, xs; grad=gc!, max_iters=60, tol=0.0, step_rule=mk(), variant=var)
                x_f, r_f = solve(nothing, o, xs; fg=fgc, max_iters=60, tol=0.0, step_rule=mk(), variant=var)
                @test x_f == x_s
                @test (r_f.objective, r_f.gap, r_f.iterations, r_f.discards, r_f.lower_bound, r_f.drop_steps) ==
                      (r_s.objective, r_s.gap, r_s.iterations, r_s.discards, r_s.lower_bound, r_s.drop_steps)
                @test nfg[] == nf[]
                @test ng[] ≥ 1
            end
        end
        # Conflicting arguments are rejected
        fg0(g, x) = (∇f!(g, x); f(x))
        @test_throws ArgumentError solve(f, lmo, x0; fg=fg0, max_iters=5)
        @test_throws ArgumentError solve(nothing, lmo, x0; fg=fg0, grad=∇f!, max_iters=5)
        # The callback and stopping keywords work with fg
        _, r = solve(nothing, lmo, x0; fg=fg0, tol=0.0, rel_tol=0.05)
        @test r.converged
    end
end
