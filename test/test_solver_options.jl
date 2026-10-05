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
end
