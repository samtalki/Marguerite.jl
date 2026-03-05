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

# examples/bench_vs_clarabel.jl
#
# Benchmark: Marguerite (Frank-Wolfe) vs Clarabel (interior-point) on
# random QPs over ProbSimplex and Box constraints.
#
# Usage:
#   julia --project=. examples/bench_vs_clarabel.jl

using Marguerite
using LinearAlgebra
using BenchmarkTools
using Random
using Printf
using JuMP
using Clarabel

# ── helpers ──────────────────────────────────────────────────────

function random_qp_data(rng, n; epsilon=0.1)
    A = randn(rng, n, n)
    Q = A'A + epsilon * I
    c = randn(rng, n)
    return Q, c
end

function make_qp(Q, c)
    f(x) = 0.5 * dot(x, Q, x) + dot(c, x)
    function ∇f!(g, x)
        mul!(g, Q, x)
        g .+= c
        return g
    end
    return f, ∇f!
end

function solve_clarabel_ps(Q, c, n)
    model = Model(Clarabel.Optimizer)
    set_silent(model)
    @variable(model, y[1:n] >= 0)
    @constraint(model, sum(y) == 1.0)
    @objective(model, Min, 0.5 * y' * Q * y + dot(c, y))
    optimize!(model)
    return value.(y)
end

function solve_clarabel_box(Q, c, lo, hi, n)
    model = Model(Clarabel.Optimizer)
    set_silent(model)
    @variable(model, lo[i] <= y[i=1:n] <= hi[i])
    @objective(model, Min, 0.5 * y' * Q * y + dot(c, y))
    optimize!(model)
    return value.(y)
end

# ── benchmarks ───────────────────────────────────────────────────

function bench_probsimplex(; n=20, seed=9999)
    rng = MersenneTwister(seed)
    Q, c = random_qp_data(rng, n)
    f, ∇f! = make_qp(Q, c)

    lmo = ProbSimplex(1.0)
    x0 = fill(1.0 / n, n)

    # warmup
    solve(f, ∇f!, lmo, x0; max_iters=5000, tol=1e-5,
          step_rule=Marguerite.AdaptiveStepSize())
    solve_clarabel_ps(Q, c, n)

    t_fw = @belapsed solve($f, $∇f!, $lmo, $x0;
        max_iters=5000, tol=1e-5,
        step_rule=Marguerite.AdaptiveStepSize())

    t_cl = @belapsed solve_clarabel_ps($Q, $c, $n)

    speedup = t_cl / t_fw
    @printf("ProbSimplex(n=%d): FW=%.2f ms, Clarabel=%.2f ms, speedup=%.1f×\n",
            n, t_fw * 1e3, t_cl * 1e3, speedup)
end

function bench_box(; n=20, seed=9999)
    rng = MersenneTwister(seed + 1)
    Q, c = random_qp_data(rng, n)
    f, ∇f! = make_qp(Q, c)

    lo = -ones(n)
    hi = 2.0 * ones(n)
    lmo = Box(lo, hi)
    x0 = zeros(n)

    # warmup
    solve(f, ∇f!, lmo, x0; max_iters=10000, tol=1e-5,
          step_rule=Marguerite.AdaptiveStepSize())
    solve_clarabel_box(Q, c, lo, hi, n)

    t_fw = @belapsed solve($f, $∇f!, $lmo, $x0;
        max_iters=10000, tol=1e-5,
        step_rule=Marguerite.AdaptiveStepSize())

    t_cl = @belapsed solve_clarabel_box($Q, $c, $lo, $hi, $n)

    speedup = t_cl / t_fw
    @printf("Box(n=%d): FW=%.2f ms, Clarabel=%.2f ms, speedup=%.1f×\n",
            n, t_fw * 1e3, t_cl * 1e3, speedup)
end

# ── main ─────────────────────────────────────────────────────────

println("Benchmark: Marguerite vs Clarabel on random QPs")
println()
bench_probsimplex()
bench_box()
