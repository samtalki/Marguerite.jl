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
using LinearAlgebra: dot

@testset "show methods" begin

    # ── Result ────────────────────────────────────────────────────────
    @testset "Result" begin
        r = Marguerite.Result(1.23e-4, 5.67e-6, 100, true, 0)
        s = sprint(show, r)
        @test contains(s, "Result(")
        @test contains(s, "converged")
        sp = sprint(show, MIME("text/plain"), r)
        @test contains(sp, "Frank-Wolfe Result")
        @test contains(sp, "iterations: 100")

        r2 = Marguerite.Result(0.5, 0.1, 10, false, 3)
        s2 = sprint(show, r2)
        @test contains(s2, "not converged")
        @test contains(s2, "discards=3")
    end

    # ── CGResult ──────────────────────────────────────────────────────
    @testset "CGResult" begin
        cg = Marguerite.CGResult(25, 1e-8, true)
        s = sprint(show, cg)
        @test contains(s, "CGResult(")
        @test contains(s, "iters=25")
        sp = sprint(show, MIME("text/plain"), cg)
        @test contains(sp, "CG Result")
    end

    # ── SolveResult ───────────────────────────────────────────────────
    @testset "SolveResult" begin
        r = Marguerite.Result(0.5, 1e-5, 50, true, 0)
        sr = SolveResult([0.3, 0.7], r)
        s = sprint(show, sr)
        @test contains(s, "SolveResult{Float64}")
        sp = sprint(show, MIME("text/plain"), sr)
        @test contains(sp, "2-element")
    end

    # ── BilevelResult ─────────────────────────────────────────────────
    @testset "BilevelResult" begin
        r = Marguerite.CGResult(10, 1e-6, true)
        br = BilevelResult([0.5, 0.5], [0.1, -0.1], r)
        s = sprint(show, br)
        @test contains(s, "BilevelResult{Float64}")
        @test contains(s, "cg_iters=10")
        sp = sprint(show, MIME("text/plain"), br)
        @test contains(sp, "θ_grad")
    end

    # ── Step size rules ───────────────────────────────────────────────
    @testset "step size rules" begin
        s = sprint(show, MonotonicStepSize())
        @test contains(s, "MonotonicStepSize")
        s2 = sprint(show, AdaptiveStepSize())
        @test contains(s2, "AdaptiveStepSize")
    end

    # ── Cache ─────────────────────────────────────────────────────────
    @testset "Cache" begin
        c = Cache(5)
        s = sprint(show, c)
        @test contains(s, "Cache{Float64}")
        @test contains(s, "dim=5")
    end

    # ── Oracles ───────────────────────────────────────────────────────
    @testset "oracles" begin
        @test contains(sprint(show, Simplex()), "Simplex")
        @test contains(sprint(show, Simplex()), "≤")
        @test contains(sprint(show, ProbSimplex()), "ProbSimplex")
        @test contains(sprint(show, ProbSimplex()), "=")
        @test contains(sprint(show, Box(zeros(2), ones(2))), "Box")
        @test contains(sprint(show, Box(0.0, 1.0)), "ScalarBox")
        @test contains(sprint(show, Knapsack(3, 5)), "Knapsack")
        @test contains(sprint(show, Knapsack(3, 5)), "budget=3")
        @test contains(sprint(show, WeightedSimplex(ones(3), 1.0, zeros(3))), "WeightedSimplex")
        @test contains(sprint(show, Spectraplex(3)), "Spectraplex")
        @test contains(sprint(show, Spectraplex(3)), "n=3")

        fn_oracle = FunctionOracle(v -> v)
        @test contains(sprint(show, fn_oracle), "FunctionOracle")

        # Box with dim > 4 uses short form
        big_box = Box(zeros(10), ones(10))
        @test contains(sprint(show, big_box), "dim=10")
    end

    # ── ParametricOracles ─────────────────────────────────────────────
    @testset "parametric oracles" begin
        @test contains(sprint(show, ParametricBox(identity, identity)), "ParametricBox")
        @test contains(sprint(show, ParametricProbSimplex(identity)), "ParametricProbSimplex")
        @test contains(sprint(show, ParametricSimplex(identity)), "ParametricSimplex")
        @test contains(sprint(show, ParametricWeightedSimplex(identity, identity, identity)),
                       "ParametricWeightedSimplex")
    end

    # ── ActiveConstraints ─────────────────────────────────────────────
    @testset "ActiveConstraints" begin
        ac = ActiveConstraints{Float64}([1, 2], [0.0, 1.0], BitVector([true, false]),
                                         [3, 4], Vector{Float64}[], Float64[])
        s = sprint(show, ac)
        @test contains(s, "ActiveConstraints{Float64}")
        @test contains(s, "2 bound")
        @test contains(s, "2 free")
    end
end
