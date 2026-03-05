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

module Marguerite

using LinearAlgebra: dot, copyto!
using Printf: @printf
import DifferentiationInterface as DI
import ForwardDiff
using ChainRulesCore: ChainRulesCore, rrule, NoTangent
using PrecompileTools: @compile_workload

"""
    DEFAULT_BACKEND

Default AD backend for first-order gradients (`DI.AutoForwardDiff()`).
Override by passing `backend=` to auto-gradient or parameterized `solve` variants, `bilevel_solve`, etc.
"""
const DEFAULT_BACKEND = DI.AutoForwardDiff()

"""
    SECOND_ORDER_BACKEND

Default AD backend for Hessian-vector products in implicit differentiation
(`DI.SecondOrder(DI.AutoForwardDiff(), DI.AutoForwardDiff())`).
Override by passing `hvp_backend=` to parameterized `solve` variants, `bilevel_solve`, etc.
"""
const SECOND_ORDER_BACKEND = DI.SecondOrder(DI.AutoForwardDiff(), DI.AutoForwardDiff())

include("types.jl")
include("lmo.jl")
include("solver.jl")
include("diff_rules.jl")
include("bilevel.jl")
include("show.jl")

export solve, Result, CGResult, Cache, MonotonicStepSize, AdaptiveStepSize, SECOND_ORDER_BACKEND
export bilevel_solve, bilevel_gradient
export AbstractOracle, Simplex, ProbSimplex, ProbabilitySimplex, Knapsack, MaskedKnapsack, Box, WeightedSimplex
export ParametricOracle, ParametricBox, ParametricSimplex, ParametricProbSimplex, ParametricWeightedSimplex
export ActiveConstraints, active_set, materialize

@compile_workload begin
    # n=2 workload to precompile solver infrastructure and LMOs.
    _H = [2.0 0.5; 0.5 1.0]
    _f(x) = 0.5 * dot(x, _H * x)
    _∇f!(g, x) = (g .= _H * x)
    _lmo = ProbabilitySimplex()
    _x0 = [0.5, 0.5]

    # Manual-gradient solve (no AD)
    solve(_f, _∇f!, _lmo, _x0; max_iters=5)

    # Auto-gradient solve
    solve(_f, _lmo, _x0; max_iters=5)

    # AdaptiveStepSize path
    solve(_f, _∇f!, _lmo, _x0; max_iters=5, step_rule=AdaptiveStepSize())

    # Box oracle solve
    _box = Box(zeros(2), ones(2))
    solve(_f, _∇f!, _box, [0.5, 0.5]; max_iters=5)

    # Knapsack oracle solve
    _knap = Knapsack(1, 2)
    solve(_f, _∇f!, _knap, [0.5, 0.5]; max_iters=5)

    # Parametric manual-gradient solve
    _fp(x, θ) = 0.5 * dot(x, _H * x) - dot(θ, x)
    _∇fp!(g, x, θ) = (g .= _H * x .- θ)
    _θ = [1.0, 0.5]
    solve(_fp, _∇fp!, _lmo, _x0, _θ; max_iters=5)

    # Parametric auto-gradient solve
    solve(_fp, _lmo, _x0, _θ; max_iters=5)

    # rrule + pullback (manual gradient, precompile HVP/CG/implicit-diff paths)
    (_x_star, _res), _pb = rrule(solve, _fp, _∇fp!, _lmo, _x0, _θ; max_iters=5, diff_λ=1e-2)
    _pb((2.0 .* _x_star, nothing))

    # rrule + pullback (auto gradient, precompile joint HVP path)
    (_x_star2, _res2), _pb2 = rrule(solve, _fp, _lmo, _x0, _θ; max_iters=5, diff_λ=1e-2)
    _pb2((2.0 .* _x_star2, nothing))

    # ParametricOracle path
    _plmo = ParametricBox(θ -> zeros(2), θ -> ones(2))
    solve(_fp, _∇fp!, _plmo, [0.5, 0.5], _θ; max_iters=5)

    # bilevel_solve (manual and auto gradient)
    _outer(x) = sum(x .^ 2)
    bilevel_solve(_outer, _fp, _∇fp!, _lmo, _x0, _θ; max_iters=5, diff_λ=1e-2)
    bilevel_solve(_outer, _fp, _lmo, _x0, _θ; max_iters=5, diff_λ=1e-2)
end

end
