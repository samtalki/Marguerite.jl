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
import DifferentiationInterface as DI
import ForwardDiff
using ChainRulesCore: ChainRulesCore, rrule, NoTangent
using PrecompileTools: @compile_workload

const DEFAULT_BACKEND = DI.AutoForwardDiff()
const SECOND_ORDER_BACKEND = DI.SecondOrder(DI.AutoForwardDiff(), DI.AutoForwardDiff())

include("types.jl")
include("lmo.jl")
include("solver.jl")
include("diff_rules.jl")
include("bilevel.jl")

export solve, Result, CGResult, MonotonicStepSize, AdaptiveStepSize, SECOND_ORDER_BACKEND
export bilevel_solve, bilevel_gradient
export LinearOracle, Simplex, ProbSimplex, ProbabilitySimplex, Knapsack, MaskedKnapsack, Box, WeightedSimplex

@compile_workload begin
    # n=2 workload to precompile solver infrastructure and LMOs.
    _H = [2.0 0.5; 0.5 1.0]
    _f(x) = 0.5 * dot(x, _H * x)
    _∇f!(g, x) = (g .= _H * x)
    _lmo = ProbabilitySimplex()
    _x0 = [0.5, 0.5]

    # Manual-gradient solve (no AD)
    solve(_f, _∇f!, _lmo, _x0; max_iters=5)

    # Auto-gradient solve (ForwardDiff — no eval, safe to precompile)
    solve(_f, _lmo, _x0; max_iters=5)

    # Parametric manual-gradient solve
    _fp(x, θ) = 0.5 * dot(x, _H * x) - dot(θ, x)
    _∇fp!(g, x, θ) = (g .= _H * x .- θ)
    _θ = [1.0, 0.5]
    solve(_fp, _∇fp!, _lmo, _x0, _θ; max_iters=5)

    # Parametric auto-gradient solve
    solve(_fp, _lmo, _x0, _θ; max_iters=5)
end

end
