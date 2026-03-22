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

"""
    Marguerite

A minimal, differentiable Frank-Wolfe solver for constrained convex optimization.

The main entry point is [`solve`](@ref). For bilevel problems, use [`bilevel_solve`](@ref).
"""
module Marguerite

using LinearAlgebra: copyto!, eigen, mul!, pinv, Symmetric
import LinearAlgebra: dot
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

export solve, Result, CGResult, SolveResult, BilevelResult, Cache, MonotonicStepSize, AdaptiveStepSize, SECOND_ORDER_BACKEND
export bilevel_solve, bilevel_gradient
export AbstractOracle, FunctionOracle, Simplex, ProbSimplex, ProbabilitySimplex, Knapsack, MaskedKnapsack, Box, ScalarBox, WeightedSimplex, Spectraplex
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
    solve(_f, _lmo, _x0; grad=_∇f!, max_iters=5)

    # Auto-gradient solve
    solve(_f, _lmo, _x0; max_iters=5)

    # Plain function auto-wrap path
    _plain_lmo(v, g) = (fill!(v, 0.0); i = argmin(g); v[i] = 1.0; v)
    solve(_f, _plain_lmo, _x0; grad=_∇f!, max_iters=5)

    # Parametric manual-gradient solve
    _fp(x, θ) = 0.5 * dot(x, _H * x) - dot(θ, x)
    _∇fp!(g, x, θ) = (g .= _H * x .- θ)
    _θ = [1.0, 0.5]
    solve(_fp, _lmo, _x0, _θ; grad=_∇fp!, max_iters=5)

    # Parametric auto-gradient solve
    solve(_fp, _lmo, _x0, _θ; max_iters=5)

    # rrule + pullback (precompile HVP/CG/implicit-diff paths)
    (_x_star, _res), _pb = rrule(solve, _fp, _lmo, _x0, _θ; grad=_∇fp!, max_iters=5, diff_lambda=1e-2)
    _pb((2.0 .* _x_star, nothing))

    # ParametricOracle path
    _plmo = ParametricBox(θ -> zeros(2), θ -> ones(2))
    solve(_fp, _plmo, [0.5, 0.5], _θ; grad=_∇fp!, max_iters=5)
end

end
