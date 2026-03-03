module Marguerite

using LinearAlgebra: dot, copyto!
import DifferentiationInterface as DI
import Mooncake
import ForwardDiff
using ChainRulesCore: ChainRulesCore, rrule, NoTangent
using PrecompileTools: @compile_workload

const DEFAULT_BACKEND = DI.AutoMooncake(; config=nothing)
const SECOND_ORDER_BACKEND = DI.SecondOrder(
    DEFAULT_BACKEND,
    DI.AutoForwardDiff()
)

include("types.jl")
include("lmo.jl")
include("solver.jl")
include("diff_rules.jl")
include("bilevel.jl")

export solve, Result, CGResult, MonotonicStepSize
export bilevel_solve, bilevel_gradient
export LinearOracle, Simplex, ProbSimplex, ProbabilitySimplex, Knapsack, MaskedKnapsack, Box, WeightedSimplex

@compile_workload begin
    # n=2 workload to precompile solver infrastructure, LMOs, and CG.
    # Mooncake rule compilation uses eval, which is incompatible with
    # Julia's precompilation model, so auto-gradient paths are excluded.
    _H = [2.0 0.5; 0.5 1.0]
    _f(x) = 0.5 * dot(x, _H * x)
    _∇f!(g, x) = (g .= _H * x)
    _lmo = ProbabilitySimplex()
    _x0 = [0.5, 0.5]

    # Manual-gradient solve (no AD)
    solve(_f, _∇f!, _lmo, _x0; max_iters=5)

    # Parametric manual-gradient solve
    _fp(x, θ) = 0.5 * dot(x, _H * x) - dot(θ, x)
    _∇fp!(g, x, θ) = (g .= _H * x .- θ)
    _θ = [1.0, 0.5]
    solve(_fp, _∇fp!, _lmo, _x0, _θ; max_iters=5)
end

end
