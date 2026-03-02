module Marguerite

using LinearAlgebra: dot, copyto!
import DifferentiationInterface as DI
import Mooncake
using ChainRulesCore: ChainRulesCore, rrule, NoTangent

const DEFAULT_BACKEND = DI.AutoMooncake(; config=nothing)

include("types.jl")
include("lmo.jl")
include("solver.jl")
include("diff_rules.jl")

export solve, Result, MonotonicStepSize
export LinearOracle, Simplex, ProbSimplex, ProbabilitySimplex, Knapsack, MaskedKnapsack, Box, WeightedSimplex

end
