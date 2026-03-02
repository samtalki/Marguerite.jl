module Marguerite

using LinearAlgebra: dot
import DifferentiationInterface as DI
import Mooncake

const DEFAULT_BACKEND = DI.AutoMooncake(; config=nothing)

include("types.jl")
include("lmo.jl")
include("solver.jl")
include("diff_rules.jl")

export solve, Result
export Simplex, ProbSimplex, ProbabilitySimplex, Knapsack, Box, WeightedSimplex

end
