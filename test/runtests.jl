using Marguerite
using Test

@testset "Marguerite.jl" begin
    include("test_lmo.jl")
    include("test_solver.jl")
    include("test_differentiation.jl")
    include("test_convergence.jl")
    include("test_bilevel.jl")
    include("test_verification.jl")
end
