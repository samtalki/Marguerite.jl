using Marguerite
using Test
using LinearAlgebra

@testset "Linear Oracles" begin

    @testset "Simplex" begin
        lmo = Simplex()
        v = zeros(3)

        # Negative gradient → select most negative
        lmo(v, [-1.0, -3.0, -2.0])
        @test v ≈ [0.0, 1.0, 0.0]

        # All positive gradient → origin
        lmo(v, [1.0, 2.0, 3.0])
        @test v ≈ [0.0, 0.0, 0.0]

        # Custom radius
        lmo2 = Simplex(2.5)
        lmo2(v, [-1.0, 0.0, 0.0])
        @test v ≈ [2.5, 0.0, 0.0]
    end

    @testset "ProbabilitySimplex" begin
        lmo = ProbabilitySimplex()
        v = zeros(3)

        # Always picks the vertex with minimum gradient
        lmo(v, [1.0, -1.0, 0.5])
        @test v ≈ [0.0, 1.0, 0.0]

        # All positive → still picks minimum
        lmo(v, [3.0, 2.0, 1.0])
        @test v ≈ [0.0, 0.0, 1.0]

        # Custom radius
        lmo2 = ProbabilitySimplex(5.0)
        lmo2(v, [0.0, 0.0, -1.0])
        @test v ≈ [0.0, 0.0, 5.0]
    end

    @testset "Knapsack" begin
        # 6 items, budget = 3
        lmo = Knapsack(3, 6)
        v = zeros(6)

        g = [0.0, 0.0, -5.0, -1.0, -3.0, 2.0]
        lmo(v, g)
        # Picks 3 most negative: indices 3 (-5), 5 (-3), 4 (-1)
        @test v[3] ≈ 1.0
        @test v[5] ≈ 1.0
        @test v[4] ≈ 1.0
        @test v[1] ≈ 0.0
        @test v[2] ≈ 0.0
        @test v[6] ≈ 0.0

        # Budget = 0 → all zeros
        lmo0 = Knapsack(0, 4)
        v0 = zeros(4)
        lmo0(v0, [-1.0, -2.0, -3.0, -4.0])
        @test v0 ≈ zeros(4)
    end

    @testset "MaskedKnapsack" begin
        # 6 items, masked = [1, 2], budget = 4
        masked = [1, 2]
        lmo = MaskedKnapsack(4, masked, 6)
        v = zeros(6)

        g = [0.0, 0.0, -5.0, -1.0, -3.0, 2.0]
        lmo(v, g)
        # Masked fixed to 1, then picks 2 (=4-2) most negative from remaining
        # remaining gradients: [-5, -1, -3, 2] at indices [3,4,5,6]
        # most negative 2: indices 3 (-5) and 5 (-3)
        @test v[1] ≈ 1.0  # masked
        @test v[2] ≈ 1.0  # masked
        @test v[3] ≈ 1.0  # most negative
        @test v[5] ≈ 1.0  # second most negative
        @test v[4] ≈ 0.0
        @test v[6] ≈ 0.0
    end

    @testset "Box" begin
        lmo = Box([0.0, -1.0, 0.0], [1.0, 1.0, 2.0])
        v = zeros(3)

        lmo(v, [1.0, -1.0, 0.0])
        @test v ≈ [0.0, 1.0, 0.0]  # positive → lb, negative → ub, zero → lb

        lmo(v, [-1.0, -1.0, -1.0])
        @test v ≈ [1.0, 1.0, 2.0]  # all negative → upper bounds
    end

    @testset "WeightedSimplex" begin
        α = [1.0, 2.0, 1.0]
        β = 6.0
        lb = [1.0, 1.0, 1.0]
        lmo = WeightedSimplex(α, β, lb)
        v = zeros(3)

        # β̄ = 6 - (1+2+1) = 2
        # g = [-2.0, -1.0, 0.5]
        # ratios for negative: g[1]/α[1] = -2, g[2]/α[2] = -0.5
        # best (most negative) = index 1
        # v[1] = β̄/α[1] + lb[1] = 2/1 + 1 = 3
        lmo(v, [-2.0, -1.0, 0.5])
        @test v ≈ [3.0, 1.0, 1.0]

        # All non-negative gradient → stay at lower bound
        lmo(v, [1.0, 1.0, 1.0])
        @test v ≈ [1.0, 1.0, 1.0]
    end

    @testset "Function as oracle" begin
        # Plain function works as oracle (no subtyping)
        my_lmo(v, g) = (v .= (g .< 0) .* 1.0; v)
        v = zeros(3)
        my_lmo(v, [-1.0, 0.5, -2.0])
        @test v ≈ [1.0, 0.0, 1.0]
    end
end
