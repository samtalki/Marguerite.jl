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

using Marguerite
using Test
using LinearAlgebra

@testset "Oracles" begin

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

        # Budget exceeds negative count → only negative-gradient indices set
        lmo3 = Knapsack(4, 5)
        v3 = zeros(5)
        lmo3(v3, [-3.0, -1.0, 0.5, 1.0, 2.0])
        @test v3 ≈ [1.0, 1.0, 0.0, 0.0, 0.0]

        # All non-negative → all zeros
        lmo4 = Knapsack(3, 4)
        v4 = zeros(4)
        lmo4(v4, [1.0, 0.0, 2.0, 3.0])
        @test v4 ≈ zeros(4)

        # Zero gradient at sorted boundary → not selected (>= threshold)
        lmo5 = Knapsack(3, 3)
        v5 = zeros(3)
        lmo5(v5, [-2.0, 0.0, -1.0])
        @test v5 ≈ [1.0, 0.0, 1.0]

        # Invalid constructor args
        @test_throws ErrorException Knapsack(-1, 5)
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

        # Budget exceeds negative count among free indices
        lmo2 = MaskedKnapsack(5, [1], 5)
        v2 = zeros(5)
        lmo2(v2, [0.0, -2.0, 1.0, 3.0, 4.0])
        @test v2 ≈ [1.0, 1.0, 0.0, 0.0, 0.0]

        # All free gradients non-negative → only masked entries set
        lmo3 = MaskedKnapsack(4, [1, 2], 5)
        v3 = zeros(5)
        lmo3(v3, [0.0, 0.0, 1.0, 0.5, 2.0])
        @test v3 ≈ [1.0, 1.0, 0.0, 0.0, 0.0]

        # Budget equals masked count (k=0) → only masked entries set
        lmo4 = MaskedKnapsack(2, [1, 2], 5)
        v4 = zeros(5)
        lmo4(v4, [-5.0, -3.0, -1.0, -2.0, -4.0])
        @test v4 ≈ [1.0, 1.0, 0.0, 0.0, 0.0]

        # Invalid constructor args: budget < |masked|
        @test_throws ErrorException MaskedKnapsack(1, [1, 2, 3], 5)
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
