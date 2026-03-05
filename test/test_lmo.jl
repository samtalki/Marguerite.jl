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
using BenchmarkTools
using Random

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

    @testset "_partial_sort_negative!" begin
        _psn! = Marguerite._partial_sort_negative!

        @testset "all positive → count=0" begin
            perm = [0, 0, 0]
            count = _psn!(perm, [1.0, 2.0, 3.0], 3)
            @test count == 0
        end

        @testset "all negative, k < n" begin
            perm = zeros(Int, 5)
            count = _psn!(perm, [-3.0, -1.0, -5.0, -2.0, -4.0], 3)
            @test count == 3
            # perm[1:3] should index the 3 most negative in sorted order
            vals = [-3.0, -1.0, -5.0, -2.0, -4.0]
            selected = vals[perm[1:count]]
            @test issorted(selected)
            @test selected ≈ [-5.0, -4.0, -3.0]
        end

        @testset "k = 0" begin
            perm = zeros(Int, 3)
            count = _psn!(perm, [-1.0, -2.0, -3.0], 0)
            @test count == 0
        end

        @testset "NaN values skipped" begin
            perm = zeros(Int, 4)
            count = _psn!(perm, [NaN, -1.0, NaN, -2.0], 3)
            @test count == 2
            vals = [NaN, -1.0, NaN, -2.0]
            selected = vals[perm[1:count]]
            @test issorted(selected)
            @test selected ≈ [-2.0, -1.0]
        end
    end

    @testset "_lmo_and_gap! specializations" begin
        _lag! = Marguerite._lmo_and_gap!

        @testset "Dense fallback (plain function)" begin
            my_lmo(v, g) = (v .= (g .< 0) .* 1.0; v)
            c = Cache{Float64}(3)
            x = [0.3, 0.4, 0.3]
            c.gradient .= [-1.0, 0.5, -2.0]
            gap, nnz = _lag!(my_lmo, c, x, 3)
            @test nnz == -1
            # vertex = [1, 0, 1], gap = g'(x - v) = (-1)(0.3-1) + 0.5(0.4-0) + (-2)(0.3-1)
            @test gap ≈ (-1.0)*(-0.7) + 0.5*0.4 + (-2.0)*(-0.7)
        end

        @testset "Simplex (probability)" begin
            lmo = ProbabilitySimplex()
            c = Cache{Float64}(3)
            x = [0.5, 0.3, 0.2]
            c.gradient .= [1.0, -3.0, -2.0]
            gap, nnz = _lag!(lmo, c, x, 3)
            @test nnz == 1
            @test c.vertex_nzind[1] == 2  # argmin
            @test c.vertex_nzval[1] ≈ 1.0
            # gap = dot(g,x) - r*g_min = (0.5 - 0.9 - 0.4) - 1.0*(-3.0) = -0.8 + 3.0
            @test gap ≈ dot([1.0, -3.0, -2.0], x) - 1.0*(-3.0)
        end

        @testset "Simplex (capped, all positive)" begin
            lmo = Simplex()
            c = Cache{Float64}(3)
            x = [0.0, 0.0, 0.0]
            c.gradient .= [1.0, 2.0, 3.0]
            gap, nnz = _lag!(lmo, c, x, 3)
            @test nnz == 0  # origin vertex
            @test gap ≈ 0.0
        end

        @testset "Simplex (non-unit radius)" begin
            lmo = Simplex(2.5)
            c = Cache{Float64}(3)
            x = [1.0, 0.5, 1.0]
            c.gradient .= [-3.0, -1.0, -2.0]
            gap, nnz = _lag!(lmo, c, x, 3)
            @test nnz == 1
            @test c.vertex_nzval[1] ≈ 2.5
            # Verify gap matches dense
            v_dense = zeros(3)
            lmo(v_dense, c.gradient)
            @test gap ≈ dot(c.gradient, x .- v_dense)
        end

        @testset "Knapsack" begin
            lmo = Knapsack(2, 5)
            c = Cache{Float64}(5)
            x = [0.0, 1.0, 0.5, 0.5, 0.0]
            c.gradient .= [1.0, -4.0, -2.0, 0.5, -1.0]
            gap, nnz = _lag!(lmo, c, x, 5)
            @test nnz == 2
            # Most negative 2: index 2 (-4), index 3 (-2)
            selected = Set(c.vertex_nzind[1:nnz])
            @test selected == Set([2, 3])
            # Verify gap matches dense computation
            v_dense = zeros(5)
            lmo(v_dense, c.gradient)
            expected_gap = dot(c.gradient, x .- v_dense)
            @test gap ≈ expected_gap
        end

        @testset "Knapsack (budget=0)" begin
            lmo = Knapsack(0, 3)
            c = Cache{Float64}(3)
            x = [0.5, 0.3, 0.2]
            c.gradient .= [-1.0, -2.0, -3.0]
            gap, nnz = _lag!(lmo, c, x, 3)
            @test nnz == 0
            @test gap ≈ dot(c.gradient, x)
        end

        @testset "MaskedKnapsack (sparse path)" begin
            lmo = MaskedKnapsack(3, [1], 6)  # 1 masked + 2 free budget
            c = Cache{Float64}(6)
            x = [1.0, 0.5, 0.0, 0.0, 0.5, 0.0]
            c.gradient .= [0.5, -3.0, -1.0, 0.0, -2.0, 1.0]
            gap, nnz = _lag!(lmo, c, x, 6)
            # n_masked=1, k=2, total=3, n/2=3, so 3 > 3 is false → sparse path
            # Verify gap matches dense computation
            v_dense = zeros(6)
            lmo(v_dense, c.gradient)
            expected_gap = dot(c.gradient, x .- v_dense)
            @test gap ≈ expected_gap
            @test nnz == 3  # 1 masked + 2 selected from free
        end

        @testset "MaskedKnapsack (dense fallback)" begin
            # Make nnz > n/2 to trigger dense fallback
            lmo = MaskedKnapsack(4, [1, 2, 3], 6)  # 3 masked + 1 free = 4 > 3
            c = Cache{Float64}(6)
            x = [1.0, 1.0, 1.0, 0.5, 0.0, 0.0]
            c.gradient .= [0.0, 0.0, 0.0, -2.0, -1.0, 1.0]
            gap, nnz = _lag!(lmo, c, x, 6)
            @test nnz == -1  # dense fallback
            v_dense = zeros(6)
            lmo(v_dense, c.gradient)
            expected_gap = dot(c.gradient, x .- v_dense)
            @test gap ≈ expected_gap
        end

        @testset "Sparse vs dense equivalence" begin
            # For every oracle with a specialization, verify sparse gap matches dense
            Random.seed!(123)
            n = 20
            x = rand(n); x ./= sum(x)  # on probability simplex
            g = randn(n)

            for lmo in [ProbabilitySimplex(), Simplex(), Knapsack(5, n),
                        MaskedKnapsack(5, [1, 2], n)]
                c = Cache{Float64}(n)
                c.gradient .= g
                gap_sparse, _ = _lag!(lmo, c, x, n)

                c2 = Cache{Float64}(n)
                c2.gradient .= g
                v = zeros(n)
                lmo(v, g)
                gap_dense = dot(g, x .- v)

                @test gap_sparse ≈ gap_dense atol=1e-12
            end
        end
    end

    @testset "Allocations" begin
        n = 100
        v = zeros(n)
        g = randn(n)

        @testset "Simplex" begin
            lmo = Simplex()
            lmo(v, g)  # warmup
            @test (@ballocations $lmo($v, $g)) == 0
        end

        @testset "ProbabilitySimplex" begin
            lmo = ProbabilitySimplex()
            lmo(v, g)
            @test (@ballocations $lmo($v, $g)) == 0
        end

        @testset "Box" begin
            lmo = Box(zeros(n), ones(n))
            lmo(v, g)
            @test (@ballocations $lmo($v, $g)) == 0
        end

        @testset "Knapsack" begin
            lmo = Knapsack(10, n)
            lmo(v, g)
            @test (@ballocations $lmo($v, $g)) == 0
        end

        @testset "MaskedKnapsack" begin
            lmo = MaskedKnapsack(15, collect(1:5), n)
            lmo(v, g)
            @test (@ballocations $lmo($v, $g)) == 0
        end

        @testset "WeightedSimplex" begin
            α = abs.(randn(n)) .+ 0.1
            β = sum(α) * 0.8
            lb = zeros(n)
            lmo = WeightedSimplex(α, β, lb)
            lmo(v, g)
            @test (@ballocations $lmo($v, $g)) == 0
        end
    end
end
