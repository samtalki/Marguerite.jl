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

using Marguerite
using Test
using LinearAlgebra

@testset "Active Set Identification" begin

    @testset "Box oracle" begin
        lmo = Box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

        # Interior point
        as = active_set(lmo, [0.5, 0.5, 0.5])
        @test isempty(as.bound_indices)
        @test sort(as.free_indices) == [1, 2, 3]

        # All at lower bounds
        as = active_set(lmo, [0.0, 0.0, 0.0])
        @test sort(as.bound_indices) == [1, 2, 3]
        @test as.bound_values ≈ [0.0, 0.0, 0.0]
        @test isempty(as.free_indices)
        @test all(as.bound_is_lower)

        # Mixed: some at bounds, some free
        as = active_set(lmo, [0.0, 0.5, 1.0])
        @test sort(as.bound_indices) == [1, 3]
        @test as.free_indices == [2]
        @test as.bound_is_lower[findfirst(==(1), as.bound_indices)] == true
        @test as.bound_is_lower[findfirst(==(3), as.bound_indices)] == false
    end

    @testset "ProbabilitySimplex oracle" begin
        lmo = ProbabilitySimplex()

        # Vertex: e_1
        as = active_set(lmo, [1.0, 0.0, 0.0])
        @test sort(as.bound_indices) == [2, 3]
        @test all(v -> v ≈ 0.0, as.bound_values)
        @test as.free_indices == [1]
        @test length(as.eq_normals) == 1
        @test as.eq_normals[1] ≈ ones(3)
        @test as.eq_rhs ≈ [1.0]

        # Edge: split between 1 and 2
        as = active_set(lmo, [0.6, 0.4, 0.0])
        @test as.bound_indices == [3]
        @test sort(as.free_indices) == [1, 2]
        @test length(as.eq_normals) == 1

        # Interior of simplex (all positive)
        as = active_set(lmo, [0.4, 0.3, 0.3])
        @test isempty(as.bound_indices)
        @test sort(as.free_indices) == [1, 2, 3]
        @test length(as.eq_normals) == 1  # budget always active
    end

    @testset "Simplex (capped) oracle" begin
        lmo = Simplex(1.0)

        # At origin (budget not active)
        as = active_set(lmo, [0.0, 0.0, 0.0])
        @test sort(as.bound_indices) == [1, 2, 3]
        @test isempty(as.eq_normals)  # budget not active since ∑=0 ≠ 1

        # Budget active
        as = active_set(lmo, [0.5, 0.5, 0.0])
        @test as.bound_indices == [3]
        @test length(as.eq_normals) == 1
    end

    @testset "Knapsack oracle" begin
        lmo = Knapsack(3, 5)

        # Vertex: 3 ones, 2 zeros, budget active
        as = active_set(lmo, [1.0, 1.0, 1.0, 0.0, 0.0])
        @test sort(as.bound_indices) == [1, 2, 3, 4, 5]
        @test isempty(as.free_indices)
        @test length(as.eq_normals) == 1

        # Fractional: budget active
        as = active_set(lmo, [1.0, 0.5, 1.0, 0.5, 0.0])
        @test 5 in as.bound_indices  # x_5 = 0
        @test 1 in as.bound_indices  # x_1 = 1
        @test 3 in as.bound_indices  # x_3 = 1
        @test 2 in as.free_indices || 4 in as.free_indices
    end

    @testset "MaskedKnapsack oracle" begin
        lmo = MaskedKnapsack(4, [1, 2], 5)

        # Masked always pinned
        as = active_set(lmo, [1.0, 1.0, 1.0, 1.0, 0.0])
        @test 1 in as.bound_indices
        @test 2 in as.bound_indices
        @test 5 in as.bound_indices
    end

    @testset "WeightedSimplex oracle" begin
        α = [1.0, 2.0, 1.0]
        β = 6.0
        lb = [1.0, 1.0, 1.0]
        lmo = WeightedSimplex(α, β, lb)

        # At lower bound, budget not active
        as = active_set(lmo, [1.0, 1.0, 1.0])
        @test sort(as.bound_indices) == [1, 2, 3]
        @test isempty(as.eq_normals)  # ⟨α, x⟩ = 4 ≠ 6

        # Budget active
        x = [3.0, 1.0, 1.0]  # ⟨α, x⟩ = 3 + 2 + 1 = 6 = β
        as = active_set(lmo, x)
        @test 2 in as.bound_indices  # x_2 = lb_2
        @test 3 in as.bound_indices  # x_3 = lb_3
        @test as.free_indices == [1]
        @test length(as.eq_normals) == 1
        @test as.eq_normals[1] ≈ α
    end

    @testset "Default fallback (custom oracle)" begin
        my_lmo(v, g) = (v .= (g .< 0) .* 1.0; v)
        as = active_set(my_lmo, [0.5, 0.5, 0.5])
        @test isempty(as.bound_indices)
        @test sort(as.free_indices) == [1, 2, 3]
        @test isempty(as.eq_normals)
    end

    @testset "materialize" begin
        # ParametricBox
        plmo = ParametricBox(θ -> θ[1:2], θ -> θ[3:4])
        θ = [0.0, 0.0, 1.0, 1.0]
        lmo = materialize(plmo, θ)
        @test lmo isa Box
        @test lmo.lb ≈ [0.0, 0.0]
        @test lmo.ub ≈ [1.0, 1.0]

        # ParametricProbSimplex
        plmo_s = ParametricProbSimplex(θ -> θ[1])
        lmo_s = materialize(plmo_s, [2.0])
        @test lmo_s isa Simplex{Float64, true}
        @test lmo_s.r ≈ 2.0

        # ParametricWeightedSimplex
        plmo_w = ParametricWeightedSimplex(θ -> [1.0, 2.0], θ -> θ[1], θ -> [0.0, 0.0])
        lmo_w = materialize(plmo_w, [5.0])
        @test lmo_w isa WeightedSimplex
        @test lmo_w.β ≈ 5.0
    end
end
