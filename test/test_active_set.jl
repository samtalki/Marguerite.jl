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

    # Verify that the box oracle correctly identifies bound and free coordinates
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

    # Verify that the probability simplex identifies zero-bound coordinates and the equality constraint
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

    # Verify that the capped simplex detects the budget constraint only when the sum equals the radius
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

    # Verify that the knapsack identifies bound coordinates and the budget equality constraint
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

    # Verify that masked indices are always identified as bound in the active set
    @testset "MaskedKnapsack oracle" begin
        lmo = MaskedKnapsack(4, [1, 2], 5)

        # Masked always pinned
        as = active_set(lmo, [1.0, 1.0, 1.0, 1.0, 0.0])
        @test 1 in as.bound_indices
        @test 2 in as.bound_indices
        @test 5 in as.bound_indices
    end

    # Verify that the weighted simplex identifies lower-bound coordinates and the weighted budget
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

    # Verify that the spectraplex identifies symmetry, trace, and rank-deficient face constraints
    @testset "Spectraplex oracle" begin
        # Rank-1 (2×2): X* = [1 0; 0 0]
        lmo2 = Spectraplex(2)
        x_rank1_2 = vec([1.0 0.0; 0.0 0.0])
        as = active_set(lmo2, x_rank1_2)
        @test isempty(as.bound_indices)
        @test length(as.free_indices) == 4
        @test length(as.eq_normals) == 4  # 1 symmetry + 1 trace + 1 mixed + 1 null-space
        eq = as.eq_normals
        @test size(eq.U, 2) == 1   # rank 1
        @test size(eq.V_perp, 2) == 1  # nullity 1
        @test eq.trace_rhs ≈ 1.0
        # Trace constraint RHS is at the right position
        sym_count = Marguerite._spectraplex_sym_count(2)
        @test as.eq_rhs[sym_count + 1] ≈ 1.0

        # Rank-1 (3×3): X* = [1 0 0; 0 0 0; 0 0 0]
        lmo3 = Spectraplex(3)
        x_rank1_3 = vec([1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0])
        as3 = active_set(lmo3, x_rank1_3)
        @test isempty(as3.bound_indices)
        @test length(as3.free_indices) == 9
        @test length(as3.eq_normals) == 9  # 3 symmetry + 1 trace + 2 mixed + 3 null-space
        eq3 = as3.eq_normals
        @test size(eq3.U, 2) == 1   # rank 1
        @test size(eq3.V_perp, 2) == 2  # nullity 2

        # Full rank: X = I/n → only trace + symmetry constraints (no mixed/null)
        x_full = vec(Matrix(1.0I, 3, 3) ./ 3)
        as_full = active_set(lmo3, x_full)
        @test length(as_full.eq_normals) == 4  # 3 symmetry + 1 trace
        eq_full = as_full.eq_normals
        @test size(eq_full.U, 2) == 3  # full rank
        @test size(eq_full.V_perp, 2) == 0  # no null space
        @test eq_full.trace_rhs ≈ 1.0

        # Custom radius
        lmo_r = Spectraplex(2, 3.0)
        x_r = vec([3.0 0.0; 0.0 0.0])
        as_r = active_set(lmo_r, x_r)
        @test as_r.eq_normals.trace_rhs ≈ 3.0

        # Small trace radius: X = (r/2)I is full-rank and should not pick up
        # rank-deficient face constraints just because r < tol.
        r_small = 1e-8
        lmo_small = Spectraplex(2, r_small)
        x_small = vec((r_small / 2) .* Matrix(1.0I, 2, 2))
        as_small = active_set(lmo_small, x_small; tol=1e-6)
        @test isempty(as_small.bound_indices)
        @test length(as_small.eq_normals) == 2  # symmetry + trace only
        @test size(as_small.eq_normals.U, 2) == 2  # full rank
        @test size(as_small.eq_normals.V_perp, 2) == 0

        # Mixed precision: iterate type and radius type should promote cleanly.
        as_x32 = active_set(lmo2, Float32.(x_rank1_2))
        @test as_x32 isa Marguerite.ActiveConstraints{Float64}
        @test length(as_x32.eq_normals) == 4

        lmo32 = Spectraplex(2, Float32(1))
        as_r32 = active_set(lmo32, x_rank1_2)
        @test as_r32 isa Marguerite.ActiveConstraints{Float64}
        @test length(as_r32.eq_normals) == 4

        # Active-face storage should stay compact for moderate full-rank instances.
        n_mem = 30
        lmo_mem = Spectraplex(n_mem)
        x_mem = vec(Matrix(1.0I, n_mem, n_mem) ./ n_mem)
        as_mem = active_set(lmo_mem, x_mem)
        @test Base.summarysize(as_mem) < 300_000
    end

    # Verify that constructing a box with lower bound above upper bound raises an error
    @testset "Box rejects inverted bounds" begin
        @test_throws ArgumentError Box([2.0, 0.0], [1.0, 1.0])
        @test_throws ArgumentError Box([0.0, 1.1], [1.0, 1.0])
        # Valid bounds should work fine
        @test Box([0.0, 0.0], [1.0, 1.0]) isa Box
        @test Box([0.5, 0.5], [0.5, 0.5]) isa Box  # equal bounds OK
    end

    # Verify that a custom oracle with no active_set method returns all-free with no constraints
    @testset "Default fallback (custom oracle)" begin
        my_lmo(v, g) = (v .= (g .< 0) .* 1.0; v)
        as = active_set(my_lmo, [0.5, 0.5, 0.5])
        @test isempty(as.bound_indices)
        @test sort(as.free_indices) == [1, 2, 3]
        @test isempty(as.eq_normals)
    end

    # Verify that parametric oracles materialize into the correct concrete oracle types
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
