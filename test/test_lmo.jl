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
using BenchmarkTools
using Random

@testset "Oracles" begin

    # Verify that the capped simplex oracle selects the correct vertex and respects custom radius
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

        @test_throws ArgumentError Simplex(-1.0)
        @test_throws ArgumentError Marguerite.materialize(Marguerite.ParametricSimplex(_ -> -1.0), nothing)
    end

    # Verify that the probability simplex oracle always picks the minimum-gradient vertex
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

        @test_throws ArgumentError ProbabilitySimplex(-1.0)
        @test_throws ArgumentError Marguerite.materialize(Marguerite.ParametricProbSimplex(_ -> -1.0), nothing)
    end

    # Verify that the knapsack oracle selects the top-k most negative gradient entries
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
        @test_throws ArgumentError Knapsack(-1, 5)
    end

    # Verify that the masked knapsack oracle pins masked indices and selects from the rest
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
        @test_throws ArgumentError MaskedKnapsack(1, [1, 2, 3], 5)
    end

    # Verify that the box oracle picks lower or upper bound per coordinate based on gradient sign
    @testset "Box" begin
        lmo = Box([0.0, -1.0, 0.0], [1.0, 1.0, 2.0])
        v = zeros(3)

        lmo(v, [1.0, -1.0, 0.0])
        @test v ≈ [0.0, 1.0, 0.0]  # positive → lb, negative → ub, zero → lb

        lmo(v, [-1.0, -1.0, -1.0])
        @test v ≈ [1.0, 1.0, 2.0]  # all negative → upper bounds
    end

    # Verify that the weighted simplex oracle selects the best ratio-adjusted vertex
    @testset "WeightedSimplex" begin
        α = [1.0, 2.0, 1.0]
        β = 6.0
        lb = [1.0, 1.0, 1.0]
        lmo = WeightedSimplex(α, β, lb)
        v = zeros(3)

        # β_bar = 6 - (1+2+1) = 2
        # g = [-2.0, -1.0, 0.5]
        # ratios for negative: g[1]/α[1] = -2, g[2]/α[2] = -0.5
        # best (most negative) = index 1
        # v[1] = β_bar/α[1] + lb[1] = 2/1 + 1 = 3
        lmo(v, [-2.0, -1.0, 0.5])
        @test v ≈ [3.0, 1.0, 1.0]

        # All non-negative gradient → stay at lower bound
        lmo(v, [1.0, 1.0, 1.0])
        @test v ≈ [1.0, 1.0, 1.0]

        @test_throws ArgumentError WeightedSimplex([1.0, 2.0], 1.0, [0.0])
        @test_throws ArgumentError WeightedSimplex([1.0, 1.0], 0.5, [1.0, 0.0])
    end

    # Verify that scalar box uses uniform bounds, produces correct active sets, and solves correctly
    @testset "ScalarBox" begin
        # Box(scalar, scalar) creates ScalarBox
        lmo = Box(0.0, 1.0)
        @test lmo isa Marguerite.ScalarBox{Float64}

        # Oracle correctness
        v = zeros(3)
        lmo(v, [1.0, -1.0, 0.0])
        @test v ≈ [0.0, 1.0, 0.0]  # positive → lb, negative → ub, zero → lb

        lmo(v, [-1.0, -2.0, -3.0])
        @test v ≈ [1.0, 1.0, 1.0]  # all negative → upper bound

        lmo(v, [1.0, 2.0, 3.0])
        @test v ≈ [0.0, 0.0, 0.0]  # all positive → lower bound

        # Custom bounds
        lmo2 = Box(-2.0, 3.0)
        lmo2(v, [-1.0, 1.0, -1.0])
        @test v ≈ [3.0, -2.0, 3.0]

        # Integer promotion
        lmo3 = Box(0, 1)
        @test lmo3 isa Marguerite.ScalarBox{Float64}

        # active_set at bounds
        x_lb = [0.0, 0.0, 0.0]
        as = Marguerite.active_set(lmo, x_lb)
        @test length(as.bound_indices) == 3
        @test all(as.bound_is_lower)

        x_ub = [1.0, 1.0, 1.0]
        as = Marguerite.active_set(lmo, x_ub)
        @test length(as.bound_indices) == 3
        @test !any(as.bound_is_lower)

        x_mid = [0.5, 0.0, 1.0]
        as = Marguerite.active_set(lmo, x_mid)
        @test 1 ∉ as.bound_indices
        @test 2 ∈ as.bound_indices
        @test 3 ∈ as.bound_indices

        # Invalid bounds
        @test_throws ArgumentError Box(5.0, 1.0)

        # Integration: solve with ScalarBox converges
        f(x) = sum((x .- 0.7).^2)
        x0 = [0.1, 0.9, 0.5]
        sr = solve(f, lmo, x0; max_iters=5000, tol=1e-3)
        x_sol, result = sr
        @test all(x -> 0.0 ≤ x ≤ 1.0, x_sol)
        @test x_sol ≈ [0.7, 0.7, 0.7] atol=1e-2
    end

    # Verify that the spectraplex oracle returns rank-1 PSD vertices with correct trace
    @testset "Spectraplex" begin
        # Check that a 2x2 spectraplex vertex is rank-1, PSD, and has unit trace
        @testset "Basic LMO: 2×2" begin
            lmo = Spectraplex(2)
            v = zeros(4)
            # G = [2 1; 1 3], eigenvalues ≈ 1.382, 3.618
            # min eigenvector points toward (negative of) larger off-diag
            g = [2.0, 1.0, 1.0, 3.0]
            lmo(v, g)
            V = reshape(v, 2, 2)
            # Vertex is rank-1 PSD with trace = 1
            @test tr(V) ≈ 1.0 atol=1e-12
            @test V ≈ V' atol=1e-12
            @test eigvals(Symmetric(V))[1] ≥ -1e-12  # PSD
            @test rank(V) == 1 || isapprox(eigvals(Symmetric(V))[1], 0.0, atol=1e-10)
        end

        # Check that the vertex trace equals the spectraplex radius
        @testset "Trace preservation" begin
            lmo = Spectraplex(3)
            v = zeros(9)
            g = randn(9)
            lmo(v, g)
            V = reshape(v, 3, 3)
            @test tr(V) ≈ 1.0 atol=1e-12
        end

        # Check that a non-unit radius produces vertices with the correct trace
        @testset "Custom radius" begin
            r = 2.5
            lmo = Spectraplex(3, r)
            v = zeros(9)
            g = randn(9)
            lmo(v, g)
            V = reshape(v, 3, 3)
            @test tr(V) ≈ r atol=1e-12
        end

        # Check that the default spectraplex has unit radius
        @testset "Unit spectraplex constructor" begin
            lmo = Spectraplex(4)
            @test lmo.n == 4
            @test lmo.r ≈ 1.0
        end

        # Check that invalid dimensions and negative radii are rejected
        @testset "Constructor validation" begin
            @test_throws ArgumentError Spectraplex(0)
            @test_throws ArgumentError Spectraplex(3, -1.0)
            @test_throws ArgumentError Spectraplex(3, -1)
            @test Spectraplex(3, 0.0).r == 0.0
        end

        # Check that the spectraplex gap specialization matches the dense fallback computation
        @testset "_lmo_and_gap! specialization" begin
            _lag! = Marguerite._lmo_and_gap!
            n = 3
            m = n^2
            lmo = Spectraplex(n)
            c = Cache{Float64}(m)
            # Start at X0 = I/3 (on spectraplex)
            X0 = Matrix{Float64}(I, n, n) / n
            x = vec(X0)
            c.gradient .= randn(m)
            gap, nnz = _lag!(lmo, c, x, m)
            @test nnz == -1  # dense path
            # Verify gap matches dense computation
            v_dense = zeros(m)
            lmo(v_dense, c.gradient)
            expected_gap = dot(c.gradient, x .- v_dense)
            @test gap ≈ expected_gap atol=1e-10
        end
    end

    # Verify that a plain function works as an oracle without subtyping AbstractOracle
    @testset "Function as oracle" begin
        # Plain function works as oracle (no subtyping)
        my_lmo(v, g) = (v .= (g .< 0) .* 1.0; v)
        v = zeros(3)
        my_lmo(v, [-1.0, 0.5, -2.0])
        @test v ≈ [1.0, 0.0, 1.0]
    end

    # Verify that the partial sort finds the k most negative values and handles edge cases
    @testset "_partial_sort_negative!" begin
        _psn! = Marguerite._partial_sort_negative!

        # Check that all-positive values produce zero selected elements
        @testset "all positive → count=0" begin
            perm = [0, 0, 0]
            count = _psn!(perm, [1.0, 2.0, 3.0], 3)
            @test count == 0
        end

        # Check that only the k most negative values are selected in sorted order
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

        # Check that budget zero selects nothing
        @testset "k = 0" begin
            perm = zeros(Int, 3)
            count = _psn!(perm, [-1.0, -2.0, -3.0], 0)
            @test count == 0
        end

        # Check that NaN values are safely skipped during selection
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

    # Verify that each oracle's gap specialization matches the dense dot-product computation
    @testset "_lmo_and_gap! specializations" begin
        _lag! = Marguerite._lmo_and_gap!

        # Check that the generic fallback computes the correct gap for a plain function oracle
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

        # Check that the probability simplex gap uses the sparse single-vertex formula
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

        # Check that the capped simplex returns the origin vertex when all gradients are positive
        @testset "Simplex (capped, all positive)" begin
            lmo = Simplex()
            c = Cache{Float64}(3)
            x = [0.0, 0.0, 0.0]
            c.gradient .= [1.0, 2.0, 3.0]
            gap, nnz = _lag!(lmo, c, x, 3)
            @test nnz == 0  # origin vertex
            @test gap ≈ 0.0
        end

        # Check that a non-unit radius simplex scales the vertex and gap correctly
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

        # Check that the knapsack gap selects the top-k indices and matches the dense gap
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

        # Check that a zero-budget knapsack returns the origin and gap equals dot(g, x)
        @testset "Knapsack (budget=0)" begin
            lmo = Knapsack(0, 3)
            c = Cache{Float64}(3)
            x = [0.5, 0.3, 0.2]
            c.gradient .= [-1.0, -2.0, -3.0]
            gap, nnz = _lag!(lmo, c, x, 3)
            @test nnz == 0
            @test gap ≈ dot(c.gradient, x)
        end

        # Check that the masked knapsack uses the sparse path when nnz is small
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

        # Check that the masked knapsack falls back to dense when nnz exceeds half the dimension
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

        # Check that the scalar box always uses the dense gap path
        @testset "ScalarBox (dense fallback)" begin
            lmo = Box(0.0, 1.0)
            c = Cache{Float64}(3)
            x = [0.3, 0.7, 0.5]
            c.gradient .= [1.0, -1.0, 0.5]
            gap, nnz = _lag!(lmo, c, x, 3)
            @test nnz == -1  # dense path
            v_dense = zeros(3)
            lmo(v_dense, c.gradient)
            @test gap ≈ dot(c.gradient, x .- v_dense)
        end

        # Check that the sparse gap formula matches the dense dot-product for all oracles
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

    # Verify that all oracles handle a zero gradient vector without error
    @testset "Zero gradient" begin
        # All oracles must handle a zero gradient vector without error
        g_zero = zeros(5)

        # Check that the capped simplex returns the origin for zero gradient
        @testset "Simplex" begin
            v = zeros(5)
            Simplex()(v, g_zero)
            @test v ≈ zeros(5)
        end

        # Check that the probability simplex picks the first index when all gradients are tied at zero
        @testset "ProbabilitySimplex" begin
            v = zeros(5)
            ProbabilitySimplex()(v, g_zero)
            # All gradients equal (zero) → picks argmin = 1
            @test v[1] ≈ 1.0
            @test sum(v) ≈ 1.0
        end

        # Check that the knapsack returns the empty set for zero gradient
        @testset "Knapsack" begin
            v = zeros(5)
            Knapsack(3, 5)(v, g_zero)
            @test v ≈ zeros(5)  # no negative gradients → empty
        end

        # Check that the box oracle defaults to lower bounds for zero gradient
        @testset "Box" begin
            v = zeros(5)
            Box(zeros(5), ones(5))(v, g_zero)
            @test v ≈ zeros(5)  # zero gradient → lower bounds
        end

        # Check that the scalar box oracle defaults to lower bound for zero gradient
        @testset "ScalarBox" begin
            v = zeros(5)
            Box(0.0, 1.0)(v, g_zero)
            @test v ≈ zeros(5)  # zero gradient → lower bound
        end

        # Check that the weighted simplex returns lower bounds for zero gradient
        @testset "WeightedSimplex" begin
            α = ones(5)
            β = 3.0
            lb = zeros(5)
            v = zeros(5)
            WeightedSimplex(α, β, lb)(v, g_zero)
            @test v ≈ zeros(5)  # no negative gradients → lower bounds
        end

        # Check that the spectraplex returns a valid PSD vertex for zero gradient
        @testset "Spectraplex" begin
            v = zeros(9)
            Spectraplex(3)(v, zeros(9))
            V = reshape(v, 3, 3)
            @test tr(V) ≈ 1.0
            @test V ≈ V'  # symmetric
            eigs = eigvals(Symmetric(V))
            @test all(eigs .>= -1e-12)  # PSD
        end
    end

    # Verify that a box with equal lower and upper bounds forces that coordinate to a fixed value
    @testset "Box with equal bounds (singleton)" begin
        # lb == ub on some dimensions → oracle must return that value
        lb = [0.0, 0.5, 0.0]
        ub = [1.0, 0.5, 1.0]
        lmo = Box(lb, ub)
        v = zeros(3)

        lmo(v, [1.0, -1.0, -1.0])
        @test v[2] ≈ 0.5  # singleton dimension forced to 0.5

        lmo(v, [-1.0, 1.0, 1.0])
        @test v[2] ≈ 0.5  # still forced regardless of gradient sign

        # Solve with singleton constraint
        f(x) = sum((x .- [0.3, 0.7, 0.8]).^2)
        x0 = [0.5, 0.5, 0.5]
        x, res = solve(f, lmo, x0; max_iters=5000, tol=1e-3)
        @test x[2] ≈ 0.5 atol=1e-2
    end

    # Verify that all oracle calls are zero-allocation.
    # Note: Spectraplex is excluded because (lmo::Spectraplex)(v, g) intentionally
    # allocates via eigen!. Zero-alloc for Spectraplex goes through _lmo_and_gap!
    # which uses pre-allocated Cache buffers.
    @testset "Allocations" begin
        n = 100
        v = zeros(n)
        g = randn(n)

        # Check that capped simplex oracle allocates nothing
        @testset "Simplex" begin
            lmo = Simplex()
            lmo(v, g)  # warmup
            @test (@ballocations $lmo($v, $g)) == 0
        end

        # Check that probability simplex oracle allocates nothing
        @testset "ProbabilitySimplex" begin
            lmo = ProbabilitySimplex()
            lmo(v, g)
            @test (@ballocations $lmo($v, $g)) == 0
        end

        # Check that vector box oracle allocates nothing
        @testset "Box" begin
            lmo = Box(zeros(n), ones(n))
            lmo(v, g)
            @test (@ballocations $lmo($v, $g)) == 0
        end

        # Check that scalar box oracle allocates nothing
        @testset "ScalarBox" begin
            lmo = Box(0.0, 1.0)
            lmo(v, g)
            @test (@ballocations $lmo($v, $g)) == 0
        end

        # Check that knapsack oracle allocates nothing
        @testset "Knapsack" begin
            lmo = Knapsack(10, n)
            lmo(v, g)
            @test (@ballocations $lmo($v, $g)) == 0
        end

        # Check that masked knapsack oracle allocates nothing
        @testset "MaskedKnapsack" begin
            lmo = MaskedKnapsack(15, collect(1:5), n)
            lmo(v, g)
            @test (@ballocations $lmo($v, $g)) == 0
        end

        # Check that weighted simplex oracle allocates nothing
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
