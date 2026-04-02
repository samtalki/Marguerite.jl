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

using JuMP, Clarabel, LinearAlgebra, Random

# Exhaustive JuMP/Clarabel verification sweeps.
# Representative default-path checks live in test/test_verification_fast.jl.

include("test_common.jl")

# Verify that every oracle produces the same solution as a JuMP+Clarabel reference solver
@testset "Verification vs JuMP+Clarabel" begin

    # ------------------------------------------------------------------
    # ProbSimplex
    # ------------------------------------------------------------------
    # Verify that ProbSimplex solves a QP on the probability simplex correctly
    @testset "ProbSimplex" begin
        rng = MersenneTwister(2024)
        for (n, max_iters) in [(5, 5_000)]
            @testset "n=$n" begin
                Q, c = random_qp_data(rng, n)
                f, ∇f! = make_qp(Q, c)

                lmo = ProbSimplex(1.0)
                x0 = fill(1.0 / n, n)
                x_fw, res = solve(f, lmo, x0;
                    grad=∇f!, max_iters=max_iters, tol=1e-5, step_rule=Marguerite.AdaptiveStepSize())

                model = Model(Clarabel.Optimizer)
                set_silent(model)
                @variable(model, y[1:n] >= 0)
                @constraint(model, sum(y) == 1.0)
                @objective(model, Min, 0.5 * y' * Q * y + dot(c, y))
                optimize!(model)
                x_jump = value.(y)
                obj_jump = objective_value(model)

                @test isapprox(f(x_fw), obj_jump; atol=5e-3)
                @test isapprox(x_fw, x_jump; atol=5e-2)
                @test all(x_fw .>= -1e-8)
                @test isapprox(sum(x_fw), 1.0; atol=1e-6)
            end
        end

        # Verify that ProbSimplex works with a non-unit radius
        @testset "non-unit radius r=2.5" begin
            n = 5
            Q, c = random_qp_data(rng, n)
            f, ∇f! = make_qp(Q, c)
            r = 2.5

            lmo = ProbSimplex(r)
            x0 = fill(r / n, n)
            x_fw, res = solve(f, lmo, x0;
                max_iters=5_000, tol=1e-5, step_rule=Marguerite.AdaptiveStepSize())

            model = Model(Clarabel.Optimizer)
            set_silent(model)
            @variable(model, y[1:n] >= 0)
            @constraint(model, sum(y) == r)
            @objective(model, Min, 0.5 * y' * Q * y + dot(c, y))
            optimize!(model)
            x_jump = value.(y)
            obj_jump = objective_value(model)

            @test isapprox(f(x_fw), obj_jump; atol=5e-3)
            @test isapprox(x_fw, x_jump; atol=5e-2)
            @test all(x_fw .>= -1e-8)
            @test isapprox(sum(x_fw), r; atol=1e-6)
        end
    end

    # ------------------------------------------------------------------
    # Simplex (capped)
    # ------------------------------------------------------------------
    # Verify that Simplex (capped) solves a QP on the capped simplex correctly
    @testset "Simplex (capped)" begin
        rng = MersenneTwister(2025)
        for (n, max_iters) in [(5, 5_000)]
            @testset "n=$n" begin
                Q, c = random_qp_data(rng, n)
                f, ∇f! = make_qp(Q, c)

                lmo = Simplex(1.0)
                x0 = fill(1.0 / n, n)
                x_fw, res = solve(f, lmo, x0;
                    grad=∇f!, max_iters=max_iters, tol=1e-5, step_rule=Marguerite.AdaptiveStepSize())

                model = Model(Clarabel.Optimizer)
                set_silent(model)
                @variable(model, y[1:n] >= 0)
                @constraint(model, sum(y) <= 1.0)
                @objective(model, Min, 0.5 * y' * Q * y + dot(c, y))
                optimize!(model)
                x_jump = value.(y)
                obj_jump = objective_value(model)

                @test isapprox(f(x_fw), obj_jump; atol=5e-3)
                @test isapprox(x_fw, x_jump; atol=5e-2)
                @test all(x_fw .>= -1e-8)
                @test sum(x_fw) <= 1.0 + 1e-6
            end
        end
    end

    # ------------------------------------------------------------------
    # Box
    # ------------------------------------------------------------------
    # Box convergence is slower due to FW zig-zagging on box boundaries;
    # wider tolerances and higher iteration counts compensate.
    # Verify that Box solves a QP on a box-constrained domain correctly
    @testset "Box" begin
        rng = MersenneTwister(2026)
        for (n, max_iters) in [(5, 10_000)]
            @testset "n=$n" begin
                Q, c = random_qp_data(rng, n)
                f, ∇f! = make_qp(Q, c)

                lo = -ones(n)
                hi = 2.0 * ones(n)
                lmo = Box(lo, hi)
                x0 = zeros(n)
                x_fw, res = solve(f, lmo, x0;
                    grad=∇f!, max_iters=max_iters, tol=1e-5, step_rule=Marguerite.AdaptiveStepSize())

                model = Model(Clarabel.Optimizer)
                set_silent(model)
                @variable(model, lo[i] <= y[i=1:n] <= hi[i])
                @objective(model, Min, 0.5 * y' * Q * y + dot(c, y))
                optimize!(model)
                x_jump = value.(y)
                obj_jump = objective_value(model)

                @test isapprox(f(x_fw), obj_jump; atol=1e-2)
                @test isapprox(x_fw, x_jump; atol=0.1)
                @test all(x_fw .>= lo .- 1e-8)
                @test all(x_fw .<= hi .+ 1e-8)
            end
        end
    end

    # ------------------------------------------------------------------
    # Knapsack
    # ------------------------------------------------------------------
    # Verify that Knapsack solves a QP on the knapsack polytope correctly
    @testset "Knapsack" begin
        rng = MersenneTwister(2027)
        for (m, q, max_iters) in [(5, 3, 5_000), (20, 8, 10_000)]
            @testset "m=$m, q=$q" begin
                Q, c = random_qp_data(rng, m)
                f, ∇f! = make_qp(Q, c)

                lmo = Knapsack(q, m)
                x0 = fill(Float64(q) / m, m)
                x_fw, res = solve(f, lmo, x0;
                    grad=∇f!, max_iters=max_iters, tol=1e-5, step_rule=Marguerite.AdaptiveStepSize())

                model = Model(Clarabel.Optimizer)
                set_silent(model)
                @variable(model, 0 <= y[1:m] <= 1)
                @constraint(model, sum(y) <= q)
                @objective(model, Min, 0.5 * y' * Q * y + dot(c, y))
                optimize!(model)
                x_jump = value.(y)
                obj_jump = objective_value(model)

                @test isapprox(f(x_fw), obj_jump; atol=5e-3)
                @test isapprox(x_fw, x_jump; atol=5e-2)
                @test all(x_fw .>= -1e-8)
                @test all(x_fw .<= 1.0 + 1e-8)
                @test sum(x_fw) <= q + 1e-6
            end
        end
    end

    # ------------------------------------------------------------------
    # MaskedKnapsack
    # ------------------------------------------------------------------
    # Verify that MaskedKnapsack solves a QP with forced-active indices correctly
    @testset "MaskedKnapsack" begin
        rng = MersenneTwister(2028)
        for (m, q, masked, max_iters) in [
            (10, 6, [2, 5, 8], 10_000),
            (20, 10, [3, 7, 11, 15], 10_000),
        ]
            @testset "m=$m, q=$q" begin
                Q, c = random_qp_data(rng, m)
                f, ∇f! = make_qp(Q, c)

                lmo = MaskedKnapsack(q, masked, m)
                x0 = zeros(m)
                x0[masked] .= 1.0
                remaining = q - length(masked)
                free = setdiff(1:m, masked)
                x0[free] .= remaining / length(free)
                x_fw, res = solve(f, lmo, x0;
                    grad=∇f!, max_iters=max_iters, tol=1e-5, step_rule=Marguerite.AdaptiveStepSize())

                model = Model(Clarabel.Optimizer)
                set_silent(model)
                @variable(model, 0 <= y[1:m] <= 1)
                @constraint(model, sum(y) <= q)
                for i in masked
                    @constraint(model, y[i] == 1)
                end
                @objective(model, Min, 0.5 * y' * Q * y + dot(c, y))
                optimize!(model)
                x_jump = value.(y)
                obj_jump = objective_value(model)

                @test isapprox(f(x_fw), obj_jump; atol=5e-3)
                @test isapprox(x_fw, x_jump; atol=5e-2)
                @test all(x_fw .>= -1e-8)
                @test all(x_fw .<= 1.0 + 1e-8)
                @test sum(x_fw) <= q + 1e-6
                for i in masked
                    @test isapprox(x_fw[i], 1.0; atol=1e-6)
                end
            end
        end
    end

    # ------------------------------------------------------------------
    # WeightedSimplex
    # ------------------------------------------------------------------
    # Verify that WeightedSimplex solves a QP on a weighted simplex correctly
    @testset "WeightedSimplex" begin
        rng = MersenneTwister(2029)
        for (n, max_iters) in [(5, 5_000)]
            @testset "n=$n" begin
                Q, c = random_qp_data(rng, n)
                f, ∇f! = make_qp(Q, c)

                α = abs.(randn(rng, n)) .+ 0.1
                β = sum(α) * 0.8
                lb = zeros(n)

                lmo = WeightedSimplex(α, β, lb)
                x0 = copy(lb)
                x0[1] = β / α[1]
                x_fw, res = solve(f, lmo, x0;
                    grad=∇f!, max_iters=max_iters, tol=1e-5, step_rule=Marguerite.AdaptiveStepSize())

                model = Model(Clarabel.Optimizer)
                set_silent(model)
                @variable(model, y[1:n] >= 0)
                @constraint(model, dot(α, y) <= β)
                @objective(model, Min, 0.5 * y' * Q * y + dot(c, y))
                optimize!(model)
                x_jump = value.(y)
                obj_jump = objective_value(model)

                @test isapprox(f(x_fw), obj_jump; atol=5e-3)
                @test isapprox(x_fw, x_jump; atol=5e-2)
                @test all(x_fw .>= lb .- 1e-8)
                @test dot(α, x_fw) <= β + 1e-6
            end
        end

        # Verify that WeightedSimplex handles non-zero lower bounds correctly
        @testset "non-zero lower bounds" begin
            n = 5
            Q, c = random_qp_data(rng, n)
            f, ∇f! = make_qp(Q, c)

            α = abs.(randn(rng, n)) .+ 0.1
            lb = abs.(randn(rng, n)) .* 0.5
            β = dot(α, lb) + sum(α) * 0.5

            lmo = WeightedSimplex(α, β, lb)
            x0 = copy(lb)
            x0[1] += (β - dot(α, lb)) / α[1]
            x_fw, res = solve(f, lmo, x0;
                max_iters=10_000, tol=1e-5, step_rule=Marguerite.AdaptiveStepSize())

            model = Model(Clarabel.Optimizer)
            set_silent(model)
            @variable(model, y[i=1:n] >= lb[i])
            @constraint(model, dot(α, y) <= β)
            @objective(model, Min, 0.5 * y' * Q * y + dot(c, y))
            optimize!(model)
            x_jump = value.(y)
            obj_jump = objective_value(model)

            @test isapprox(f(x_fw), obj_jump; atol=5e-3)
            @test isapprox(x_fw, x_jump; atol=5e-2)
            @test all(x_fw .>= lb .- 1e-8)
            @test dot(α, x_fw) <= β + 1e-6
        end
    end

end
