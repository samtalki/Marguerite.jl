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
using JuMP, Clarabel, LinearAlgebra, Random

function random_qp_data(rng, n; epsilon=0.1)
    A = randn(rng, n, n)
    Q = A' * A + epsilon * I
    c = randn(rng, n)
    return Q, c
end

function make_qp(Q, c)
    f(x) = 0.5 * dot(x, Q, x) + dot(c, x)
    function ∇f!(g, x)
        mul!(g, Q, x)
        g .+= c
        return g
    end
    return f, ∇f!
end

function check_match(f, x_fw, x_jump, obj_jump; obj_atol, x_atol)
    @test isapprox(f(x_fw), obj_jump; atol=obj_atol)
    @test isapprox(x_fw, x_jump; atol=x_atol)
end

# Verify that each oracle matches a JuMP+Clarabel reference solution on one representative problem
@testset "Verification vs JuMP+Clarabel (fast representative coverage)" begin

    # Coverage map:
    # - one JuMP/Clarabel cross-check per oracle family
    # Exhaustive size/scenario sweeps remain in test/test_verification.jl.

    # Verify that ProbSimplex matches the reference on a unit probability simplex QP
    @testset "ProbSimplex" begin
        rng = MersenneTwister(2024)
        n = 5
        Q, c = random_qp_data(rng, n)
        f, ∇f! = make_qp(Q, c)

        lmo = ProbSimplex(1.0)
        x0 = fill(1.0 / n, n)
        x_fw, _ = solve(f, lmo, x0; grad=∇f!, max_iters=5_000, tol=1e-5,
                        step_rule=Marguerite.AdaptiveStepSize())

        model = Model(Clarabel.Optimizer)
        set_silent(model)
        @variable(model, y[1:n] >= 0)
        @constraint(model, sum(y) == 1.0)
        @objective(model, Min, 0.5 * y' * Q * y + dot(c, y))
        optimize!(model)

        x_jump = value.(y)
        obj_jump = objective_value(model)
        check_match(f, x_fw, x_jump, obj_jump; obj_atol=5e-3, x_atol=5e-2)
        @test all(x_fw .>= -1e-8)
        @test isapprox(sum(x_fw), 1.0; atol=1e-6)
    end

    # Verify that Simplex (capped) matches the reference on a capped simplex QP
    @testset "Simplex (capped)" begin
        rng = MersenneTwister(2025)
        n = 5
        Q, c = random_qp_data(rng, n)
        f, ∇f! = make_qp(Q, c)

        lmo = Simplex(1.0)
        x0 = fill(1.0 / n, n)
        x_fw, _ = solve(f, lmo, x0; grad=∇f!, max_iters=5_000, tol=1e-5,
                        step_rule=Marguerite.AdaptiveStepSize())

        model = Model(Clarabel.Optimizer)
        set_silent(model)
        @variable(model, y[1:n] >= 0)
        @constraint(model, sum(y) <= 1.0)
        @objective(model, Min, 0.5 * y' * Q * y + dot(c, y))
        optimize!(model)

        x_jump = value.(y)
        obj_jump = objective_value(model)
        check_match(f, x_fw, x_jump, obj_jump; obj_atol=5e-3, x_atol=5e-2)
        @test all(x_fw .>= -1e-8)
        @test sum(x_fw) <= 1.0 + 1e-6
    end

    # Verify that Box matches the reference on a box-constrained QP
    @testset "Box" begin
        rng = MersenneTwister(2026)
        n = 5
        Q, c = random_qp_data(rng, n)
        f, ∇f! = make_qp(Q, c)

        lo = -ones(n)
        hi = 2.0 * ones(n)
        lmo = Box(lo, hi)
        x0 = zeros(n)
        x_fw, _ = solve(f, lmo, x0; grad=∇f!, max_iters=10_000, tol=1e-5,
                        step_rule=Marguerite.AdaptiveStepSize())

        model = Model(Clarabel.Optimizer)
        set_silent(model)
        @variable(model, lo[i] <= y[i=1:n] <= hi[i])
        @objective(model, Min, 0.5 * y' * Q * y + dot(c, y))
        optimize!(model)

        x_jump = value.(y)
        obj_jump = objective_value(model)
        check_match(f, x_fw, x_jump, obj_jump; obj_atol=1e-2, x_atol=0.1)
        @test all(x_fw .>= lo .- 1e-8)
        @test all(x_fw .<= hi .+ 1e-8)
    end

    # Verify that Knapsack matches the reference on a knapsack-constrained QP
    @testset "Knapsack" begin
        rng = MersenneTwister(2027)
        m = 5
        q = 3
        Q, c = random_qp_data(rng, m)
        f, ∇f! = make_qp(Q, c)

        lmo = Knapsack(q, m)
        x0 = fill(Float64(q) / m, m)
        x_fw, _ = solve(f, lmo, x0; grad=∇f!, max_iters=5_000, tol=1e-5,
                        step_rule=Marguerite.AdaptiveStepSize())

        model = Model(Clarabel.Optimizer)
        set_silent(model)
        @variable(model, 0 <= y[1:m] <= 1)
        @constraint(model, sum(y) <= q)
        @objective(model, Min, 0.5 * y' * Q * y + dot(c, y))
        optimize!(model)

        x_jump = value.(y)
        obj_jump = objective_value(model)
        check_match(f, x_fw, x_jump, obj_jump; obj_atol=5e-3, x_atol=5e-2)
        @test all(x_fw .>= -1e-8)
        @test all(x_fw .<= 1.0 + 1e-8)
        @test sum(x_fw) <= q + 1e-6
    end

    # Verify that MaskedKnapsack matches the reference with forced-active indices
    @testset "MaskedKnapsack" begin
        rng = MersenneTwister(2028)
        m = 10
        q = 6
        masked = [2, 5, 8]
        Q, c = random_qp_data(rng, m)
        f, ∇f! = make_qp(Q, c)

        lmo = MaskedKnapsack(q, masked, m)
        x0 = zeros(m)
        x0[masked] .= 1.0
        remaining = q - length(masked)
        free = setdiff(1:m, masked)
        x0[free] .= remaining / length(free)
        x_fw, _ = solve(f, lmo, x0; grad=∇f!, max_iters=10_000, tol=1e-5,
                        step_rule=Marguerite.AdaptiveStepSize())

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
        check_match(f, x_fw, x_jump, obj_jump; obj_atol=5e-3, x_atol=5e-2)
        @test all(x_fw .>= -1e-8)
        @test all(x_fw .<= 1.0 + 1e-8)
        @test sum(x_fw) <= q + 1e-6
        for i in masked
            @test isapprox(x_fw[i], 1.0; atol=1e-6)
        end
    end

    # Verify that WeightedSimplex matches the reference with non-zero lower bounds
    @testset "WeightedSimplex" begin
        rng = MersenneTwister(2029)
        n = 5
        Q, c = random_qp_data(rng, n)
        f, ∇f! = make_qp(Q, c)

        α = abs.(randn(rng, n)) .+ 0.1
        lb = abs.(randn(rng, n)) .* 0.5
        β = dot(α, lb) + sum(α) * 0.5

        lmo = WeightedSimplex(α, β, lb)
        x0 = copy(lb)
        x0[1] += (β - dot(α, lb)) / α[1]
        x_fw, _ = solve(f, lmo, x0; grad=∇f!, max_iters=10_000, tol=1e-5,
                        step_rule=Marguerite.AdaptiveStepSize())

        model = Model(Clarabel.Optimizer)
        set_silent(model)
        @variable(model, y[i=1:n] >= lb[i])
        @constraint(model, dot(α, y) <= β)
        @objective(model, Min, 0.5 * y' * Q * y + dot(c, y))
        optimize!(model)

        x_jump = value.(y)
        obj_jump = objective_value(model)
        check_match(f, x_fw, x_jump, obj_jump; obj_atol=5e-3, x_atol=5e-2)
        @test all(x_fw .>= lb .- 1e-8)
        @test dot(α, x_fw) <= β + 1e-6
    end

    # Verify that Spectraplex matches the reference on a linear SDP over the spectraplex
    @testset "Spectraplex matches JuMP+Clarabel SDP" begin
        n_sp = 3
        C = [2.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0]
        c_vec = vec(C)
        lmo_sp = Spectraplex(n_sp)
        x0_sp = vec(Matrix(1.0I, n_sp, n_sp) ./ n_sp)
        f_sp(x) = dot(c_vec, x)
        x_fw, res_fw = solve(f_sp, lmo_sp, x0_sp; max_iters=10_000, tol=1e-8)
        @test res_fw.converged

        model = Model(Clarabel.Optimizer)
        set_silent(model)
        @variable(model, X[1:n_sp, 1:n_sp], PSD)
        @constraint(model, tr(X) == 1.0)
        @objective(model, Min, sum(C .* X))
        optimize!(model)

        obj_jump = objective_value(model)
        @test isapprox(f_sp(x_fw), obj_jump; atol=1e-6)
    end

end
