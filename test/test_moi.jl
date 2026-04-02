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
using JuMP
using LinearAlgebra: dot, norm

@testset "MathOptInterface" begin

    @testset "Box-constrained QP" begin
        model = Model(Marguerite.Optimizer)
        set_silent(model)
        set_attribute(model, "tol", 1e-3)
        set_attribute(model, "max_iters", 10000)

        @variable(model, 0 <= x[1:3] <= 1)
        @objective(model, Min, x[1]^2 + x[2]^2 + x[3]^2 - x[1] - 0.5 * x[2])
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        @test primal_status(model) == MOI.FEASIBLE_POINT
        # Unconstrained minimum is [0.5, 0.25, 0] which is in [0,1]^3
        @test isapprox(value(x[1]), 0.5, atol=1e-2)
        @test isapprox(value(x[2]), 0.25, atol=1e-2)
        @test isapprox(value(x[3]), 0.0, atol=1e-2)
    end

    @testset "Simplex-constrained QP (ProbSimplex)" begin
        model = Model(Marguerite.Optimizer)
        set_silent(model)
        set_attribute(model, "tol", 1e-5)

        @variable(model, x[1:3] >= 0)
        @constraint(model, sum(x) == 1.0)
        @objective(model, Min, x[1]^2 + x[2]^2 + x[3]^2)
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        # Symmetric QP on simplex → uniform solution [1/3, 1/3, 1/3]
        for i in 1:3
            @test isapprox(value(x[i]), 1/3, atol=1e-2)
        end
        @test isapprox(sum(value.(x)), 1.0, atol=1e-4)
    end

    @testset "Capped Simplex QP" begin
        model = Model(Marguerite.Optimizer)
        set_silent(model)
        set_attribute(model, "tol", 1e-3)

        @variable(model, x[1:3] >= 0)
        @constraint(model, sum(x) <= 1.0)
        @objective(model, Min, (x[1] - 0.5)^2 + (x[2] - 0.5)^2 + x[3]^2)
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        # Unconstrained min is (0.5, 0.5, 0), sum=1.0 which is feasible
        @test isapprox(value(x[1]), 0.5, atol=1e-2)
        @test isapprox(value(x[2]), 0.5, atol=1e-2)
        @test isapprox(value(x[3]), 0.0, atol=1e-2)
    end

    @testset "Linear objective on Box" begin
        model = Model(Marguerite.Optimizer)
        set_silent(model)

        @variable(model, -1 <= x[1:2] <= 1)
        @objective(model, Min, x[1] + 2 * x[2])
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        # Linear on box → corner solution [-1, -1]
        @test isapprox(value(x[1]), -1.0, atol=1e-2)
        @test isapprox(value(x[2]), -1.0, atol=1e-2)
    end

    @testset "MAX sense" begin
        model = Model(Marguerite.Optimizer)
        set_silent(model)
        set_attribute(model, "tol", 1e-5)

        @variable(model, x[1:3] >= 0)
        @constraint(model, sum(x) == 1.0)
        @objective(model, Max, x[1] + 2 * x[2] + 0.5 * x[3])
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        # Linear on simplex → max at e_2 (coefficient 2 is largest)
        @test isapprox(value(x[2]), 1.0, atol=1e-2)
    end

    @testset "ScalarBox (uniform bounds)" begin
        model = Model(Marguerite.Optimizer)
        set_silent(model)
        set_attribute(model, "tol", 1e-3)

        @variable(model, 0 <= x[1:4] <= 1)
        @objective(model, Min, sum((x[i] - 0.3)^2 for i in 1:4))
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        for i in 1:4
            @test isapprox(value(x[i]), 0.3, atol=1e-2)
        end
    end

    @testset "Solver attributes" begin
        opt = Marguerite.Optimizer()
        @test MOI.get(opt, MOI.SolverName()) == "Marguerite"

        MOI.set(opt, MOI.RawOptimizerAttribute("tol"), 1e-8)
        @test MOI.get(opt, MOI.RawOptimizerAttribute("tol")) == 1e-8

        MOI.set(opt, MOI.RawOptimizerAttribute("max_iters"), 5000)
        @test MOI.get(opt, MOI.RawOptimizerAttribute("max_iters")) == 5000

        MOI.set(opt, MOI.Silent(), true)
        @test MOI.get(opt, MOI.Silent()) == true

        @test_throws MOI.UnsupportedAttribute MOI.set(opt, MOI.RawOptimizerAttribute("nonexistent"), 1)
    end

    @testset "Empty optimizer" begin
        opt = Marguerite.Optimizer()
        @test MOI.is_empty(opt)
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
    end

    @testset "Objective value includes constant" begin
        model = Model(Marguerite.Optimizer)
        set_silent(model)
        set_attribute(model, "tol", 1e-3)

        @variable(model, 0 <= x[1:2] <= 1)
        @objective(model, Min, x[1]^2 + x[2]^2 + 42.0)
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        @test isapprox(objective_value(model), 42.0, atol=1e-2)
    end
end
