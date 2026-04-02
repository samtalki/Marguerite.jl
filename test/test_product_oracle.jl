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
using LinearAlgebra: dot, norm, I
using JuMP
using Clarabel

@testset "Product Oracle" begin

    @testset "Construction" begin
        lmo = ProductOracle(1:3 => ProbSimplex(), 4:5 => Box(0.0, 1.0))
        @test lmo.n == 5
        @test length(lmo.lmos) == 2
        @test lmo.block_ranges == [1:3, 4:5]

        # Must have >= 2 blocks
        @test_throws ArgumentError ProductOracle(1:5 => ProbSimplex())

        # Blocks must be contiguous
        @test_throws ArgumentError ProductOracle(1:3 => ProbSimplex(), 5:7 => Box(0.0, 1.0))

        # Must start at 1
        @test_throws ArgumentError ProductOracle(2:4 => ProbSimplex(), 5:6 => Box(0.0, 1.0))
    end

    @testset "LMO callable" begin
        lmo = ProductOracle(1:3 => ProbSimplex(), 4:5 => Box(0.0, 1.0))
        g = [1.0, 2.0, -1.0, 0.5, -0.3]
        v = zeros(5)
        lmo(v, g)

        # Block 1 (simplex): argmin of gradient → vertex at index 3 (g[3]=-1)
        @test v[3] == 1.0
        @test v[1] == 0.0
        @test v[2] == 0.0

        # Block 2 (box [0,1]): v[i] = 0 if g[i]>=0, 1 if g[i]<0
        @test v[4] == 0.0   # g[4]=0.5 >= 0
        @test v[5] == 1.0   # g[5]=-0.3 < 0
    end

    @testset "Solve: simplex + box" begin
        # min 0.5*||x||^2 s.t. x[1:3] ∈ Δ₃, x[4:5] ∈ [0,1]²
        lmo = ProductOracle(1:3 => ProbSimplex(), 4:5 => Box(0.0, 1.0))
        f(x) = 0.5 * dot(x, x)
        grad!(g, x) = (g .= x)
        x0 = [1/3, 1/3, 1/3, 0.5, 0.5]

        x, result = solve(f, lmo, x0; grad=grad!, max_iters=10000, tol=1e-4)
        @test result.converged
        # Simplex block → [1/3, 1/3, 1/3], box block → [0, 0]
        @test isapprox(x[1:3], fill(1/3, 3), atol=1e-2)
        @test isapprox(x[4:5], [0.0, 0.0], atol=1e-2)
    end

    @testset "Solve: two boxes" begin
        # min (x1-0.3)^2 + (x2-0.7)^2 + (x3+0.5)^2 + (x4-0.2)^2
        # s.t. x[1:2] ∈ [0,1]², x[3:4] ∈ [-1,1]²
        lmo = ProductOracle(
            1:2 => Box(0.0, 1.0),
            3:4 => Box(-1.0, 1.0))
        target = [0.3, 0.7, -0.5, 0.2]
        f(x) = 0.5 * sum((x[i] - target[i])^2 for i in 1:4)
        grad!(g, x) = (g .= x .- target)
        x0 = [0.5, 0.5, 0.0, 0.0]

        x, result = solve(f, lmo, x0; grad=grad!, max_iters=10000, tol=1e-4)
        @test result.converged
        @test isapprox(x, target, atol=1e-2)
    end

    @testset "Active set" begin
        lmo = ProductOracle(1:3 => ProbSimplex(), 4:5 => Box(0.0, 1.0))
        x = [0.5, 0.5, 0.0, 0.0, 0.7]
        as = active_set(lmo, x)

        # x[3]=0 → bound (lower), x[4]=0 → bound (lower)
        @test 3 in as.bound_indices
        @test 4 in as.bound_indices
        # x[5]=0.7 → free in box block
        @test 5 in as.free_indices
        # Simplex equality constraint should be present (sum = 1)
        @test length(as.eq_normals) >= 1
    end

    @testset "Show" begin
        lmo = ProductOracle(1:3 => ProbSimplex(), 4:5 => Box(0.0, 1.0))
        s = sprint(show, lmo)
        @test contains(s, "ProductOracle")
        @test contains(s, "2 blocks")
    end

    # ------------------------------------------------------------------
    # Verification against Clarabel (baseline solver)
    # ------------------------------------------------------------------

    @testset "vs Clarabel: simplex + box QP" begin
        n1, n2 = 3, 2
        n = n1 + n2
        Q = Matrix(1.0I, n, n)
        Q[1,2] = Q[2,1] = 0.3
        c = [-0.5, 0.2, -0.1, 0.3, -0.4]

        # Solve with Marguerite (ProductOracle)
        lmo = ProductOracle(1:n1 => ProbSimplex(), (n1+1):n => Box(0.0, 1.0))
        f(x) = 0.5 * dot(x, Q * x) + dot(c, x)
        grad!(g, x) = (g .= Q * x .+ c)
        x0 = [1/3, 1/3, 1/3, 0.5, 0.5]
        x_marg, r_marg = solve(f, lmo, x0; grad=grad!, max_iters=20000, tol=1e-4)

        # Solve with Clarabel via JuMP
        model = Model(Clarabel.Optimizer)
        set_silent(model)
        @variable(model, y[1:n])
        @constraint(model, y[1:n1] .>= 0)
        @constraint(model, sum(y[1:n1]) == 1.0)
        @constraint(model, y[(n1+1):n] .>= 0)
        @constraint(model, y[(n1+1):n] .<= 1)
        @objective(model, Min, 0.5 * sum(Q[i,j]*y[i]*y[j] for i in 1:n, j in 1:n) + dot(c, y))
        optimize!(model)
        x_clar = value.(y)

        # Marguerite should match Clarabel within FW convergence tolerance
        @test norm(x_marg - x_clar) < 0.05
        @test isapprox(f(x_marg), f(x_clar), atol=0.01)
    end

    @testset "vs Clarabel: three blocks" begin
        # Block 1: simplex (1:3), Block 2: box (4:5), Block 3: simplex (6:8)
        n = 8
        Q = Matrix(2.0I, n, n)
        c = [0.1, -0.3, 0.2, 0.5, -0.1, -0.2, 0.4, -0.3]

        lmo = ProductOracle(
            1:3 => ProbSimplex(),
            4:5 => Box(0.0, 1.0),
            6:8 => ProbSimplex())
        f(x) = 0.5 * dot(x, Q * x) + dot(c, x)
        grad!(g, x) = (g .= Q * x .+ c)
        x0 = [1/3, 1/3, 1/3, 0.5, 0.5, 1/3, 1/3, 1/3]
        x_marg, r_marg = solve(f, lmo, x0; grad=grad!, max_iters=20000, tol=1e-4)

        # Clarabel reference
        model = Model(Clarabel.Optimizer)
        set_silent(model)
        @variable(model, y[1:n])
        @constraint(model, y[1:3] .>= 0)
        @constraint(model, sum(y[1:3]) == 1.0)
        @constraint(model, y[4:5] .>= 0)
        @constraint(model, y[4:5] .<= 1)
        @constraint(model, y[6:8] .>= 0)
        @constraint(model, sum(y[6:8]) == 1.0)
        @objective(model, Min, 0.5 * sum(Q[i,j]*y[i]*y[j] for i in 1:n, j in 1:n) + dot(c, y))
        optimize!(model)
        x_clar = value.(y)

        @test norm(x_marg - x_clar) < 0.05
        @test isapprox(f(x_marg), f(x_clar), atol=0.01)
    end
end
