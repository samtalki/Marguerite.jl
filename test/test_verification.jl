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

using JuMP, Clarabel, LinearAlgebra, Random
using BenchmarkTools

"""
    random_qp_data(rng, n; epsilon=0.1)

Generate a random QP with positive-definite Hessian Q and linear term c.
Returns `(Q, c)` where `Q = A'A + εI`.
"""
function random_qp_data(rng, n; epsilon=0.1)
    A = randn(rng, n, n)
    Q = A'A + epsilon * I
    c = randn(rng, n)
    return Q, c
end

"""
    make_qp(Q, c)

Returns `(f, ∇f!)` for the QP `min 0.5 x'Qx + c'x`.
"""
function make_qp(Q, c)
    f(x) = 0.5 * dot(x, Q, x) + dot(c, x)
    function ∇f!(g, x)
        mul!(g, Q, x)
        g .+= c
        return g
    end
    return f, ∇f!
end

function bench_clarabel_ps(Q, c, n)
    model = Model(Clarabel.Optimizer)
    set_silent(model)
    @variable(model, y[1:n] >= 0)
    @constraint(model, sum(y) == 1.0)
    @objective(model, Min, 0.5 * y' * Q * y + dot(c, y))
    optimize!(model)
    return value.(y)
end

function bench_clarabel_box(Q, c, lo, hi, n)
    model = Model(Clarabel.Optimizer)
    set_silent(model)
    @variable(model, lo[i] <= y[i=1:n] <= hi[i])
    @objective(model, Min, 0.5 * y' * Q * y + dot(c, y))
    optimize!(model)
    return value.(y)
end

@testset "Verification vs JuMP+Clarabel" begin

    # ------------------------------------------------------------------
    # ProbSimplex
    # ------------------------------------------------------------------
    @testset "ProbSimplex" begin
        rng = MersenneTwister(2024)
        for (n, max_iters) in [(5, 5_000), (20, 10_000)]
            @testset "n=$n" begin
                Q, c = random_qp_data(rng, n)
                f, ∇f! = make_qp(Q, c)

                lmo = ProbSimplex(1.0)
                x0 = fill(1.0 / n, n)
                x_fw, res = solve(f, ∇f!, lmo, x0;
                    max_iters=max_iters, tol=1e-5, step_rule=Marguerite.AdaptiveStepSize())

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

        @testset "non-unit radius r=2.5" begin
            n = 5
            Q, c = random_qp_data(rng, n)
            f, ∇f! = make_qp(Q, c)
            r = 2.5

            lmo = ProbSimplex(r)
            x0 = fill(r / n, n)
            x_fw, res = solve(f, ∇f!, lmo, x0;
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
    @testset "Simplex (capped)" begin
        rng = MersenneTwister(2025)
        for (n, max_iters) in [(5, 5_000), (20, 10_000)]
            @testset "n=$n" begin
                Q, c = random_qp_data(rng, n)
                f, ∇f! = make_qp(Q, c)

                lmo = Simplex(1.0)
                x0 = fill(1.0 / n, n)
                x_fw, res = solve(f, ∇f!, lmo, x0;
                    max_iters=max_iters, tol=1e-5, step_rule=Marguerite.AdaptiveStepSize())

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
    @testset "Box" begin
        rng = MersenneTwister(2026)
        for (n, max_iters) in [(5, 10_000), (20, 100_000)]
            @testset "n=$n" begin
                Q, c = random_qp_data(rng, n)
                f, ∇f! = make_qp(Q, c)

                lo = -ones(n)
                hi = 2.0 * ones(n)
                lmo = Box(lo, hi)
                x0 = zeros(n)
                x_fw, res = solve(f, ∇f!, lmo, x0;
                    max_iters=max_iters, tol=1e-5, step_rule=Marguerite.AdaptiveStepSize())

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
    @testset "Knapsack" begin
        rng = MersenneTwister(2027)
        for (m, q, max_iters) in [(5, 3, 5_000), (20, 8, 10_000)]
            @testset "m=$m, q=$q" begin
                Q, c = random_qp_data(rng, m)
                f, ∇f! = make_qp(Q, c)

                lmo = Knapsack(q, m)
                x0 = fill(Float64(q) / m, m)
                x_fw, res = solve(f, ∇f!, lmo, x0;
                    max_iters=max_iters, tol=1e-5, step_rule=Marguerite.AdaptiveStepSize())

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
                x_fw, res = solve(f, ∇f!, lmo, x0;
                    max_iters=max_iters, tol=1e-5, step_rule=Marguerite.AdaptiveStepSize())

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
    @testset "WeightedSimplex" begin
        rng = MersenneTwister(2029)
        for (n, max_iters) in [(5, 5_000), (15, 20_000)]
            @testset "n=$n" begin
                Q, c = random_qp_data(rng, n)
                f, ∇f! = make_qp(Q, c)

                α = abs.(randn(rng, n)) .+ 0.1
                β = sum(α) * 0.8
                lb = zeros(n)

                lmo = WeightedSimplex(α, β, lb)
                x0 = copy(lb)
                x0[1] = β / α[1]
                x_fw, res = solve(f, ∇f!, lmo, x0;
                    max_iters=max_iters, tol=1e-5, step_rule=Marguerite.AdaptiveStepSize())

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
            x_fw, res = solve(f, ∇f!, lmo, x0;
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

    # ------------------------------------------------------------------
    # Benchmarks: Marguerite vs Clarabel
    # ------------------------------------------------------------------
    @testset "Benchmarks: Marguerite vs Clarabel" begin
        rng = MersenneTwister(9999)
        n_bench = 20

        @testset "ProbSimplex" begin
            Q, c = random_qp_data(rng, n_bench)
            f, ∇f! = make_qp(Q, c)

            lmo = ProbSimplex(1.0)
            x0 = fill(1.0 / n_bench, n_bench)

            # warmup
            solve(f, ∇f!, lmo, x0; max_iters=5000, tol=1e-5,
                  step_rule=Marguerite.AdaptiveStepSize())

            t_fw = @belapsed solve($f, $∇f!, $lmo, $x0;
                max_iters=5000, tol=1e-5,
                step_rule=Marguerite.AdaptiveStepSize())

            bench_clarabel_ps(Q, c, n_bench)  # warmup

            t_cl = @belapsed bench_clarabel_ps($Q, $c, $n_bench)

            speedup = t_cl / t_fw
            @info "ProbSimplex(n=$n_bench): FW=$(round(t_fw * 1e3; digits=2)) ms, Clarabel=$(round(t_cl * 1e3; digits=2)) ms, speedup=$(round(speedup; digits=1))×"
        end

        @testset "Box" begin
            Q, c = random_qp_data(rng, n_bench)
            f, ∇f! = make_qp(Q, c)

            lo = -ones(n_bench)
            hi = 2.0 * ones(n_bench)
            lmo = Box(lo, hi)
            x0 = zeros(n_bench)

            # warmup
            solve(f, ∇f!, lmo, x0; max_iters=10000, tol=1e-5,
                  step_rule=Marguerite.AdaptiveStepSize())

            t_fw = @belapsed solve($f, $∇f!, $lmo, $x0;
                max_iters=10000, tol=1e-5,
                step_rule=Marguerite.AdaptiveStepSize())

            bench_clarabel_box(Q, c, lo, hi, n_bench)  # warmup

            t_cl = @belapsed bench_clarabel_box($Q, $c, $lo, $hi, $n_bench)

            speedup = t_cl / t_fw
            @info "Box(n=$n_bench): FW=$(round(t_fw * 1e3; digits=2)) ms, Clarabel=$(round(t_cl * 1e3; digits=2)) ms, speedup=$(round(speedup; digits=1))×"
        end
    end
end
