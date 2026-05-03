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

# Real-device GPU backend smoke suite. Run with:
#   MARGUERITE_TEST_GROUP=gpu julia --project=. -e 'using Pkg; Pkg.test()'
#
# Auto-detects Metal (Apple Silicon), then CUDA, then AMDGPU. Skips cleanly
# if none is loadable. CUDA / Metal / AMDGPU register their KernelAbstractions
# backends through their own package extensions, so loading any of them
# automatically lights up Marguerite's device path via
# `KernelAbstractions.get_backend(X0)`.

using Marguerite
using Test
using LinearAlgebra: dot, mul!, I

# Try to load a GPU backend in priority order.
const GPU_BACKEND = let
    backend = nothing
    if Sys.isapple()
        try
            @eval using Metal
            if Metal.functional()
                backend = (name="Metal", arr=Metal.MtlArray)
            end
        catch e
            @info "Metal not loadable, trying others" exception=e
        end
    end
    if backend === nothing
        try
            @eval using CUDA
            if CUDA.functional()
                backend = (name="CUDA", arr=CUDA.CuArray)
            end
        catch e
            @info "CUDA not loadable, trying others" exception=e
        end
    end
    if backend === nothing
        try
            @eval using AMDGPU
            if AMDGPU.functional()
                backend = (name="AMDGPU", arr=AMDGPU.ROCArray)
            end
        catch e
            @info "AMDGPU not loadable, no GPU backend available" exception=e
        end
    end
    backend
end

# Metal does not support Float64 on the GPU; CUDA / AMDGPU do.
const SUPPORTS_F64 = GPU_BACKEND === nothing ? false : GPU_BACKEND.name != "Metal"

@testset "GPU Backend" begin
    if GPU_BACKEND === nothing
        @info "No functional GPU backend found — skipping device tests"
        @test true
    else
        @info "Running GPU backend tests on $(GPU_BACKEND.name)"
        Arr = GPU_BACKEND.arr

        @testset "ScalarBox: batch_solve on device (F64)" begin
            if !SUPPORTS_F64
                @info "skipping F64 GPU testset on $(GPU_BACKEND.name) (Float64 unsupported)"
                @test true
            else
                n, B = 8, 4
                H = Arr(collect(I(n) .* 2.0))
                X0 = Arr(fill(0.5, n, B))

                f_per_col(x, _, b) = 0.5 * dot(x, H * x)
                grad_per_col!(g, x, _, b) = (mul!(g, H, x); g)
                expr = BatchedExpression(f_per_col, grad_per_col!)

                X, result = batch_solve(expr, Box(0.0, 1.0), X0;
                                        max_iters=500, tol=1e-3)
                @test all(result.converged)
                @test size(X) == (n, B)
                X_host = Array(X)
                @test all(X_host .>= -1e-6)
                @test all(X_host .<= 1.0 + 1e-6)
            end
        end

        @testset "ProbSimplex: batch_solve on device (F64)" begin
            if !SUPPORTS_F64
                @info "skipping F64 GPU testset on $(GPU_BACKEND.name) (Float64 unsupported)"
                @test true
            else
                n, B = 6, 3
                H = Arr(collect(I(n) .* 2.0))
                X0 = Arr(fill(1.0 / n, n, B))

                f_per_col(x, _, b) = 0.5 * dot(x, H * x)
                grad_per_col!(g, x, _, b) = (mul!(g, H, x); g)
                expr = BatchedExpression(f_per_col, grad_per_col!)

                X, result = batch_solve(expr, ProbSimplex(), X0;
                                        max_iters=500, tol=1e-3)
                @test all(result.converged)
                X_host = Array(X)
                for b in 1:B
                    @test sum(X_host[:, b]) ≈ 1.0 atol=1e-3
                    @test all(X_host[:, b] .>= -1e-6)
                end
            end
        end

        @testset "Float32 ScalarBox on device" begin
            n, B = 8, 4
            H = Arr(collect(Float32.(I(n)) .* 2.0f0))
            X0 = Arr(fill(0.5f0, n, B))

            f_per_col(x, _, b) = 0.5f0 * dot(x, H * x)
            grad_per_col!(g, x, _, b) = (mul!(g, H, x); g)
            expr = BatchedExpression(f_per_col, grad_per_col!)

            X, result = batch_solve(expr, Box(0.0f0, 1.0f0), X0;
                                    max_iters=500, tol=1.0f-3)
            @test all(result.converged)
            @test eltype(X) === Float32
        end

        @testset "Float32 ProbSimplex on device" begin
            n, B = 6, 3
            H = Arr(collect(Float32.(I(n)) .* 2.0f0))
            X0 = Arr(fill(1.0f0 / n, n, B))

            f_per_col(x, _, b) = 0.5f0 * dot(x, H * x)
            grad_per_col!(g, x, _, b) = (mul!(g, H, x); g)
            expr = BatchedExpression(f_per_col, grad_per_col!)

            X, result = batch_solve(expr, ProbSimplex(1.0f0), X0;
                                    max_iters=500, tol=1.0f-3)
            @test all(result.converged)
            @test eltype(X) === Float32
            X_host = Array(X)
            for b in 1:B
                @test sum(X_host[:, b]) ≈ 1.0f0 atol=1.0f-3
                @test all(X_host[:, b] .>= -1.0f-5)
            end
        end

        @testset "Float32 capped Simplex on device" begin
            n, B = 6, 3
            H = Arr(collect(Float32.(I(n)) .* 2.0f0))
            c = Arr(Float32[-1, -0.5, -0.3, 0.1, 0.2, 0.3])
            X0 = Arr(fill(0.05f0, n, B))

            f_per_col(x, _, b) = 0.5f0 * dot(x, H * x) + dot(c, x)
            grad_per_col!(g, x, _, b) = (mul!(g, H, x); g .+= c; g)
            expr = BatchedExpression(f_per_col, grad_per_col!)

            X, result = batch_solve(expr, Simplex(1.0f0), X0;
                                    max_iters=3000, tol=1.0f-3)
            @test all(result.converged)
            X_host = Array(X)
            for b in 1:B
                @test sum(X_host[:, b]) <= 1.0f0 + 1.0f-3
                @test all(X_host[:, b] .>= -1.0f-5)
            end
        end
    end
end
