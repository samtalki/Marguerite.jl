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
# if none is loadable.

using Marguerite
using Test
using LinearAlgebra: dot

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
        catch
        end
    end
    if backend === nothing
        try
            @eval using AMDGPU
            if AMDGPU.functional()
                backend = (name="AMDGPU", arr=AMDGPU.ROCArray)
            end
        catch
        end
    end
    backend
end

# Metal does not support Float64 on the GPU; the F64 testsets below would
# error at `MtlMatrix{Float64}` construction. CUDA / AMDGPU support F64.
const SUPPORTS_F64 = GPU_BACKEND === nothing ? false : GPU_BACKEND.name != "Metal"

@testset "GPU Backend" begin
    if GPU_BACKEND === nothing
        @info "No functional GPU backend found — skipping device tests"
        @test true  # placeholder so the testset isn't empty
    else
        @info "Running GPU backend tests on $(GPU_BACKEND.name)"
        Arr = GPU_BACKEND.arr

        @testset "ScalarBox: batch_solve on device (F64)" begin
            if !SUPPORTS_F64
                @info "skipping F64 GPU testset on $(GPU_BACKEND.name) (Float64 unsupported)"
                @test true  # placeholder so the testset isn't empty
            else
                n, B = 8, 4
                H = collect(I(n) .* 2.0)
                X0_cpu = fill(0.5, n, B)
                X0 = Arr(X0_cpu)

                f_batch(X) = [0.5 * dot(X[:, b], H * X[:, b]) for b in 1:B]
                grad_batch!(G, X) = (G .= H * X)

                X, result = batch_solve(f_batch, Box(0.0, 1.0), X0;
                                        grad_batch=grad_batch!,
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
                @test true  # placeholder so the testset isn't empty
            else
                n, B = 6, 3
                H = collect(I(n) .* 2.0)
                X0_cpu = fill(1.0 / n, n, B)
                X0 = Arr(X0_cpu)

                f_batch(X) = [0.5 * dot(X[:, b], H * X[:, b]) for b in 1:B]
                grad_batch!(G, X) = (G .= H * X)

                X, result = batch_solve(f_batch, ProbSimplex(), X0;
                                        grad_batch=grad_batch!,
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
            H32 = collect(Float32.(I(n)) .* 2.0f0)
            X0 = Arr(fill(0.5f0, n, B))

            f_batch(X) = [0.5f0 * dot(X[:, b], H32 * X[:, b]) for b in 1:B]
            grad_batch!(G, X) = (G .= H32 * X)

            X, result = batch_solve(f_batch, Box(0.0f0, 1.0f0), X0;
                                    grad_batch=grad_batch!,
                                    max_iters=500, tol=1.0f-3)
            @test all(result.converged)
            @test eltype(X) === Float32
        end
    end
end
