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
using KernelAbstractions

# A trivial stub backend used to drive Marguerite's dispatch into its
# `::KernelAbstractions.Backend` methods without needing a real device. The
# stub is registered for `MockGPUArray{N, T}`, which wraps a plain `Array`
# and reports its backend as `MockGPUBackend()`.
#
# This lets us exercise Marguerite's CPU-vs-non-CPU dispatch (guards,
# unsupported-oracle errors, and the parent walk through views/reshapes)
# without spinning up a real GPU. The full forward and rrule paths against
# real backends live in test/test_gpu_backend.jl.

struct MockGPUBackend <: KernelAbstractions.Backend end

mutable struct MockGPUArray{T, N} <: AbstractArray{T, N}
    data::Array{T, N}
end
MockGPUArray(a::Array{T, N}) where {T, N} = MockGPUArray{T, N}(a)

Base.size(a::MockGPUArray) = size(a.data)
Base.IndexStyle(::Type{<:MockGPUArray}) = IndexLinear()
Base.getindex(a::MockGPUArray, i::Int) = a.data[i]
Base.setindex!(a::MockGPUArray, v, i::Int) = (a.data[i] = v; a)
Base.similar(a::MockGPUArray) = MockGPUArray(similar(a.data))
Base.similar(a::MockGPUArray, ::Type{T}) where T = MockGPUArray(similar(a.data, T))
Base.similar(a::MockGPUArray, ::Type{T}, dims::Dims) where T = MockGPUArray(similar(a.data, T, dims))

KernelAbstractions.get_backend(::MockGPUArray) = MockGPUBackend()

@testset "GPU compatibility (MockGPUBackend)" begin

    @testset "Cache construction reads backend from x0" begin
        x0_cpu = ones(5)
        c_cpu = Cache(x0_cpu)
        @test c_cpu.gradient isa Vector{Float64}

        x0 = MockGPUArray(ones(5))
        c = Cache(x0)
        @test c.gradient isa MockGPUArray
        @test c.vertex isa MockGPUArray
        @test c.x_trial isa MockGPUArray
        @test c.direction isa MockGPUArray
        # sparse buffers stay CPU
        @test c.vertex_nzind isa Vector{Int}
        @test c.vertex_nzval isa Vector{Float64}
    end

    @testset "BatchCache construction reads backend from X0" begin
        X0_cpu = ones(3, 4)
        c_cpu = BatchCache(X0_cpu)
        @test c_cpu.gradient isa Matrix{Float64}
        @test KernelAbstractions.get_backend(c_cpu.gradient) isa KernelAbstractions.CPU

        X0 = MockGPUArray(ones(3, 4))
        c = BatchCache(X0)
        @test c.gradient isa MockGPUArray
        @test KernelAbstractions.get_backend(c.gradient) === MockGPUBackend()
    end

    @testset "ForwardDiff guard on solve with non-CPU backend" begin
        # solve auto-grad needs ForwardDiff, which doesn't run on a stub backend.
        # The guard fires at the keyword check before any kernel launch.
        x0 = MockGPUArray([0.5, 0.5])
        f(x) = dot(x.data, x.data)
        @test_throws ArgumentError solve(f, ProbSimplex(), x0; max_iters=5)
    end

    @testset "AdaptiveStepSize rejected on non-CPU backend" begin
        x0 = MockGPUArray([0.5, 0.5])
        f(x) = dot(x.data, x.data)
        grad!(g, x) = (copyto!(g.data, x.data .* 2); g)
        @test_throws ArgumentError solve(f, Box(0.0, 1.0), x0;
                                          grad=grad!, step_rule=AdaptiveStepSize(), max_iters=5)
    end

    @testset "AdaptiveStepSize rejected on non-CPU backend (4-arg parametric)" begin
        # The 4-arg parametric solve previously bypassed the GPU+adaptive guard
        # on the manual-grad branch by routing straight to _solve_core.
        x0 = MockGPUArray([0.5, 0.5])
        fp(x, θ) = θ * dot(x.data, x.data)
        gradp!(g, x, θ) = (copyto!(g.data, x.data .* (2 * θ)); g)
        θ = 1.0
        @test_throws ArgumentError solve(fp, Box(0.0, 1.0), x0, θ;
                                          grad=gradp!, step_rule=AdaptiveStepSize(), max_iters=5)
    end

    @testset "Unsupported oracles on non-CPU backend" begin
        x0 = MockGPUArray(fill(0.5, 5))
        f(x) = 0.5 * dot(x.data, x.data)
        grad!(g, x) = (copyto!(g.data, x.data); g)

        @test_throws ArgumentError solve(f, Knapsack(2, 5), x0; grad=grad!, max_iters=5)
        @test_throws ArgumentError solve(f, MaskedKnapsack(2, Int[], 5), x0; grad=grad!, max_iters=5)
        @test_throws ArgumentError solve(f, Box(zeros(5), ones(5)), x0; grad=grad!, max_iters=5)
        @test_throws ArgumentError solve(f, WeightedSimplex(ones(5), 1.0, zeros(5)), x0; grad=grad!, max_iters=5)

        x0_sp = MockGPUArray(fill(0.04, 25))
        f_sp(x) = 0.5 * dot(x.data, x.data)
        grad_sp!(g, x) = (copyto!(g.data, x.data); g)
        @test_throws ArgumentError solve(f_sp, Spectraplex(5), x0_sp; grad=grad_sp!, max_iters=5)
    end

    @testset "Backend trait survives views and reshapes" begin
        # Finding 2: views and reshapes of a non-CPU array must keep their backend.
        # KernelAbstractions.get_backend handles parent walking through these wrappers.
        X = MockGPUArray(ones(3, 4))
        @test KernelAbstractions.get_backend(view(X, :, 1)) === MockGPUBackend()
        @test KernelAbstractions.get_backend(reshape(X, 12)) === MockGPUBackend()

        # Same for plain Array → CPU
        A = ones(3, 4)
        @test KernelAbstractions.get_backend(view(A, :, 1)) isa KernelAbstractions.CPU
        @test KernelAbstractions.get_backend(reshape(A, 12)) isa KernelAbstractions.CPU
    end

    @testset "BatchedExpression unsupported oracle on non-CPU" begin
        n = 3; B = 2
        X0 = MockGPUArray(fill(0.5, n, B))
        f_per_col(x, _, b) = sum(x)
        grad_per_col!(g, x, _, b) = (fill!(g, 1.0); g)
        expr = BatchedExpression(f_per_col, grad_per_col!)
        # Knapsack is CPU-only — expect error from the non-CPU dispatch path.
        @test_throws ArgumentError batch_solve(expr, Knapsack(2, n), X0; max_iters=1)
    end

    @testset "SolveResult preserves array type" begin
        r = Marguerite.Result(0.5, 1e-5, 50, true, 0)
        x = MockGPUArray([0.3, 0.7])
        sr = SolveResult(x, r)
        @test sr.x isa MockGPUArray
        s = sprint(show, MIME("text/plain"), sr)
        @test contains(s, "MockGPUArray")
    end

    @testset "ParametricOracle without θ is rejected" begin
        plmo = ParametricBox(θ -> zeros(2), θ -> ones(2))
        f(x) = dot(x, x)
        grad!(g, x) = (g .= 2 .* x; g)
        # Without θ, the parametric oracle cannot be reduced. The 3-arg solve
        # should raise an actionable ArgumentError instead of an obscure
        # MethodError(getindex, ...) deep inside materialize.
        @test_throws ArgumentError solve(f, plmo, [0.5, 0.5]; grad=grad!, max_iters=5)
    end

    @testset "BatchedExpression rejects nothing gradient" begin
        f_per_col(x, _, b) = sum(x)
        @test_throws ArgumentError BatchedExpression(f_per_col, nothing)
    end
end
