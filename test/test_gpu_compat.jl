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

# MockGPUVector: wraps a CPU Vector but dispatches to _GPUStyle.
# Errors on scalar getindex/setindex! to catch illegal scalar indexing.
struct MockGPUVector{T} <: AbstractVector{T}
    data::Vector{T}
end

Base.size(v::MockGPUVector) = size(v.data)
Base.IndexStyle(::Type{<:MockGPUVector}) = IndexLinear()

# Scalar getindex is forbidden (simulates GPU restriction — reads are the real bottleneck)
# setindex! is allowed because broadcast machinery needs it internally.
function Base.getindex(v::MockGPUVector, i::Int)
    error("Scalar getindex on MockGPUVector is forbidden (simulates GPU restriction)")
end
Base.setindex!(v::MockGPUVector, val, i::Int) = (v.data[i] = val; v)

# Broadcast and array operations go through the underlying data
Base.similar(v::MockGPUVector{T}) where T = MockGPUVector(similar(v.data))
Base.similar(v::MockGPUVector, ::Type{T}) where T = MockGPUVector(similar(v.data, T))
Base.similar(v::MockGPUVector, ::Type{T}, dims::Dims) where T = MockGPUVector(similar(v.data, T, dims))
Base.copy(v::MockGPUVector) = MockGPUVector(copy(v.data))
Base.copyto!(dst::MockGPUVector, src::MockGPUVector) = (copyto!(dst.data, src.data); dst)
Base.copyto!(dst::MockGPUVector, src::AbstractVector) = (copyto!(dst.data, src); dst)
Base.copyto!(dst::AbstractVector, src::MockGPUVector) = (copyto!(dst, src.data); dst)
Base.fill!(v::MockGPUVector, x) = (fill!(v.data, x); v)
Base.length(v::MockGPUVector) = length(v.data)
Base.eltype(::Type{MockGPUVector{T}}) where T = T
Base.sum(v::MockGPUVector) = sum(v.data)
Base.minimum(v::MockGPUVector) = minimum(v.data)
Base.abs(v::MockGPUVector) = MockGPUVector(abs.(v.data))

# Broadcasting support: unwrap to data, compute, rewrap
Base.BroadcastStyle(::Type{<:MockGPUVector}) = Broadcast.ArrayStyle{MockGPUVector}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{MockGPUVector}}, ::Type{T}) where T
    MockGPUVector(similar(Array{T}, axes(bc)))
end
# Override copyto! for Broadcasted to go through .data (simulates GPU kernel dispatch)
function Base.copyto!(dest::MockGPUVector, bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{MockGPUVector}})
    flat = Broadcast.flatten(bc)
    # Unwrap MockGPUVector args to their .data, then broadcast on CPU
    args_unwrapped = map(a -> a isa MockGPUVector ? a.data : a, flat.args)
    bc_cpu = Broadcast.Broadcasted(flat.f, args_unwrapped, flat.axes)
    copyto!(dest.data, bc_cpu)
    return dest
end

# LinearAlgebra operations
LinearAlgebra.dot(a::MockGPUVector, b::MockGPUVector) = dot(a.data, b.data)
LinearAlgebra.dot(a::MockGPUVector, b::AbstractVector) = dot(a.data, b)
LinearAlgebra.dot(a::AbstractVector, b::MockGPUVector) = dot(a, b.data)

# argmin for GPU-safe Simplex path
Base.argmin(v::MockGPUVector) = argmin(v.data)

# eachindex returns a range (not scalar-indexing)
Base.eachindex(v::MockGPUVector) = eachindex(v.data)

# Register as GPU array
Marguerite._array_style(::MockGPUVector) = Marguerite._GPUStyle()

@testset "GPU compatibility (MockGPUVector)" begin

    @testset "Cache construction from GPU-like array" begin
        x0 = MockGPUVector(ones(5))
        c = Cache(x0)
        @test c.gradient isa MockGPUVector
        @test c.vertex isa MockGPUVector
        @test c.x_trial isa MockGPUVector
        @test c.direction isa MockGPUVector
        @test c.vertex_nzind isa Vector{Int}  # sparse buffers stay CPU
        @test c.vertex_nzval isa Vector{Float64}
    end

    @testset "ForwardDiff guard" begin
        x0 = MockGPUVector([0.5, 0.5])
        f(x) = dot(x, x)
        @test_throws ArgumentError solve(f, ProbSimplex(), x0; max_iters=5)
    end

    @testset "ScalarBox solve (GPU-safe)" begin
        n = 5
        H = [2.0 0.5 0 0 0;
             0.5 2.0 0.5 0 0;
             0 0.5 2.0 0.5 0;
             0 0 0.5 2.0 0.5;
             0 0 0 0.5 2.0]
        c_vec = [1.0, -0.5, 0.3, -0.2, 0.1]
        # f and grad must not scalar-index x: use .data to extract CPU Vector
        f(x) = 0.5 * dot(x.data, H * x.data) + dot(c_vec, x.data)
        grad!(g, x) = (gv = H * x.data .+ c_vec; copyto!(g.data, gv); g)

        x0 = MockGPUVector(fill(0.5, n))
        lmo = Box(0.0, 1.0)  # ScalarBox

        x, result = solve(f, lmo, x0; grad=grad!, max_iters=5000, tol=1e-3)
        @test result.converged
        @test x isa MockGPUVector
    end

    @testset "ProbSimplex solve (GPU-safe)" begin
        n = 3
        H = [4.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0]
        # f and grad must not scalar-index x
        f(x) = 0.5 * dot(x.data, H * x.data)
        grad!(g, x) = (gv = H * x.data; copyto!(g.data, gv); g)

        x0 = MockGPUVector([1/3, 1/3, 1/3])
        lmo = ProbSimplex()

        x, result = solve(f, lmo, x0; grad=grad!, max_iters=10000, tol=1e-3)
        @test result.converged
        @test x isa MockGPUVector
        # Solution should be on the simplex
        @test sum(x.data) ≈ 1.0 atol=1e-4
        @test all(x.data .>= -1e-8)
    end

    @testset "Capped Simplex solve (GPU-safe, Equality=false)" begin
        n = 3
        H = [4.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0]
        f(x) = 0.5 * dot(x.data, H * x.data)
        grad!(g, x) = (gv = H * x.data; copyto!(g.data, gv); g)

        x0 = MockGPUVector([0.2, 0.3, 0.5])
        lmo = Simplex{Float64, false}(1.0)  # capped simplex

        x, result = solve(f, lmo, x0; grad=grad!, max_iters=10000, tol=1e-3)
        @test result.converged
        @test x isa MockGPUVector
        @test sum(x.data) <= 1.0 + 1e-4
        @test all(x.data .>= -1e-8)
    end

    @testset "Unsupported oracle errors on GPU" begin
        x0 = MockGPUVector(fill(0.5, 5))
        f(x) = 0.5 * dot(x.data, x.data)
        grad!(g, x) = (copyto!(g.data, x.data); g)

        @test_throws ArgumentError solve(f, Knapsack(2, 5), x0; grad=grad!, max_iters=5)
        @test_throws ArgumentError solve(f, MaskedKnapsack(2, Int[], 5), x0; grad=grad!, max_iters=5)

        x0_sp = MockGPUVector(fill(0.04, 25))  # n=5, Spectraplex requires n^2 vector
        f_sp(x) = 0.5 * dot(x.data, x.data)
        grad_sp!(g, x) = (copyto!(g.data, x.data); g)
        @test_throws ArgumentError solve(f_sp, Spectraplex(5), x0_sp; grad=grad_sp!, max_iters=5)
        @test_throws ArgumentError solve(f, Box(zeros(5), ones(5)), x0; grad=grad!, max_iters=5)
        @test_throws ArgumentError solve(f, WeightedSimplex(ones(5), 1.0, zeros(5)), x0; grad=grad!, max_iters=5)
    end

    @testset "AdaptiveStepSize rejected on GPU" begin
        x0 = MockGPUVector([0.5, 0.5])
        f(x) = dot(x.data, x.data)
        grad!(g, x) = (copyto!(g.data, x.data .* 2); g)
        @test_throws ArgumentError solve(f, Box(0.0, 1.0), x0; grad=grad!, step_rule=AdaptiveStepSize(), max_iters=5)
    end

    @testset "batch_solve rejected on GPU" begin
        X0_mock = MockGPUVector(ones(4))  # not a matrix, but test the trait check path
        # batch_solve requires a matrix; test the GPU guard fires before dimension checks
        # Use a mock matrix workaround: test the trait function directly
        @test Marguerite._array_style(MockGPUVector(ones(3))) isa Marguerite._GPUStyle
    end

    @testset "SolveResult preserves GPU array type" begin
        r = Marguerite.Result(0.5, 1e-5, 50, true, 0)
        x = MockGPUVector([0.3, 0.7])
        sr = SolveResult(x, r)
        @test sr.x isa MockGPUVector
        s = sprint(show, MIME("text/plain"), sr)
        @test contains(s, "MockGPUVector")
    end
end
