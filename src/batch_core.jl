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

"""
    BatchCache{T, M<:AbstractMatrix{T}}

Pre-allocated working buffers for batched Frank-Wolfe. Each matrix buffer is
`(n, B)` where `n` is the problem dimension and `B` is the batch size.
Per-problem scalars (gap, objective, discards) are `(B,)` vectors.

No sparse vertex protocol: batched mode always uses dense vertices.
"""
struct BatchCache{T<:Real, M<:AbstractMatrix{T}}
    gradient::M          # (n, B)
    vertex::M            # (n, B)
    x_trial::M           # (n, B)
    gap::Vector{T}       # (B,)
    objective::Vector{T} # (B,)
    active::BitVector    # (B,) convergence mask — true if still iterating
    discards::Vector{Int} # (B,)

    function BatchCache{T,M}(gradient::M, vertex::M, x_trial::M,
                             gap::Vector{T}, objective::Vector{T},
                             active::BitVector, discards::Vector{Int}) where {T<:Real, M<:AbstractMatrix{T}}
        n, B = size(gradient)
        (size(vertex) == (n, B) && size(x_trial) == (n, B)) ||
            throw(DimensionMismatch(
                "BatchCache matrix buffers must all be ($n, $B) (got vertex=$(size(vertex)), x_trial=$(size(x_trial)))"))
        (length(gap) == B && length(objective) == B && length(active) == B && length(discards) == B) ||
            throw(DimensionMismatch(
                "BatchCache vector buffers must all have length $B (got gap=$(length(gap)), objective=$(length(objective)), active=$(length(active)), discards=$(length(discards)))"))
        new{T,M}(gradient, vertex, x_trial, gap, objective, active, discards)
    end
end

function BatchCache{T}(n::Int, B::Int) where {T<:Real}
    n > 0 || throw(ArgumentError("BatchCache dimension must be positive, got n=$n"))
    B > 0 || throw(ArgumentError("BatchCache batch size must be positive, got B=$B"))
    BatchCache{T, Matrix{T}}(
        zeros(T, n, B), zeros(T, n, B), zeros(T, n, B),
        zeros(T, B), zeros(T, B), trues(B), zeros(Int, B))
end

function BatchCache(X0::AbstractMatrix{T}) where {T<:Real}
    n, B = size(X0)
    n > 0 || throw(ArgumentError("BatchCache dimension must be positive, got n=$n"))
    B > 0 || throw(ArgumentError("BatchCache batch size must be positive, got B=$B"))
    _zl(X) = fill!(similar(X), zero(T))
    M = typeof(_zl(X0))
    BatchCache{T, M}(
        _zl(X0), _zl(X0), _zl(X0),
        zeros(T, B), zeros(T, B), trues(B), zeros(Int, B))
end

"""
    BatchResult{T<:Real}

Diagnostics from a batched Frank-Wolfe solve.

# Fields
- `objectives::Vector{T}` -- final per-problem objective values
- `gaps::Vector{T}` -- final per-problem FW gaps
- `iterations::Int` -- total lockstep iterations
- `converged::BitVector` -- per-problem convergence flags
- `discards::Vector{Int}` -- per-problem rejected non-improving updates
"""
struct BatchResult{T<:Real}
    objectives::Vector{T}
    gaps::Vector{T}
    iterations::Int
    converged::BitVector
    discards::Vector{Int}
end

"""
    BatchSolveResult{T, M<:AbstractMatrix{T}}

Wrapper for `batch_solve` output. Supports tuple unpacking:
`X, result = batch_solve(...)`.

# Fields
- `X::M` -- `(n, B)` solution matrix (columns are solutions)
- `result::BatchResult{T}` -- per-problem diagnostics
"""
struct BatchSolveResult{T<:Real, M<:AbstractMatrix{T}}
    X::M
    result::BatchResult{T}
end

Base.iterate(br::BatchSolveResult) = (br.X, Val(:result))
Base.iterate(br::BatchSolveResult, ::Val{:result}) = (br.result, nothing)
Base.iterate(::BatchSolveResult, ::Nothing) = nothing
Base.length(::BatchSolveResult) = 2
Base.IteratorSize(::Type{<:BatchSolveResult}) = Base.HasLength()
