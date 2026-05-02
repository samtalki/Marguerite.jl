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
    BatchCache{T, M<:AbstractMatrix{T}, V<:AbstractVector{Bool}}

Pre-allocated working buffers for batched Frank-Wolfe. Each matrix buffer is
`(n, B)` where `n` is the problem dimension and `B` is the batch size.
Per-problem scalars (gap, objective, discards) are `(B,)` vectors.

`mask_dev` mirrors `accepted` on the device so the GPU acceptance broadcast
avoids a per-iter device allocation; on CPU it aliases `accepted`.

No sparse vertex protocol: batched mode always uses dense vertices.
"""
struct BatchCache{T<:Real, M<:AbstractMatrix{T}, V<:AbstractVector{Bool}}
    gradient::M               # (n, B)
    vertex::M                 # (n, B)
    x_trial::M                # (n, B)
    gap::Vector{T}            # (B,)
    objective::Vector{T}      # (B,)
    active::BitVector         # (B,) — true if still iterating
    discards::Vector{Int}     # (B,)
    step_gamma::Vector{T}     # (B,) per-problem step (adaptive only)
    obj_trial::Vector{T}      # (B,) per-problem trial objective (adaptive only)
    reuse_grad::Vector{Bool}  # (B,) skip gradient recompute next iter
    accepted::Vector{Bool}    # (B,) trial accepted this iter (CPU)
    mask_dev::V               # (B,) device mirror of `accepted` (= accepted on CPU)

    function BatchCache{T,M,V}(gradient::M, vertex::M, x_trial::M,
                                gap::Vector{T}, objective::Vector{T},
                                active::BitVector, discards::Vector{Int},
                                step_gamma::Vector{T}, obj_trial::Vector{T},
                                reuse_grad::Vector{Bool}, accepted::Vector{Bool},
                                mask_dev::V) where {T<:Real, M<:AbstractMatrix{T}, V<:AbstractVector{Bool}}
        n, B = size(gradient)
        (size(vertex) == (n, B) && size(x_trial) == (n, B)) ||
            throw(DimensionMismatch(
                "BatchCache matrix buffers must all be ($n, $B) (got vertex=$(size(vertex)), x_trial=$(size(x_trial)))"))
        all(length(v) == B for v in (gap, objective, active, discards,
                                     step_gamma, obj_trial, reuse_grad, accepted, mask_dev)) ||
            throw(DimensionMismatch("BatchCache vector buffers must all have length $B"))
        new{T,M,V}(gradient, vertex, x_trial, gap, objective, active, discards,
                   step_gamma, obj_trial, reuse_grad, accepted, mask_dev)
    end
end

function BatchCache{T}(n::Int, B::Int) where {T<:Real}
    n > 0 || throw(ArgumentError("BatchCache dimension must be positive, got n=$n"))
    B > 0 || throw(ArgumentError("BatchCache batch size must be positive, got B=$B"))
    accepted = zeros(Bool, B)
    BatchCache{T, Matrix{T}, Vector{Bool}}(
        zeros(T, n, B), zeros(T, n, B), zeros(T, n, B),
        zeros(T, B), zeros(T, B), trues(B), zeros(Int, B),
        zeros(T, B), zeros(T, B), zeros(Bool, B), accepted, accepted)
end

function BatchCache(X0::AbstractMatrix{T}) where {T<:Real}
    n, B = size(X0)
    n > 0 || throw(ArgumentError("BatchCache dimension must be positive, got n=$n"))
    B > 0 || throw(ArgumentError("BatchCache batch size must be positive, got B=$B"))
    _zl(X) = fill!(similar(X), zero(T))
    accepted = zeros(Bool, B)
    is_gpu = _array_style(X0) isa _GPUStyle
    mask_dev = is_gpu ? fill!(similar(X0, Bool, B), false) : accepted
    M = typeof(_zl(X0))
    V = typeof(mask_dev)
    BatchCache{T, M, V}(
        _zl(X0), _zl(X0), _zl(X0),
        zeros(T, B), zeros(T, B), trues(B), zeros(Int, B),
        zeros(T, B), zeros(T, B), zeros(Bool, B), accepted, mask_dev)
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

"""
    BatchBilevelResult{T, S}

Wrapper for `batch_bilevel_solve` output. Supports tuple unpacking:
`X, dθ, cg_results = batch_bilevel_solve(...)`.

# Fields
- `X::Matrix{T}` -- `(n, B)` inner solutions
- `theta_grad::S` -- gradient ``\\nabla_\\theta L(X^*(\\theta))`` (summed across problems)
- `cg_results::Vector{CGResult{T}}` -- per-problem CG diagnostics
"""
struct BatchBilevelResult{T<:Real, S}
    X::Matrix{T}
    theta_grad::S
    cg_results::Vector{CGResult{T}}
end

Base.iterate(br::BatchBilevelResult) = (br.X, Val(:tg))
Base.iterate(br::BatchBilevelResult, ::Val{:tg}) = (br.theta_grad, Val(:cg))
Base.iterate(br::BatchBilevelResult, ::Val{:cg}) = (br.cg_results, nothing)
Base.iterate(::BatchBilevelResult, ::Nothing) = nothing
Base.length(::BatchBilevelResult) = 3
Base.IteratorSize(::Type{<:BatchBilevelResult}) = Base.HasLength()
