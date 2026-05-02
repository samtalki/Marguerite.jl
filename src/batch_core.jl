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
    BatchedExpression(f_per_col, grad_per_col!)
    BatchedExpression(f_per_col, grad_per_col!, cross_hvp)

User facing input to `batch_solve` and `batch_bilevel_solve`. Per-problem
callbacks evaluate one column of the batched problem at a time, so the
implicit differentiation pipeline does not have to round-trip a full `(n, B)`
buffer per column.

# Fields
- `f_per_col(x_b, θ, b) -> Real` -- objective for problem `b` at point `x_b`
  with shared parameters `θ`. For non-parametric problems pass `θ = nothing`.
- `grad_per_col!(g_b, x_b, θ, b) -> g_b` -- writes the length-`n` gradient of
  problem `b` into `g_b`. Same `θ` convention.
- `cross_hvp(out_θ, x_b, θ, u_b, b) -> out_θ` -- optional. If provided, writes
  ``-(\\partial^2 f_b / \\partial x \\partial \\theta)^\\top u_b`` into
  `out_θ`. Used to bypass AD on the cross derivative when an analytic form is
  cheaper. Defaults to `nothing`, in which case the rrule falls back to
  `grad_per_col!` plus AD over `θ`, or a joint HVP on `f_per_col`.

`f_per_col` and `grad_per_col!` must be column independent: the value and
gradient at column `b` depend only on `x_b`, not on the other columns of the
batched iterate. This is the API contract that lets the rrule do `O(n·B)`
work per pullback rather than `O(n·B²)`.
"""
struct BatchedExpression{F, G, H}
    f_per_col::F
    grad_per_col!::G
    cross_hvp::H
end
BatchedExpression(f, g!) = BatchedExpression(f, g!, nothing)

"""
    BatchSolveConfig(; kwargs...)

Single configuration object for `batch_solve`, `batch_bilevel_solve`, and
the corresponding rrules. Replaces the long keyword list previously spread
across every public method.

# Forward solve
- `max_iters::Int = 10000`
- `tol::Real = 1e-4` -- per-problem convergence tolerance
- `step_rule = MonotonicStepSize()`
- `monotonic::Bool = true` -- reject non-improving updates
- `verbose::Bool = false`
- `compaction_interval::Int = 16` -- iterations between active set
  compaction passes on GPU. CPU paths short-circuit per problem and
  ignore this knob.

# Differentiation
- `backend_ad = DEFAULT_BACKEND` -- AD backend for first-order gradients
- `hvp_backend = SECOND_ORDER_BACKEND` -- AD backend for HVPs
- `diff_cg_maxiter::Int = 50`
- `diff_cg_tol::Real = 1e-6`
- `diff_lambda::Real = 1e-4` -- starting Tikhonov regularization on the
  reduced Hessian. Increased automatically once if the residual exceeds
  the recovery threshold; the increased value persists across pullback
  calls on the same `_PullbackState`.
- `assume_interior::Bool = false`

# Active set identification
- `active_set_tol::Real = ACTIVE_SET_TOL_CEILING`
- `refine_active_set::Bool = false` -- run multiplier-sign refinement
  after threshold-based identification (CCOpt-style, opt in)

Per-call keyword overrides flow through the public methods and override
the matching field on the supplied (or default) config.
"""
Base.@kwdef mutable struct BatchSolveConfig{SR}
    max_iters::Int = 10000
    tol::Real = 1e-4
    step_rule::SR = MonotonicStepSize()
    monotonic::Bool = true
    verbose::Bool = false
    compaction_interval::Int = 16
    backend_ad = DEFAULT_BACKEND
    hvp_backend = SECOND_ORDER_BACKEND
    diff_cg_maxiter::Int = 50
    diff_cg_tol::Real = 1e-6
    diff_lambda::Real = 1e-4
    assume_interior::Bool = false
    active_set_tol::Real = 1e-7
    refine_active_set::Bool = false
end

"""
    _merge_config(cfg::BatchSolveConfig, kwargs)

Return a new `BatchSolveConfig` whose fields are taken from `kwargs` where
present and from `cfg` otherwise. Used by the public methods to honor
per-call keyword overrides.
"""
function _merge_config(cfg::BatchSolveConfig, kwargs)
    isempty(kwargs) && return cfg
    overrides = (k => get(kwargs, k, getfield(cfg, k)) for k in fieldnames(BatchSolveConfig))
    BatchSolveConfig(; overrides...)
end

"""
    BatchPullbackDiagnostic(residual_rel, cg_iters, reduced_dim, lambda)

Per-problem diagnostics from the rrule pullback. Aggregated into
`BatchResult.diagnostics`.
"""
struct BatchPullbackDiagnostic
    residual_rel::Float64
    cg_iters::Int
    reduced_dim::Int
    lambda::Float64
end

"""
    BatchCache{T, M, V}

Pre-allocated working buffers for batched Frank-Wolfe. Each matrix buffer is
`(n, B)` where `n` is the problem dimension and `B` is the batch size.
Per-problem scalars (gap, objective, discards) are `(B,)` vectors.

`mask_dev` mirrors `accepted` on the device so the GPU acceptance broadcast
avoids a per-iter device allocation; on CPU it aliases `accepted`.

`active_indices_dev` and `nactive` carry the active subset of problems for
narrowing GPU kernel launches as problems converge.

No sparse vertex protocol: batched mode always uses dense vertices.
"""
struct BatchCache{T<:Real, M<:AbstractMatrix{T}, V<:AbstractVector{Bool}, VI<:AbstractVector{Int32}}
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
    active_indices_cpu::Vector{Int32}  # (B,) host staging buffer; aliases active_indices_dev on CPU
    active_indices_dev::VI    # (B,) device-resident active root indices, valid for [1:nactive]
    nactive::Base.RefValue{Int}  # how many of active_indices_dev are valid
    oracle_dev_cache::Base.RefValue{Any}  # last (lmo, args...) tuple device-prepared for the kernel

    function BatchCache{T,M,V,VI}(gradient::M, vertex::M, x_trial::M,
                                   gap::Vector{T}, objective::Vector{T},
                                   active::BitVector, discards::Vector{Int},
                                   step_gamma::Vector{T}, obj_trial::Vector{T},
                                   reuse_grad::Vector{Bool}, accepted::Vector{Bool},
                                   mask_dev::V, active_indices_cpu::Vector{Int32},
                                   active_indices_dev::VI,
                                   nactive::Base.RefValue{Int},
                                   oracle_dev_cache::Base.RefValue{Any}) where {T<:Real, M<:AbstractMatrix{T}, V<:AbstractVector{Bool}, VI<:AbstractVector{Int32}}
        n, B = size(gradient)
        (size(vertex) == (n, B) && size(x_trial) == (n, B)) ||
            throw(DimensionMismatch(
                "BatchCache matrix buffers must all be ($n, $B) (got vertex=$(size(vertex)), x_trial=$(size(x_trial)))"))
        all(length(v) == B for v in (gap, objective, active, discards,
                                     step_gamma, obj_trial, reuse_grad, accepted, mask_dev,
                                     active_indices_cpu, active_indices_dev)) ||
            throw(DimensionMismatch("BatchCache vector buffers must all have length $B"))
        new{T,M,V,VI}(gradient, vertex, x_trial, gap, objective, active, discards,
                      step_gamma, obj_trial, reuse_grad, accepted, mask_dev,
                      active_indices_cpu, active_indices_dev, nactive, oracle_dev_cache)
    end
end

function BatchCache{T}(n::Int, B::Int) where {T<:Real}
    n > 0 || throw(ArgumentError("BatchCache dimension must be positive, got n=$n"))
    B > 0 || throw(ArgumentError("BatchCache batch size must be positive, got B=$B"))
    accepted = zeros(Bool, B)
    active_indices = collect(Int32, 1:B)
    BatchCache{T, Matrix{T}, Vector{Bool}, Vector{Int32}}(
        zeros(T, n, B), zeros(T, n, B), zeros(T, n, B),
        zeros(T, B), zeros(T, B), trues(B), zeros(Int, B),
        zeros(T, B), zeros(T, B), zeros(Bool, B), accepted, accepted,
        active_indices, active_indices, Ref(B), Ref{Any}(nothing))
end

function BatchCache(X0::AbstractMatrix{T}) where {T<:Real}
    n, B = size(X0)
    n > 0 || throw(ArgumentError("BatchCache dimension must be positive, got n=$n"))
    B > 0 || throw(ArgumentError("BatchCache batch size must be positive, got B=$B"))
    backend = KernelAbstractions.get_backend(X0)
    _zl(X) = fill!(similar(X), zero(T))
    accepted = zeros(Bool, B)
    is_cpu = backend isa KernelAbstractions.CPU
    mask_dev = is_cpu ? accepted : fill!(similar(X0, Bool, B), false)
    active_indices_cpu = collect(Int32, 1:B)
    active_indices_dev = is_cpu ? active_indices_cpu : adapt(backend, active_indices_cpu)
    M = typeof(_zl(X0))
    V = typeof(mask_dev)
    VI = typeof(active_indices_dev)
    BatchCache{T, M, V, VI}(
        _zl(X0), _zl(X0), _zl(X0),
        zeros(T, B), zeros(T, B), trues(B), zeros(Int, B),
        zeros(T, B), zeros(T, B), zeros(Bool, B), accepted, mask_dev,
        active_indices_cpu, active_indices_dev, Ref(B), Ref{Any}(nothing))
end

"""
    _rebuild_active_indices!(c::BatchCache)

Rebuild `c.active_indices_dev[1:c.nactive[]]` from `c.active::BitVector`.
Returns the new `nactive`. One walk fills `c.active_indices_cpu`; on GPU
it then `copyto!`s into `c.active_indices_dev` (aliased on CPU).
"""
function _rebuild_active_indices!(c::BatchCache)
    n = 0
    @inbounds for b in eachindex(c.active)
        if c.active[b]
            n += 1
            c.active_indices_cpu[n] = Int32(b)
        end
    end
    if c.active_indices_dev !== c.active_indices_cpu
        copyto!(view(c.active_indices_dev, 1:n), view(c.active_indices_cpu, 1:n))
    end
    c.nactive[] = n
    return n
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
- `diagnostics::Vector{BatchPullbackDiagnostic}` -- per-problem pullback
  diagnostics, populated by the rrule. Empty after a forward-only solve.
"""
struct BatchResult{T<:Real}
    objectives::Vector{T}
    gaps::Vector{T}
    iterations::Int
    converged::BitVector
    discards::Vector{Int}
    diagnostics::Vector{BatchPullbackDiagnostic}
end

# Forward-only solves leave diagnostics empty
BatchResult(objectives::Vector{T}, gaps::Vector{T}, iterations::Int,
            converged::BitVector, discards::Vector{Int}) where {T<:Real} =
    BatchResult(objectives, gaps, iterations, converged, discards, BatchPullbackDiagnostic[])

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
