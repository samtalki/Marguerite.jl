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

Per-column callbacks for `batch_solve` and `batch_bilevel_solve`. Each
callback operates on a single column view of the iterate.

# Fields
- `f_per_col(x_b, θ, b) -> Real` -- objective for problem `b` at point
  `x_b` with shared parameters `θ`. Pass `θ = nothing` for non-parametric
  problems.
- `grad_per_col!(g_b, x_b, θ, b) -> g_b` -- writes the length-`n` gradient
  of problem `b` into `g_b`. Required: there is no auto-gradient fallback
  in batched mode.

`f_per_col` and `grad_per_col!` must be column independent: the value
and gradient at column `b` depend only on `x_b`, not on the other columns
of the batched iterate.
"""
struct BatchedExpression{F, G}
    f_per_col::F
    grad_per_col!::G

    function BatchedExpression(f::F, g!::G) where {F, G}
        g! === nothing && throw(ArgumentError(
            "BatchedExpression requires a per-column gradient. " *
            "Auto-gradient is not implemented for batched mode; supply " *
            "grad_per_col!(g, x, θ, b) explicitly."))
        new{F, G}(f, g!)
    end
end

# Compaction interval — number of FW iterations between active set rebuild
# passes on GPU. CPU skips compaction entirely (the per-column loops
# short-circuit on `c.active[b]`).
const _BATCH_COMPACTION_INTERVAL = 16

# Build per-column closures that hide the `b` index from the scalar diff
# pipeline (`_build_pullback_state`, `_kkt_adjoint_solve_cached`,
# `_cross_derivative_manual`).
@inline _make_fθ_per_col(expr::BatchedExpression, b::Int) =
    let b_=b; (x, θ_) -> expr.f_per_col(x, θ_, b_) end

@inline _make_grad_per_col(expr::BatchedExpression, b::Int) =
    let b_=b; (g, x, θ_) -> expr.grad_per_col!(g, x, θ_, b_) end

"""
    BatchSolveConfig(; kwargs...)

Configuration for `batch_solve`, `batch_bilevel_solve`, and the
corresponding rrules.

# Forward solve
- `max_iters::Int = 10000`
- `tol = 1e-4` -- per-problem convergence tolerance
- `step_rule = MonotonicStepSize()`
- `monotonic::Bool = true` -- reject non-improving updates
- `verbose::Bool = false`

# Differentiation
- `backend_ad = DEFAULT_BACKEND` -- AD backend for first-order gradients
- `hvp_backend = SECOND_ORDER_BACKEND` -- AD backend for HVPs
- `diff_cg_maxiter::Int = 50`
- `diff_cg_tol = 1e-6`
- `diff_lambda = 1e-4` -- starting Tikhonov regularization on the reduced
  Hessian. Increased automatically once if the residual exceeds the
  recovery threshold; the increased value persists across pullback calls
  on the same cached state.
- `assume_interior::Bool = false`

Active set tolerance is fixed at `min(tol, ACTIVE_SET_TOL_CEILING)` (see
`Marguerite.ACTIVE_SET_TOL_CEILING`); there is no separate knob.

Per-call kwargs override matching fields of `config`.
"""
struct BatchSolveConfig{T<:Real, SR, BAD, HVP}
    max_iters::Int
    tol::T
    step_rule::SR
    monotonic::Bool
    verbose::Bool
    backend_ad::BAD
    hvp_backend::HVP
    diff_cg_maxiter::Int
    diff_cg_tol::T
    diff_lambda::T
    assume_interior::Bool
end

function BatchSolveConfig(;
        max_iters::Int = 10000,
        tol::Real = 1e-4,
        step_rule = MonotonicStepSize(),
        monotonic::Bool = true,
        verbose::Bool = false,
        backend_ad = DEFAULT_BACKEND,
        hvp_backend = SECOND_ORDER_BACKEND,
        diff_cg_maxiter::Int = 50,
        diff_cg_tol::Real = 1e-6,
        diff_lambda::Real = 1e-4,
        assume_interior::Bool = false)
    T = promote_type(typeof(float(tol)), typeof(float(diff_cg_tol)), typeof(float(diff_lambda)))
    return BatchSolveConfig{T, typeof(step_rule), typeof(backend_ad), typeof(hvp_backend)}(
        max_iters, T(tol), step_rule, monotonic, verbose,
        backend_ad, hvp_backend,
        diff_cg_maxiter, T(diff_cg_tol), T(diff_lambda), assume_interior)
end

"""
    _merge_config(cfg::BatchSolveConfig, kwargs)

Return a new `BatchSolveConfig` whose fields are taken from `kwargs` where
present and from `cfg` otherwise.
"""
function _merge_config(cfg::BatchSolveConfig, kwargs)
    isempty(kwargs) && return cfg
    overrides = (k => get(kwargs, k, getfield(cfg, k)) for k in fieldnames(BatchSolveConfig))
    BatchSolveConfig(; overrides...)
end

"""
    BatchCache{T, M, V, VI}

Pre-allocated working buffers for batched Frank-Wolfe. Each matrix buffer
is `(n, B)` where `n` is the problem dimension and `B` is the batch size.
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
        # CPU alias invariant: when the matrix buffers are plain Arrays, the
        # device-mirror buffers must alias their host counterparts so
        # `copyto!(mask_dev, accepted)` and `_rebuild_active_indices!` skip
        # the device copy.
        if M <: Array
            mask_dev === accepted || throw(ArgumentError(
                "BatchCache CPU invariant: mask_dev must alias accepted on CPU"))
            active_indices_dev === active_indices_cpu || throw(ArgumentError(
                "BatchCache CPU invariant: active_indices_dev must alias active_indices_cpu on CPU"))
        end
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

Per-problem diagnostics from a batched Frank-Wolfe solve.

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
