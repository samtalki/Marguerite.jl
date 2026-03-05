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
    _solve_core(f, ∇f!, lmo, x0; kwargs...) -> (x, Result)

Core Frank-Wolfe loop. Requires `lmo <: AbstractOracle` for dispatch on
[`_lmo_and_gap!`](@ref) specializations.

Callers should use [`solve`](@ref) instead; this is an internal function.
"""
function _solve_core(f::F, ∇f!::G, lmo::L, x0::AbstractVector;
               max_iters::Int=1000, tol::Real=1e-7,
               step_rule::S=MonotonicStepSize(), monotonic::Bool=true,
               verbose::Bool=false,
               cache::Union{Cache, Nothing}=nothing) where {F, G, L<:AbstractOracle, S}
    x = copy(x0)
    T = eltype(x)
    n = length(x)
    if cache !== nothing && length(cache.gradient) != n
        throw(DimensionMismatch(
            "Cache dimension ($(length(cache.gradient))) ≠ x0 dimension ($n)"))
    end
    c = something(cache, Cache{T}(n))

    obj = f(x)
    fw_gap = T(Inf)
    discards = 0
    converged = false
    reuse_grad = false
    final_iter = max_iters

    if verbose
        @printf("  %6s   %13s   %13s\n", "Iter", "Objective", "FW Gap")
        println("  ──────   ─────────────   ─────────────")
    end

    @inbounds for t in 0:(max_iters - 1)
        if !reuse_grad
            ∇f!(c.gradient, x)
        end

        fw_gap, nnz = _lmo_and_gap!(lmo, c, x, n)

        # Convergence check
        if fw_gap ≤ tol * (one(T) + abs(obj))
            converged = true
            final_iter = t
            break
        end

        # AdaptiveStepSize needs the dense vertex buffer
        _ensure_vertex!(c, nnz, step_rule)

        γ, obj_cached = _compute_step(step_rule, t, f, x, c.gradient, c.vertex, obj, c.x_trial, c.direction)

        # Skip when AdaptiveStepSize already wrote x_trial during backtracking
        if obj_cached === nothing
            _trial_update!(c, x, γ, nnz, n)
        end

        obj_trial = something(obj_cached, f(c.x_trial))

        if !isfinite(obj_trial)
            @warn "solve: non-finite objective ($obj_trial) at iteration $t, discarding step" maxlog=3
            reuse_grad = true
            discards += 1
            continue
        end

        if monotonic && obj_trial > obj + eps(T) * max(one(T), abs(obj))
            reuse_grad = true
            discards += 1
            continue
        end

        copyto!(x, c.x_trial)
        obj = obj_trial
        reuse_grad = false

        if verbose && (t % 50 == 0 || t == max_iters - 1)
            @printf("  %6d   %13.6e   %13.4e\n", t, obj, fw_gap)
        end
    end

    if verbose
        if converged
            @printf("  Converged in %d iterations (gap=%.4e ≤ tol)\n", final_iter, fw_gap)
        else
            @printf("  Did not converge after %d iterations (gap=%.4e)\n", max_iters, fw_gap)
        end
    end

    return SolveResult(x, Result(obj, fw_gap, final_iter, converged, discards))
end

# ------------------------------------------------------------------
# Sparse vertex helpers
# ------------------------------------------------------------------

"""
    _ensure_vertex!(c::Cache, nnz, step_rule)

Materialize the dense vertex buffer `c.vertex` from the sparse representation
(`c.vertex_nzind[1:nnz]`, `c.vertex_nzval[1:nnz]`).
When `nnz = 0`, fills the vertex buffer with zeros (origin vertex).
No-op when `nnz = -1` (already dense) or for `MonotonicStepSize`.
"""
function _ensure_vertex!(c::Cache{T}, nnz::Int, step_rule) where T<:Real
    nnz < 0 && return
    fill!(c.vertex, zero(T))
    @inbounds for j in 1:nnz
        c.vertex[c.vertex_nzind[j]] = c.vertex_nzval[j]
    end
end
@inline _ensure_vertex!(c::Cache, nnz::Int, ::MonotonicStepSize) = nothing

"""
    _trial_update!(c::Cache, x, γ, nnz, n)

Compute the trial point `x_trial = (1-γ)*x + γ*v` using the sparse vertex
representation when available. When `nnz ≥ 0`, avoids touching the dense
vertex buffer by scaling `x` by `(1-γ)` and adding sparse corrections.
When `nnz = -1`, uses the equivalent form `x + γ*(v - x)`.
"""
function _trial_update!(c::Cache{T}, x, γ, nnz::Int, n::Int) where T
    if nnz < 0  # dense vertex
        @inbounds @simd for i in 1:n
            c.x_trial[i] = x[i] + γ * (c.vertex[i] - x[i])
        end
    else  # sparse vertex (including nnz=0 → just scale)
        omγ = one(T) - γ
        @inbounds @simd for i in 1:n
            c.x_trial[i] = omγ * x[i]
        end
        @inbounds for j in 1:nnz
            c.x_trial[c.vertex_nzind[j]] += γ * c.vertex_nzval[j]
        end
    end
end

# Step size dispatch: simple rules take only t; adaptive rules get full state.
# Returns (γ, obj_trial_or_nothing).
# Contract: if obj_trial_or_nothing !== nothing, buffer MUST contain the
# corresponding trial point x + γ*(vertex - x), and dir MUST contain vertex - x.
_compute_step(rule, t, f, x, gradient, vertex, obj, buffer, dir) = (eltype(x)(rule(t)), nothing)
function _compute_step(rule::AdaptiveStepSize, t, f, x, gradient, vertex, obj, buffer, dir)
    return rule(t, f, x, gradient, vertex, obj, buffer, dir)
end

"""
    _to_oracle(lmo) -> AbstractOracle
    _to_oracle(lmo, θ) -> AbstractOracle

Normalize `lmo` to an `AbstractOracle`. If `lmo` is a `ParametricOracle` and
`θ` is provided, materializes the constraint set at `θ`.
"""
_to_oracle(lmo::AbstractOracle) = lmo
_to_oracle(lmo::ParametricOracle, θ) = materialize(lmo, θ)
_to_oracle(lmo, _=nothing) = FunctionOracle(lmo)

# ------------------------------------------------------------------
# Public API: 2 methods (no θ, with θ)
# ------------------------------------------------------------------

"""
    solve(f, lmo, x0; grad=nothing, kwargs...) -> (x, Result)

Solve

```math
\\min_{x \\in C} f(x)
```

via the Frank-Wolfe algorithm.

`lmo` is a linear minimization oracle — any callable `(v, g) -> v` or `<: AbstractOracle`.

# Keyword Arguments
- `grad`: in-place gradient `grad(g, x)`. If `nothing` (default), computed automatically
  via `DifferentiationInterface` using `backend`.
- `backend`: AD backend (default: `DEFAULT_BACKEND`)
- `max_iters::Int = 1000`: maximum iterations
- `tol::Real = 1e-7`: convergence tolerance (``\\mathrm{gap} \\le \\mathrm{tol} \\cdot (1 + |f(x)|)``)
- `step_rule = MonotonicStepSize()`: step size rule (callable `t -> γ`)
- `monotonic::Bool = true`: reject non-improving updates
- `verbose::Bool = false`: print progress
- `cache::Union{Cache, Nothing} = nothing`: pre-allocated buffers
"""
@inline function solve(f, lmo, x0::AbstractVector;
                       grad=nothing,
                       backend=DEFAULT_BACKEND,
                       cache::Union{Cache, Nothing}=nothing,
                       max_iters::Int=1000, tol::Real=1e-7,
                       step_rule=MonotonicStepSize(), monotonic::Bool=true,
                       verbose::Bool=false)
    oracle = lmo isa AbstractOracle ? lmo : FunctionOracle(lmo)
    c = if cache !== nothing
        cache
    else
        Cache{eltype(x0)}(length(x0))
    end
    if grad === nothing
        prep = DI.prepare_gradient(f, backend, x0)
        ∇f!(g, x_) = DI.gradient!(f, g, prep, backend, x_)
        return _solve_core(f, ∇f!, oracle, x0; cache=c, max_iters=max_iters,
                           tol=tol, step_rule=step_rule, monotonic=monotonic, verbose=verbose)
    else
        return _solve_core(f, grad, oracle, x0; cache=c, max_iters=max_iters,
                           tol=tol, step_rule=step_rule, monotonic=monotonic, verbose=verbose)
    end
end

"""
    solve(f, lmo, x0, θ; grad=nothing, kwargs...) -> (x, Result)

Solve

```math
\\min_{x \\in C(\\theta)} f(x, \\theta)
```

with parameters ``\\theta``.

If `lmo` is a [`ParametricOracle`](@ref), the constraint set ``C(\\theta)`` is materialized
at ``\\theta`` via [`materialize`](@ref). Otherwise, ``C`` is fixed.

A `ChainRulesCore.rrule` enables ``\\partial x^* / \\partial \\theta`` via implicit differentiation.

# Keyword Arguments
- `grad`: in-place gradient `grad(g, x, θ)`. If `nothing` (default), auto-computed.
- `backend`: AD backend for first-order gradients
- `hvp_backend`: AD backend for Hessian-vector products
- `diff_cg_maxiter::Int=50`: max CG iterations for the Hessian solve
- `diff_cg_tol::Real=1e-6`: CG convergence tolerance
- `diff_lambda::Real=1e-4`: Tikhonov regularization
"""
@inline function solve(f, lmo, x0::AbstractVector, θ;
                       grad=nothing,
                       backend=DEFAULT_BACKEND,
                       hvp_backend=SECOND_ORDER_BACKEND,
                       diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_lambda::Real=1e-4,
                       cache::Union{Cache, Nothing}=nothing,
                       max_iters::Int=1000, tol::Real=1e-7,
                       step_rule=MonotonicStepSize(), monotonic::Bool=true,
                       verbose::Bool=false)
    if lmo isa ParametricOracle
        oracle = materialize(lmo, θ)
    else
        oracle = lmo isa AbstractOracle ? lmo : FunctionOracle(lmo)
    end
    fθ(x) = f(x, θ)
    kw = (; cache, max_iters, tol, step_rule, monotonic, verbose)
    if grad === nothing
        return solve(fθ, oracle, x0; backend=backend, kw...)
    else
        ∇fθ!(g, x) = grad(g, x, θ)
        return _solve_core(fθ, ∇fθ!, oracle, x0; kw...)
    end
end

# ------------------------------------------------------------------
# Adaptive step size logic
# ------------------------------------------------------------------

function (rule::AdaptiveStepSize)(t::Int, f, x, gradient, vertex, obj, buffer, dir)
    T = eltype(x)
    n = length(x)

    # direction = v - x, cached in dir buffer
    d_norm_sq = zero(T)
    grad_dot_d = zero(T)
    @inbounds @simd for i in 1:n
        di = vertex[i] - x[i]
        dir[i] = di
        d_norm_sq += di * di
        grad_dot_d += gradient[i] * di
    end

    if d_norm_sq < eps(T)
        copyto!(buffer, x)
        return zero(T), obj
    end

    L_max = floatmax(T) / rule.η  # overflow ceiling

    # Backtracking: find L such that sufficient decrease holds
    γ = zero(T)
    obj_trial = obj
    bt_converged = false
    for _ in 1:50
        γ = clamp(-grad_dot_d / (rule.L * d_norm_sq), zero(T), one(T))
        @inbounds @simd for i in 1:n
            buffer[i] = x[i] + γ * dir[i]
        end
        obj_trial = f(buffer)
        if !isfinite(obj_trial)
            @warn "AdaptiveStepSize: non-finite objective in backtracking (L=$(rule.L))" maxlog=3
            rule.L = min(rule.L * rule.η, L_max)
            break
        end
        if obj_trial ≤ obj + γ * grad_dot_d + γ^2 * rule.L * d_norm_sq / 2
            bt_converged = true
            break
        end
        rule.L = min(rule.L * rule.η, L_max)
    end
    if !bt_converged && isfinite(obj_trial)
        @warn "AdaptiveStepSize: backtracking did not converge after 50 iterations (L=$(rule.L))" maxlog=3
    end
    rule.L = max(rule.L / rule.η, eps(T))  # relax for next iteration
    return γ, obj_trial
end
