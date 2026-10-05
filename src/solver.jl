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
    _solve_core(f, ∇f!, lmo, x0; kwargs...) -> SolveResult

Core Frank-Wolfe loop. Requires `lmo <: AbstractOracle` for dispatch on
[`_lmo_and_gap!`](@ref) specializations.

The gradient, the Frank-Wolfe vertex and the gap are evaluated at the start
point and again after every accepted step, so the returned `Result.gap` always
belongs to the returned iterate. A rejected step leaves the iterate, its
gradient and its vertex unchanged, so nothing is recomputed.

Callers should use [`solve`](@ref) instead; this is an internal function.
"""
function _solve_core(f::F, ∇f!::G, lmo::L, x0::AbstractVector;
               max_iters::Int=10000, tol::Real=1e-4, rel_tol::Real=0,
               time_limit::Real=Inf,
               step_rule::S=MonotonicStepSize(), monotonic::Bool=true,
               verbose::Bool=false, callback::CB=nothing,
               cache::Union{Cache, Nothing}=nothing) where {F, G, L<:AbstractOracle, S, CB}
    t_start = time_ns()
    x = copy(x0)
    T = eltype(x)
    n = length(x)
    if cache !== nothing && length(cache.gradient) != n
        throw(DimensionMismatch(
            "Cache dimension ($(length(cache.gradient))) ≠ x0 dimension ($n)"))
    end
    c = cache === nothing ? Cache(x0) : cache   # not `something`, which would allocate a Cache eagerly

    obj = f(x)
    ∇f!(c.gradient, x)
    fw_gap, nnz = _lmo_and_gap!(lmo, c, x, n)
    lower_bound = _lower_bound_update(T(-Inf), obj, fw_gap)
    converged = _gap_converged(fw_gap, obj, tol, rel_tol)
    discards = 0
    iters = 0
    stopped_by_callback = false
    elapsed = _seconds_since(t_start)

    if verbose
        @printf("  %6s   %13s   %13s\n", "Iter", "Objective", "FW Gap")
        println("  ──────   ─────────────   ─────────────")
    end

    @inbounds while !converged && iters < max_iters && elapsed < time_limit
        t = iters

        # Step rules other than MonotonicStepSize need the dense vertex buffer
        _ensure_vertex!(c, nnz, step_rule)

        γ, obj_cached, grad_ready = _compute_step(step_rule, t, f, ∇f!, x, c, obj)

        # Skip when the step rule already wrote x_trial (e.g. during backtracking)
        if obj_cached === nothing
            _trial_update!(c, x, γ, nnz, n)
        end

        # Evaluate f only when the step rule did not (a `something(obj_cached,
        # f(x_trial))` call would evaluate f eagerly every time).
        obj_trial = obj_cached === nothing ? f(c.x_trial) : obj_cached

        accepted = false
        if !isfinite(obj_trial)
            @warn "solve: non-finite objective ($obj_trial) at iteration $t, discarding step" maxlog=3
            discards += 1
        # Trial update is O(n) flops; rounding in f(x_trial)-f(x) is O(n·ε·|f|)
        elseif monotonic && obj_trial > obj + n * eps(T) * max(one(T), abs(obj))
            discards += 1
        else
            copyto!(x, c.x_trial)
            obj = obj_trial
            accepted = true
            if grad_ready  # the step rule already evaluated ∇f at this point
                copyto!(c.gradient, c.gradient_trial)
            else
                ∇f!(c.gradient, x)
            end
            fw_gap, nnz = _lmo_and_gap!(lmo, c, x, n)
            lower_bound = _lower_bound_update(lower_bound, obj, fw_gap)
            # Guard against NaN/-Inf gaps so a corrupted gradient or trial
            # point cannot trigger spurious convergence (see _gap_converged).
            converged = _gap_converged(fw_gap, obj, tol, rel_tol)
        end
        iters += 1
        elapsed = _seconds_since(t_start)

        if verbose && accepted && (t % 50 == 0 || t == max_iters - 1)
            @printf("  %6d   %13.6e   %13.4e\n", t, obj, fw_gap)
        end

        if callback !== nothing
            state = (; t=iters, x=x, obj=obj, gap=fw_gap, lower_bound=lower_bound,
                       γ=γ, accepted=accepted, elapsed=elapsed)
            if callback(state) === true
                stopped_by_callback = true
                break
            end
        end
    end

    elapsed = _seconds_since(t_start)
    if verbose
        if converged
            @printf("  Converged in %d iterations (gap=%.4e ≤ tol)\n", iters, fw_gap)
        elseif stopped_by_callback
            @printf("  Stopped by callback after %d iterations (gap=%.4e)\n", iters, fw_gap)
        elseif iters < max_iters
            @printf("  Time limit reached after %d iterations (gap=%.4e)\n", iters, fw_gap)
        else
            @printf("  Did not converge after %d iterations (gap=%.4e)\n", iters, fw_gap)
        end
    end

    return SolveResult(x, Result(obj, fw_gap, iters, converged, discards, lower_bound, elapsed))
end

# Stopping test on the Frank-Wolfe gap: absolute-plus-relative `tol` or purely
# relative `rel_tol`. Non-finite gaps never count as converged.
@inline function _gap_converged(gap, obj, tol, rel_tol)
    isfinite(gap) || return false
    return gap ≤ tol * (one(gap) + abs(obj)) || gap ≤ rel_tol * abs(obj)
end

@inline _seconds_since(t_start::UInt64) = (time_ns() - t_start) / 1e9

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
_trial_update!(c::Cache, x, γ, nnz::Int, n::Int) = _trial_update!(KernelAbstractions.get_backend(x), c, x, γ, nnz, n)

function _trial_update!(::KernelAbstractions.CPU, c::Cache{T}, x, γ, nnz::Int, n::Int) where T
    if nnz < 0  # dense vertex
        omγ_d = one(T) - γ
        @inbounds @simd for i in 1:n
            c.x_trial[i] = omγ_d * x[i] + γ * c.vertex[i]
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

function _trial_update!(::KernelAbstractions.Backend, c::Cache{T}, x, γ, ::Int, ::Int) where T
    omγ = one(T) - γ
    c.x_trial .= omγ .* x .+ γ .* c.vertex
end

# Step size dispatch: simple rules take only t; the others get the full state.
# Returns (γ, obj_trial_or_nothing, grad_ready).
# Contract: if obj_trial_or_nothing !== nothing, c.x_trial MUST contain the
# corresponding trial point x + γ*(vertex - x), and c.direction MUST contain
# vertex - x. If grad_ready, c.gradient_trial MUST contain ∇f(c.x_trial).
_compute_step(rule, t, f, ∇f!, x, c::Cache, obj) = (eltype(x)(rule(t)), nothing, false)
function _compute_step(rule::AdaptiveStepSize, t, f, ∇f!, x, c::Cache, obj)
    γ, obj_trial = rule(t, f, x, c.gradient, c.vertex, obj, c.x_trial, c.direction)
    return γ, obj_trial, false
end
function _compute_step(rule::ShortStep, t, f, ∇f!, x, c::Cache, obj)
    T = eltype(x)
    d_norm_sq, grad_dot_d = _fw_direction!(c.direction, c.vertex, x, c.gradient)
    return _short_step(rule.L, d_norm_sq, grad_dot_d, one(T)), nothing, false
end
function _compute_step(rule::SecantLineSearch, t, f, ∇f!, x, c::Cache, obj)
    T = eltype(x)
    d_norm_sq, grad_dot_d = _fw_direction!(c.direction, c.vertex, x, c.gradient)
    return _secant_search!(rule, t, f, ∇f!, x, c, obj, d_norm_sq, grad_dot_d, one(T))
end

# x_trial = x + γ d
@inline function _step_along!(buffer, x, γ, dir)
    @inbounds @simd for i in eachindex(buffer, x, dir)
        buffer[i] = x[i] + γ * dir[i]
    end
    return buffer
end

"""
    _secant_search!(rule::SecantLineSearch, t, f, ∇f!, x, c, obj, d_norm_sq, grad_dot_d, γ_max)
        -> (γ, obj_trial_or_nothing, grad_ready)

Secant line search along `d = c.direction` over `[0, γ_max]` (see
[`SecantLineSearch`](@ref)). Trial points go to `c.x_trial`; the gradient at
trial 1 goes to `c.gradient_trial`. Returns `grad_ready = true` only when trial 1
is accepted, so the caller can reuse that gradient. Returns `obj_trial = nothing`
for the open-loop fallback, which the caller evaluates and checks.
"""
function _secant_search!(rule::SecantLineSearch, t, f, ∇f!, x, c::Cache, obj,
                         d_norm_sq, grad_dot_d, γ_max)
    T = eltype(x)
    t == 0 && (rule.γ_prev = T(rule.γ0))
    slope0 = -grad_dot_d                     # -φ'(0) > 0 along a descent direction
    if !(d_norm_sq > zero(T)) || !(slope0 > zero(T)) || !(γ_max > zero(T))
        copyto!(c.x_trial, x)
        return zero(T), obj, false
    end
    dir = c.direction

    # Trial 1: double the previous step, with value and slope
    γ = min(T(2) * T(rule.γ_prev), T(γ_max))
    _step_along!(c.x_trial, x, γ, dir)
    obj_trial = f(c.x_trial)
    if isfinite(obj_trial)
        ∇f!(c.gradient_trial, c.x_trial)
        slope = dot(c.gradient_trial, dir)   # φ'(γ)
        if slope ≤ zero(T)
            rule.γ_prev = γ
            return γ, obj_trial, true
        end
        # Trial 2: zero of the secant of φ' through (0, -slope0) and (γ, slope)
        γ = isfinite(slope) ? γ * (slope0 / (slope0 + slope)) : γ / 2
    else
        γ = γ / 2
    end

    # Trials 2..max_trials: the secant point, then halvings while f rises
    for _ in 2:rule.max_trials
        _step_along!(c.x_trial, x, γ, dir)
        obj_trial = f(c.x_trial)
        if isfinite(obj_trial) && obj_trial ≤ obj
            rule.γ_prev = γ
            return γ, obj_trial, false
        end
        γ = γ / 2
    end

    # Fallback: open-loop step, checked by the solver's monotone test
    γ = min(T(2) / T(t + 2), T(γ_max))
    rule.γ_prev = γ
    return γ, nothing, false
end

"""
    _fw_direction!(dir, vertex, x, gradient) -> (‖d‖², ⟨∇f, d⟩)

Write the Frank-Wolfe direction `d = vertex - x` into `dir` and return its
squared norm and its inner product with the gradient, in one pass.
"""
@inline function _fw_direction!(dir, vertex, x, gradient)
    T = eltype(x)
    d_norm_sq = zero(T)
    grad_dot_d = zero(T)
    @inbounds @simd for i in eachindex(dir, vertex, x, gradient)
        di = vertex[i] - x[i]
        dir[i] = di
        d_norm_sq += di * di
        grad_dot_d += gradient[i] * di
    end
    return d_norm_sq, grad_dot_d
end

# Step rules whose direction computations use scalar indexing on the CPU.
_cpu_only_step_rule(rule) = false
_cpu_only_step_rule(::AdaptiveStepSize) = true
_cpu_only_step_rule(::ShortStep) = true
_cpu_only_step_rule(::SecantLineSearch) = true

# Minimizer over [0, γ_max] of the quadratic model γ⟨∇f, d⟩ + (L/2)γ²‖d‖².
@inline function _short_step(L, d_norm_sq::T, grad_dot_d::T, γ_max::T) where {T<:Real}
    d_norm_sq > zero(T) || return zero(T)
    return clamp(-grad_dot_d / (T(L) * d_norm_sq), zero(T), γ_max)
end

"""
    _to_oracle(lmo) -> AbstractOracle
    _to_oracle(lmo, θ) -> AbstractOracle

Normalize `lmo` to an `AbstractOracle`. If `lmo` is a `ParametricOracle` and
`θ` is provided, materializes the constraint set at `θ`. Calling with a
`ParametricOracle` and no `θ` is an error: such oracles cannot be reduced
without their parameters.
"""
_to_oracle(lmo::AbstractOracle) = lmo
_to_oracle(lmo::AbstractOracle, _) = lmo
_to_oracle(lmo::ParametricOracle, θ) = materialize(lmo, θ)
_to_oracle(::ParametricOracle) = throw(ArgumentError(
    "ParametricOracle requires θ; pass it as the 4th argument to solve / batch_solve / bilevel_solve."))
_to_oracle(lmo) = FunctionOracle(lmo)
_to_oracle(lmo, _) = FunctionOracle(lmo)

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
- `max_iters::Int = 10000`: maximum iterations
- `tol::Real = 1e-4`: convergence tolerance (``\\mathrm{gap} \\le \\mathrm{tol} \\cdot (1 + |f(x)|)``)
- `rel_tol::Real = 0`: relative tolerance; the solve also stops when
  ``\\mathrm{gap} \\le \\mathrm{rel\\_tol} \\cdot |f(x)|``
- `time_limit::Real = Inf`: wall-clock limit in seconds, checked after every iteration
- `callback = nothing`: function `callback(state) -> Bool` called once after every
  iteration; returning `true` stops the solve (see below)
- `step_rule = MonotonicStepSize()`: step size rule (callable `t -> γ`)
- `monotonic::Bool = true`: reject non-improving updates
- `verbose::Bool = false`: print progress
- `cache::Union{Cache, Nothing} = nothing`: pre-allocated buffers

# Callback

`state` is a `NamedTuple` with fields
- `t`: iterations completed so far (`1` on the first call); after a stop it equals `Result.iterations`
- `x`: the current iterate ``x_t``. This is the solver's working buffer: copy it to keep it.
- `obj`, `gap`: objective and Frank-Wolfe gap at ``x_t``
- `lower_bound`: running maximum of `obj - gap` (see [`Result`](@ref))
- `γ`: the step size tried in this iteration
- `accepted`: whether the step was accepted (`false` for a rejected or non-finite trial)
- `elapsed`: seconds since the solve started

Only a return value of `true` stops the solve, so a callback that returns `nothing`
just observes. The start point is not reported; `solve(...; max_iters=0)` returns
its objective and gap.

The stopping tests run after every iteration: the solve stops as soon as the gap
test passes (`converged = true`), `elapsed ≥ time_limit`, the callback returns
`true`, or `max_iters` iterations are done. In every case the returned
`Result.gap` is the gap at the returned `x`.
"""
@inline function solve(f, lmo, x0::AbstractVector;
                       grad=nothing,
                       backend=DEFAULT_BACKEND,
                       cache::Union{Cache, Nothing}=nothing,
                       max_iters::Int=10000, tol::Real=1e-4,
                       rel_tol::Real=0, time_limit::Real=Inf,
                       callback=nothing,
                       step_rule=MonotonicStepSize(), monotonic::Bool=true,
                       verbose::Bool=false)
    oracle = _to_oracle(lmo)
    c = if cache !== nothing
        cache
    else
        Cache(x0)
    end
    if !(KernelAbstractions.get_backend(x0) isa KernelAbstractions.CPU) && _cpu_only_step_rule(step_rule)
        throw(ArgumentError(
            "$(nameof(typeof(step_rule))) is not supported with GPU arrays. " *
            "Use MonotonicStepSize (default) instead."))
    end
    if grad === nothing
        if !(KernelAbstractions.get_backend(x0) isa KernelAbstractions.CPU)
            throw(ArgumentError(
                "Auto-gradient (ForwardDiff) is not supported with GPU arrays. " *
                "Provide a manual gradient via grad=your_gradient_function."))
        end
        prep = DI.prepare_gradient(f, backend, x0)
        ∇f!(g, x_) = DI.gradient!(f, g, prep, backend, x_)
        return _solve_core(f, ∇f!, oracle, x0; cache=c, max_iters=max_iters,
                           tol=tol, rel_tol=rel_tol, time_limit=time_limit, callback=callback,
                           step_rule=step_rule, monotonic=monotonic, verbose=verbose)
    else
        return _solve_core(f, grad, oracle, x0; cache=c, max_iters=max_iters,
                           tol=tol, rel_tol=rel_tol, time_limit=time_limit, callback=callback,
                           step_rule=step_rule, monotonic=monotonic, verbose=verbose)
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
- `assume_interior::Bool=false`: for differentiated calls with custom oracles
  lacking [`active_set`](@ref), error by default; when `true`, use the interior
  active set approximation instead

The remaining keywords (`cache`, `max_iters`, `tol`, `rel_tol`, `time_limit`,
`callback`, `step_rule`, `monotonic`, `verbose`) are those of the
three-argument `solve`.
"""
@inline function solve(f, lmo, x0::AbstractVector, θ;
                       grad=nothing,
                       backend=DEFAULT_BACKEND,
                       hvp_backend=SECOND_ORDER_BACKEND,
                       diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_lambda::Real=1e-4,
                       assume_interior::Bool=false,
                       cache::Union{Cache, Nothing}=nothing,
                       max_iters::Int=10000, tol::Real=1e-4,
                       rel_tol::Real=0, time_limit::Real=Inf,
                       callback=nothing,
                       step_rule=MonotonicStepSize(), monotonic::Bool=true,
                       verbose::Bool=false)
    oracle = _to_oracle(lmo, θ)
    fθ(x) = f(x, θ)
    kw = (; cache, max_iters, tol, rel_tol, time_limit, callback, step_rule, monotonic, verbose)
    if grad === nothing
        return solve(fθ, oracle, x0; backend=backend, kw...)
    else
        ∇fθ!(g, x) = grad(g, x, θ)
        return solve(fθ, oracle, x0; grad=∇fθ!, kw...)
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
        copyto!(buffer, x)
        γ = zero(T)
        obj_trial = obj
    end
    rule.L = max(rule.L / rule.η, eps(T))  # relax for next iteration
    return γ, obj_trial
end
