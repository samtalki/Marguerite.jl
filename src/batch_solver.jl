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

# ------------------------------------------------------------------
# Auto-gradient helpers
# ------------------------------------------------------------------

"""
    _col_to_batch(x::AbstractVector, b::Int, n::Int, B::Int)

Build an `(n, B)` matrix with `x` in column `b` and zeros elsewhere.
Used internally for per-column DI gradient preparation.
"""
function _col_to_batch(x::AbstractVector{T}, b::Int, n::Int, B::Int) where T
    X = zeros(T, n, B)
    X[:, b] .= x
    return X
end

"""
    _make_batch_col_fn(f_batch, b, n, B)

Build a per-column scalar objective closure `x -> f_batch(...)[b]`.
The closure type is stable across prep and execution calls.
"""
function _make_batch_col_fn(f_batch, b::Int, n::Int, B::Int)
    return x -> f_batch(_col_to_batch(x, b, n, B))[b]
end

"""
    _make_batch_col_grad(grad_batch, b, n, B)

Build a per-column scalar gradient closure `(g, x, θ) -> ...` from a
batched gradient function. Handles ForwardDiff Dual type promotion.
"""
function _make_batch_col_grad(grad_batch, b::Int, n::Int, B::Int)
    return (g, x, θ_) -> begin
        Tg = promote_type(eltype(x), eltype(θ_))
        G_buf = zeros(Tg, n, B)
        grad_batch(G_buf, _col_to_batch(x, b, n, B), θ_)
        copyto!(g, @view(G_buf[:, b]))
    end
end

"""
    _make_auto_grad_batch(f_batch, col_fns, preps, backend, n, B)

Build an in-place batched gradient function from per-column `DI.PreparedGradient`
handles. Each column gradient is computed independently via AD.
"""
function _make_auto_grad_batch(f_batch, col_fns, preps, backend, n::Int, B::Int)
    return function ∇_batch!(G::AbstractMatrix{T}, X::AbstractMatrix{T}) where T
        @inbounds for b in 1:B
            x_b = Vector{T}(X[:, b])
            g_b = zeros(T, n)
            DI.gradient!(col_fns[b], g_b, preps[b], backend, x_b)
            G[:, b] .= g_b
        end
    end
end

# ------------------------------------------------------------------
# Adaptive step size helpers
# ------------------------------------------------------------------

"""
    _batch_adaptive_step!(rules, f_batch, c, X, t, n, B)

Per-problem adaptive backtracking step. Each problem `b` uses its own
Lipschitz estimate `rules[b].L`. Writes trial points into `c.x_trial`
and returns per-problem `(γ, obj_trial)` in `c.step_gamma` and `c.obj_trial`.
"""
function _batch_adaptive_step!(rules::Vector{<:AdaptiveStepSize},
                               f_batch, c::BatchCache{T}, X, t, n, B,
                               step_gamma::Vector{T}, obj_trial_vec::Vector{T}) where T
    G = c.gradient
    V = c.vertex

    @inbounds for b in 1:B
        c.active[b] || continue
        rule = rules[b]
        L_max = floatmax(T) / rule.η

        d_norm_sq = zero(T)
        grad_dot_d = zero(T)
        @simd for i in 1:n
            di = V[i, b] - X[i, b]
            d_norm_sq += di * di
            grad_dot_d += G[i, b] * di
        end

        if d_norm_sq < eps(T)
            step_gamma[b] = zero(T)
            @simd for i in 1:n
                c.x_trial[i, b] = X[i, b]
            end
            obj_trial_vec[b] = c.objective[b]
            continue
        end

        γ = zero(T)
        bt_converged = false
        for _ in 1:50
            γ = clamp(-grad_dot_d / (rule.L * d_norm_sq), zero(T), one(T))
            @simd for i in 1:n
                c.x_trial[i, b] = X[i, b] + γ * (V[i, b] - X[i, b])
            end
            ot = f_batch(c.x_trial)[b]
            if !isfinite(ot)
                rule.L = min(rule.L * rule.η, L_max)
                break
            end
            if ot ≤ c.objective[b] + γ * grad_dot_d + γ^2 * rule.L * d_norm_sq / 2
                obj_trial_vec[b] = ot
                bt_converged = true
                break
            end
            rule.L = min(rule.L * rule.η, L_max)
        end
        if !bt_converged
            γ = zero(T)
            @simd for i in 1:n
                c.x_trial[i, b] = X[i, b]
            end
            obj_trial_vec[b] = c.objective[b]
        end
        rule.L = max(rule.L / rule.η, eps(T))
        step_gamma[b] = γ
    end
end

# ------------------------------------------------------------------
# Public API: batch_solve (no θ)
# ------------------------------------------------------------------

"""
    batch_solve(f_batch, lmo, X0; kwargs...) -> (X, BatchResult)

Solve `B` independent Frank-Wolfe problems in lockstep:

```math
\\min_{x_b \\in C} f_b(x_b) \\quad b = 1, \\ldots, B
```

`X0` is an `(n, B)` matrix whose columns are initial points. Supports
CPU and GPU arrays (pass `CuMatrix`, `MtlMatrix`, etc.).

# Arguments
- `f_batch(X::AbstractMatrix) -> AbstractVector`: returns per-problem objectives (length `B`)
- `lmo`: oracle (applied column-wise). GPU-supported: `ScalarBox`, `Box`, `ProbSimplex`, `Simplex`.
- `X0`: `(n, B)` initial point matrix

# Keyword Arguments
- `grad_batch`: in-place batched gradient `grad_batch(G, X)` writing `(n, B)` gradients.
  If `nothing` (default), computed automatically via `DifferentiationInterface`.
  Auto-gradient is not supported with GPU arrays.
- `backend`: AD backend for auto-gradient (default: `DEFAULT_BACKEND`)
- `step_rule`: step size rule. `MonotonicStepSize()` (default) or `AdaptiveStepSize()`.
  `AdaptiveStepSize` creates independent per-problem Lipschitz estimates.
  Not supported with GPU arrays.
- `max_iters::Int = 10000`: maximum lockstep iterations
- `tol::Real = 1e-4`: convergence tolerance per problem
- `monotonic::Bool = true`: reject non-improving updates
- `verbose::Bool = false`: print progress
- `cache::Union{BatchCache, Nothing} = nothing`: pre-allocated buffers
"""
function batch_solve(f_batch, lmo, X0::AbstractMatrix;
                     grad_batch=nothing,
                     backend=DEFAULT_BACKEND,
                     max_iters::Int=10000, tol::Real=1e-4,
                     step_rule=MonotonicStepSize(),
                     monotonic::Bool=true,
                     verbose::Bool=false,
                     cache::Union{BatchCache, Nothing}=nothing)
    is_gpu = _array_style(X0) isa _GPUStyle
    if is_gpu && grad_batch === nothing
        throw(ArgumentError(
            "Auto-gradient is not supported with GPU arrays in batch_solve. " *
            "Provide grad_batch=(G, X) -> ... that writes (n, B) gradients into G in-place."))
    end
    if is_gpu && step_rule isa AdaptiveStepSize
        throw(ArgumentError(
            "AdaptiveStepSize is not supported with GPU arrays in batch_solve. " *
            "Use MonotonicStepSize (default) instead."))
    end
    oracle = _to_oracle(lmo)

    n, B = size(X0)
    if grad_batch === nothing
        T_x = eltype(X0)
        x_rep = Vector{T_x}(X0[:, 1])
        col_fns = [_make_batch_col_fn(f_batch, b, n, B) for b in 1:B]
        preps = [DI.prepare_gradient(col_fns[b], backend, x_rep) for b in 1:B]
        ∇_batch! = _make_auto_grad_batch(f_batch, col_fns, preps, backend, n, B)
    else
        ∇_batch! = grad_batch
    end

    return _batch_solve_core(f_batch, ∇_batch!, oracle, X0;
                              max_iters=max_iters, tol=tol, step_rule=step_rule,
                              monotonic=monotonic, verbose=verbose, cache=cache)
end

# ------------------------------------------------------------------
# Public API: batch_solve (with θ)
# ------------------------------------------------------------------

"""
    batch_solve(f_batch, lmo, X0, θ; kwargs...) -> (X, BatchResult)

Solve `B` independent parametric Frank-Wolfe problems:

```math
\\min_{x_b \\in C(\\theta)} f_b(x_b, \\theta) \\quad b = 1, \\ldots, B
```

If `lmo` is a [`ParametricOracle`](@ref), the constraint set ``C(\\theta)``
is materialized at ``\\theta``. Parameters ``\\theta`` are shared across all
problems.

A `ChainRulesCore.rrule` enables ``\\partial X^*/\\partial\\theta`` via
batched implicit differentiation.

# Keyword Arguments
- `grad_batch`: in-place gradient `grad_batch(G, X, θ)`. If `nothing`, auto-computed.
- All other keywords forwarded to the non-parametric `batch_solve`.

See also [`batch_bilevel_solve`](@ref).
"""
function batch_solve(f_batch, lmo, X0::AbstractMatrix, θ;
                     grad_batch=nothing,
                     backend=DEFAULT_BACKEND,
                     hvp_backend=SECOND_ORDER_BACKEND,
                     diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_lambda::Real=1e-4,
                     assume_interior::Bool=false,
                     kwargs...)
    oracle = _to_oracle(lmo, θ)
    fθ(X) = f_batch(X, θ)
    if grad_batch === nothing
        return batch_solve(fθ, oracle, X0; backend=backend, kwargs...)
    else
        ∇fθ!(G, X) = grad_batch(G, X, θ)
        return batch_solve(fθ, oracle, X0; grad_batch=∇fθ!, backend=backend, kwargs...)
    end
end

# ------------------------------------------------------------------
# Core loop
# ------------------------------------------------------------------

function _batch_solve_core(f_batch::F, ∇f_batch!::G, lmo::L, X0::AbstractMatrix;
                           max_iters::Int=10000, tol::Real=1e-4,
                           step_rule=MonotonicStepSize(),
                           monotonic::Bool=true, verbose::Bool=false,
                           cache::Union{BatchCache, Nothing}=nothing) where {F, G, L<:AbstractOracle}
    X = copy(X0)
    T = eltype(X)
    n, B = size(X)

    c = something(cache, BatchCache(X0))

    # Reset mutable state
    c.active .= true
    c.discards .= 0

    # Initial objectives
    c.objective .= f_batch(X)
    fill!(c.gap, T(Inf))

    # Adaptive step size: fan out to per-problem instances
    per_problem_rules = if step_rule isa AdaptiveStepSize
        [AdaptiveStepSize(step_rule.L; eta=step_rule.η) for _ in 1:B]
    else
        nothing
    end

    # Adaptive step size workspace
    step_gamma = per_problem_rules !== nothing ? zeros(T, B) : nothing
    obj_trial_vec = per_problem_rules !== nothing ? zeros(T, B) : nothing

    final_iter = max_iters
    reuse_grad = falses(B)
    accepted = falses(B)

    if max_iters <= 0
        ∇f_batch!(c.gradient, X)
        _batch_lmo_and_gap!(lmo, c, X)
        @inbounds for b in 1:B
            if c.gap[b] <= tol * (one(T) + abs(c.objective[b]))
                c.active[b] = false
            end
        end
        return BatchSolveResult(X, BatchResult(copy(c.objective), copy(c.gap), 0, .!c.active, copy(c.discards)))
    end

    if verbose
        @printf("  %6s   %13s   %13s   %6s\n", "Iter", "Max Obj", "Max Gap", "Active")
        println("  ──────   ─────────────   ─────────────   ──────")
    end

    @inbounds for t in 0:(max_iters - 1)
        # 1. Batched gradient (skip entirely if all problems discarded last step)
        if !all(reuse_grad)
            ∇f_batch!(c.gradient, X)
            fill!(reuse_grad, false)
        end

        # 2. Batched LMO + gap
        _batch_lmo_and_gap!(lmo, c, X)

        # 3. Convergence check
        all_done = true
        for b in 1:B
            if c.active[b] && c.gap[b] <= tol * (one(T) + abs(c.objective[b]))
                c.active[b] = false
            end
            all_done &= !c.active[b]
        end
        if all_done
            final_iter = t
            break
        end

        # 4. Step size + trial update
        if per_problem_rules !== nothing
            # Per-problem adaptive backtracking (CPU only)
            _batch_adaptive_step!(per_problem_rules, f_batch, c, X, t, n, B,
                                  step_gamma, obj_trial_vec)
        else
            # MonotonicStepSize: shared scalar step
            γ = T(2) / T(t + 2)
            omγ = one(T) - γ
            @. c.x_trial = omγ * X + γ * c.vertex
        end

        # 5. Objective evaluation + acceptance
        if per_problem_rules !== nothing
            _batch_accept!(c, X, obj_trial_vec, n, B, T, t, monotonic, reuse_grad, accepted; step_gamma=step_gamma)
        else
            obj_trial = f_batch(c.x_trial)
            _batch_accept!(c, X, obj_trial, n, B, T, t, monotonic, reuse_grad, accepted)
        end

        if verbose && (t % 50 == 0 || t == max_iters - 1)
            max_obj = maximum(c.objective)
            max_gap = maximum(c.gap)
            n_active = count(c.active)
            @printf("  %6d   %13.6e   %13.4e   %6d\n", t, max_obj, max_gap, n_active)
        end
    end

    converged = .!c.active

    if verbose
        n_conv = count(converged)
        if n_conv == B
            @printf("  All %d problems converged in %d iterations\n", B, final_iter)
        else
            @printf("  %d/%d problems converged after %d iterations\n", n_conv, B, max_iters)
        end
    end

    return BatchSolveResult(X, BatchResult(
        copy(c.objective), copy(c.gap), final_iter, converged, copy(c.discards)))
end

# ------------------------------------------------------------------
# Acceptance helpers
# ------------------------------------------------------------------

"""
    _batch_accept!(c, X, obj_trial, n, B, T, t, monotonic, reuse_grad, accepted; step_gamma=nothing)

Accept or reject trial points. GPU-compatible via masked broadcast.
When `step_gamma` is provided, skips problems with zero step (adaptive path).
"""
function _batch_accept!(c::BatchCache, X, obj_trial, n, B, ::Type{T}, t, monotonic,
                        reuse_grad, accepted; step_gamma=nothing) where T
    fill!(accepted, false)
    @inbounds for b in 1:B
        c.active[b] || continue
        if step_gamma !== nothing && step_gamma[b] == zero(T)
            reuse_grad[b] = true
            continue
        end
        ot = obj_trial[b]
        if !isfinite(ot)
            @warn "batch_solve: non-finite objective ($ot) for problem $b at iteration $t, discarding step" maxlog=3
            c.discards[b] += 1
            reuse_grad[b] = true
            continue
        end
        if monotonic && ot > c.objective[b] + n * eps(T) * max(one(T), abs(c.objective[b]))
            c.discards[b] += 1
            reuse_grad[b] = true
            continue
        end
        c.objective[b] = ot
        accepted[b] = true
        reuse_grad[b] = false
    end
    _batch_update_accepted!(X, c.x_trial, accepted, B)
end

"""
    _batch_update_accepted!(X, x_trial, accepted, B)

Copy accepted trial columns into X. Uses broadcast for GPU compatibility.
"""
function _batch_update_accepted!(X::AbstractMatrix, x_trial::AbstractMatrix, accepted::BitVector, B::Int)
    any(accepted) || return
    if _array_style(X) isa _GPUStyle
        mask = reshape(accepted, 1, B)
        X .= ifelse.(mask, x_trial, X)
    else
        n = size(X, 1)
        @inbounds for b in 1:B
            accepted[b] || continue
            @simd for i in 1:n
                X[i, b] = x_trial[i, b]
            end
        end
    end
end

