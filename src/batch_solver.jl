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
# Per-column evaluation helpers
# ------------------------------------------------------------------

@inline function _eval_objectives!(expr::BatchedExpression, dest::AbstractVector,
                                    X::AbstractMatrix, θ, c::BatchCache)
    B = size(X, 2)
    @inbounds for b in 1:B
        c.active[b] || continue
        dest[b] = expr.f_per_col(view(X, :, b), θ, b)
    end
end

@inline function _eval_gradient!(expr::BatchedExpression, c::BatchCache,
                                  X::AbstractMatrix, θ)
    B = size(X, 2)
    @inbounds for b in 1:B
        c.active[b] || continue
        expr.grad_per_col!(view(c.gradient, :, b), view(X, :, b), θ, b)
    end
end

# ------------------------------------------------------------------
# Adaptive step size (per-problem Lipschitz estimates)
# ------------------------------------------------------------------

"""
    _batch_adaptive_step!(rules, expr, c, X, θ, n, B)

Per-problem adaptive backtracking step. Each problem `b` uses its own
Lipschitz estimate `rules[b].L`. Writes trial points into `c.x_trial[:, b]`
and per-problem `(γ, obj_trial)` into `c.step_gamma` and `c.obj_trial`.
"""
function _batch_adaptive_step!(rules::Vector{<:AdaptiveStepSize},
                                expr::BatchedExpression, c::BatchCache{T},
                                X, θ, n, B) where T
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
            c.step_gamma[b] = zero(T)
            @simd for i in 1:n
                c.x_trial[i, b] = X[i, b]
            end
            c.obj_trial[b] = c.objective[b]
            continue
        end

        γ = zero(T)
        bt_converged = false
        for _ in 1:50
            γ = clamp(-grad_dot_d / (rule.L * d_norm_sq), zero(T), one(T))
            @simd for i in 1:n
                c.x_trial[i, b] = X[i, b] + γ * (V[i, b] - X[i, b])
            end
            ot = expr.f_per_col(view(c.x_trial, :, b), θ, b)
            if !isfinite(ot)
                rule.L = min(rule.L * rule.η, L_max)
                break
            end
            if ot ≤ c.objective[b] + γ * grad_dot_d + γ^2 * rule.L * d_norm_sq / 2
                c.obj_trial[b] = ot
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
            c.obj_trial[b] = c.objective[b]
        end
        rule.L = max(rule.L / rule.η, eps(T))
        c.step_gamma[b] = γ
    end
end

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

"""
    batch_solve(expr::BatchedExpression, lmo, X0; config=BatchSolveConfig(), cache=nothing, kwargs...)
    batch_solve(expr::BatchedExpression, lmo, X0, θ; config=BatchSolveConfig(), cache=nothing, kwargs...)

Solve `B` independent Frank-Wolfe problems in lockstep:

```math
\\min_{x_b \\in C} f_b(x_b) \\quad b = 1, \\ldots, B
```

or, when `θ` is provided,

```math
\\min_{x_b \\in C(\\theta)} f_b(x_b, \\theta) \\quad b = 1, \\ldots, B
```

`X0` is an `(n, B)` matrix whose columns are initial points. CPU and GPU
arrays are accepted; GPU dispatch flows through `KernelAbstractions.get_backend(X0)`.

`expr` is a [`BatchedExpression`](@ref) carrying per-problem callbacks for
the objective and gradient. Each callback is invoked on a single column
view of the iterate, not on the whole `(n, B)` matrix.

`lmo` is an oracle (applied column-wise). The device path covers
`ScalarBox`, `Box`, `ProbSimplex`, `Simplex`. `Knapsack`, `MaskedKnapsack`,
`WeightedSimplex`, `Spectraplex`, and the generic `AbstractOracle` fallback
are CPU only.

If `lmo` is a [`ParametricOracle`](@ref) and `θ` is provided, the
constraint set ``C(\\theta)`` is materialized at ``\\theta``. Parameters
``\\theta`` are shared across all problems.

A `ChainRulesCore.rrule` enables ``\\partial X^*/\\partial\\theta`` via
batched implicit differentiation.

Per-call kwargs override matching fields of `config`.
"""
function batch_solve(expr::BatchedExpression, lmo, X0::AbstractMatrix;
                     config::BatchSolveConfig=BatchSolveConfig(),
                     cache::Union{BatchCache, Nothing}=nothing,
                     kwargs...)
    cfg = _merge_config(config, kwargs)
    is_gpu = !(KernelAbstractions.get_backend(X0) isa KernelAbstractions.CPU)
    if is_gpu && cfg.step_rule isa AdaptiveStepSize
        throw(ArgumentError(
            "AdaptiveStepSize is not supported with GPU arrays in batch_solve. " *
            "Use MonotonicStepSize (default) instead."))
    end
    oracle = _to_oracle(lmo)
    return _batch_solve_core(expr, oracle, X0, nothing, cfg; cache=cache)
end

function batch_solve(expr::BatchedExpression, lmo, X0::AbstractMatrix, θ;
                     config::BatchSolveConfig=BatchSolveConfig(),
                     cache::Union{BatchCache, Nothing}=nothing,
                     kwargs...)
    cfg = _merge_config(config, kwargs)
    is_gpu = !(KernelAbstractions.get_backend(X0) isa KernelAbstractions.CPU)
    if is_gpu && cfg.step_rule isa AdaptiveStepSize
        throw(ArgumentError(
            "AdaptiveStepSize is not supported with GPU arrays in batch_solve. " *
            "Use MonotonicStepSize (default) instead."))
    end
    oracle = _to_oracle(lmo, θ)
    return _batch_solve_core(expr, oracle, X0, θ, cfg; cache=cache)
end

# ------------------------------------------------------------------
# Core loop
# ------------------------------------------------------------------

function _batch_solve_core(expr::BatchedExpression, lmo::AbstractOracle,
                            X0::AbstractMatrix, θ, cfg::BatchSolveConfig;
                            cache::Union{BatchCache, Nothing}=nothing)
    X = copy(X0)
    T = eltype(X)
    n, B = size(X)

    if cache !== nothing
        KernelAbstractions.get_backend(cache.gradient) === KernelAbstractions.get_backend(X0) ||
            throw(ArgumentError(
                "BatchCache and X0 backends disagree (cache.gradient=$(typeof(cache.gradient)), X0=$(typeof(X0))). " *
                "Allocate the cache with BatchCache(X0) to match the array kind."))
        size(cache.gradient) == size(X0) ||
            throw(DimensionMismatch(
                "BatchCache size ($(size(cache.gradient))) ≠ X0 size ($(size(X0))). " *
                "Allocate the cache with BatchCache(X0)."))
    end
    c = something(cache, BatchCache(X0))

    # Reset mutable state
    c.active .= true
    c.discards .= 0
    fill!(c.reuse_grad, false)
    fill!(c.accepted, false)
    _rebuild_active_indices!(c)

    # Initial objectives — per column
    _eval_objectives!(expr, c.objective, X, θ, c)
    fill!(c.gap, T(Inf))

    # Adaptive step size: fan out to per-problem instances
    per_problem_rules = if cfg.step_rule isa AdaptiveStepSize
        [AdaptiveStepSize(cfg.step_rule.L; eta=cfg.step_rule.η) for _ in 1:B]
    else
        nothing
    end

    final_iter = cfg.max_iters

    if cfg.max_iters <= 0
        _eval_gradient!(expr, c, X, θ)
        _batch_lmo_and_gap!(lmo, c, X)
        @inbounds for b in 1:B
            if c.gap[b] <= cfg.tol * (one(T) + abs(c.objective[b]))
                c.active[b] = false
            end
        end
        return BatchSolveResult(X, BatchResult(copy(c.objective), copy(c.gap), 0, .!c.active, copy(c.discards)))
    end

    if cfg.verbose
        @printf("  %6s   %13s   %13s   %6s\n", "Iter", "Max Obj", "Max Gap", "Active")
        println("  ──────   ─────────────   ─────────────   ──────")
    end

    @inbounds for t in 0:(cfg.max_iters - 1)
        if !all(c.reuse_grad)
            _eval_gradient!(expr, c, X, θ)
            fill!(c.reuse_grad, false)
        end

        _batch_lmo_and_gap!(lmo, c, X)

        all_done = true
        any_flipped = false
        for b in 1:B
            if c.active[b] && c.gap[b] <= cfg.tol * (one(T) + abs(c.objective[b]))
                c.active[b] = false
                any_flipped = true
            end
            all_done &= !c.active[b]
        end
        if all_done
            final_iter = t
            break
        end
        if any_flipped && (t % cfg.compaction_interval == 0)
            _rebuild_active_indices!(c)
        end

        if per_problem_rules !== nothing
            _batch_adaptive_step!(per_problem_rules, expr, c, X, θ, n, B)
            _batch_accept!(c, X, c.obj_trial, B, T, t, cfg.monotonic; adaptive=true)
        else
            γ = T(cfg.step_rule(t))
            omγ = one(T) - γ
            @. c.x_trial = omγ * X + γ * c.vertex
            _eval_objectives!(expr, c.obj_trial, c.x_trial, θ, c)
            _batch_accept!(c, X, c.obj_trial, B, T, t, cfg.monotonic; adaptive=false)
        end

        if cfg.verbose && (t % 50 == 0 || t == cfg.max_iters - 1)
            max_obj = maximum(c.objective)
            max_gap = maximum(c.gap)
            n_active = count(c.active)
            @printf("  %6d   %13.6e   %13.4e   %6d\n", t, max_obj, max_gap, n_active)
        end
    end

    converged = .!c.active

    if cfg.verbose
        n_conv = count(converged)
        if n_conv == B
            @printf("  All %d problems converged in %d iterations\n", B, final_iter)
        else
            @printf("  %d/%d problems converged after %d iterations\n", n_conv, B, cfg.max_iters)
        end
    end

    return BatchSolveResult(X, BatchResult(
        copy(c.objective), copy(c.gap), final_iter, converged, copy(c.discards)))
end

# ------------------------------------------------------------------
# Acceptance helpers
# ------------------------------------------------------------------

"""
    _batch_accept!(c, X, obj_trial, B, T, t, monotonic; adaptive)

Accept or reject trial points. When `adaptive=true`, skips problems whose
adaptive step backtracked to zero (`c.step_gamma[b] == 0`).
"""
function _batch_accept!(c::BatchCache, X, obj_trial, B, ::Type{T}, t, monotonic;
                        adaptive::Bool=false) where T
    n = size(X, 1)
    fill!(c.accepted, false)
    @inbounds for b in 1:B
        c.active[b] || continue
        if adaptive && c.step_gamma[b] == zero(T)
            c.reuse_grad[b] = true
            continue
        end
        ot = obj_trial[b]
        if !isfinite(ot)
            @warn "batch_solve: non-finite objective ($ot) for problem $b at iteration $t, discarding step" maxlog=3
            c.discards[b] += 1
            c.reuse_grad[b] = true
            continue
        end
        if monotonic && ot > c.objective[b] + n * eps(T) * max(one(T), abs(c.objective[b]))
            c.discards[b] += 1
            c.reuse_grad[b] = true
            continue
        end
        c.objective[b] = ot
        c.accepted[b] = true
        c.reuse_grad[b] = false
    end
    _batch_update_accepted!(X, c, B)
end

"""
    _batch_update_accepted!(X, c, B)

Copy accepted trial columns into X. CPU loops; GPU uses a single
broadcast against the device-resident accept mask.
"""
function _batch_update_accepted!(X::AbstractMatrix, c::BatchCache, B::Int)
    any(c.accepted) || return
    if KernelAbstractions.get_backend(X) isa KernelAbstractions.CPU
        n = size(X, 1)
        @inbounds for b in 1:B
            c.accepted[b] || continue
            @simd for i in 1:n
                X[i, b] = c.x_trial[i, b]
            end
        end
    else
        copyto!(c.mask_dev, c.accepted)
        X .= ifelse.(reshape(c.mask_dev, 1, B), c.x_trial, X)
    end
end
