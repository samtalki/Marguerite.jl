# Copyright 2026 Samuel Talkington and contributors
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
    solve(f, ∇f!, lmo, x0; kwargs...) -> (x, Result)

Solve

```math
\\min_{x \\in C} f(x)
```

via the Frank-Wolfe algorithm with user-supplied gradient `∇f!(g, x)`.

# Arguments
- `f`: objective function `f(x) -> Real`
- `∇f!`: in-place gradient `∇f!(g, x)`, writing ``\\nabla f(x)`` into `g`
- `lmo`: linear minimization oracle (callable `lmo(v, g)` or `<: AbstractOracle`)
- `x0`: initial feasible point (will be copied)

# Keyword Arguments
- `max_iters::Int = 1000`: maximum iterations
- `tol::Real = 1e-7`: convergence tolerance (``\\mathrm{gap} \\le \\mathrm{tol} \\cdot |f(x)|``)
- `step_rule = MonotonicStepSize()`: step size rule (callable `t -> γ`)
- `monotonic::Bool = true`: reject non-improving updates
- `verbose::Bool = false`: print progress
- `cache::Union{Cache, Nothing} = nothing`: pre-allocated buffers

# Returns
`(x, result)` where `x` is the solution and `result::Result` holds diagnostics.
"""
function solve(f::F, ∇f!::Function, lmo::L, x0::AbstractVector;
               max_iters::Int=1000, tol::Real=1e-7,
               step_rule::S=MonotonicStepSize(), monotonic::Bool=true,
               verbose::Bool=false,
               cache::Union{Cache, Nothing}=nothing) where {F, L, S}
    x = copy(x0)
    T = eltype(x)
    n = length(x)
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

        lmo(c.vertex, c.gradient)

        # Frank-Wolfe gap: ⟨∇f, x - v⟩
        fw_gap = zero(T)
        @simd for i in 1:n
            fw_gap += c.gradient[i] * (x[i] - c.vertex[i])
        end

        # Convergence check
        if fw_gap ≤ tol * abs(obj)
            converged = true
            final_iter = t
            break
        end

        γ, obj_cached = _compute_step(step_rule, t, f, x, c.gradient, c.vertex, obj, c.x_trial, c.direction)

        # Trial point: x + γ(v - x)
        # Skip when AdaptiveStepSize already wrote x_trial during backtracking
        if obj_cached === nothing
            @simd for i in 1:n
                c.x_trial[i] = x[i] + γ * (c.vertex[i] - x[i])
            end
        end

        obj_trial = something(obj_cached, f(c.x_trial))

        if !isfinite(obj_trial)
            @warn "solve: non-finite objective ($obj_trial) at iteration $t, discarding step" maxlog=3
            reuse_grad = true
            discards += 1
            continue
        end

        if monotonic && obj_trial > obj + eps(T)
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

    return x, Result(obj, fw_gap, final_iter, converged, discards)
end

# Step size dispatch: simple rules take only t; adaptive rules get full state.
# Returns (γ, obj_trial_or_nothing). AdaptiveStepSize already evaluates f(x_trial)
# during backtracking, so it returns the value to avoid redundant evaluation.
_compute_step(rule, t, f, x, gradient, vertex, obj, buffer, dir) = (eltype(x)(rule(t)), nothing)
function _compute_step(rule::AdaptiveStepSize, t, f, x, gradient, vertex, obj, buffer, dir)
    return rule(t, f, x, gradient, vertex, obj, buffer, dir)
end

"""
    solve(f, lmo, x0; backend=DEFAULT_BACKEND, kwargs...) -> (x, Result)

Auto-gradient variant (no parameters). Computes ``\\nabla f`` via
`DifferentiationInterface.gradient!` using the specified `backend`.
"""
function solve(f::F, lmo::L, x0::AbstractVector;
               backend=DEFAULT_BACKEND,
               cache::Union{Cache, Nothing}=nothing,
               kwargs...) where {F, L}
    T = eltype(x0)
    n = length(x0)
    c = something(cache, Cache{T}(n))
    prep = DI.prepare_gradient(f, backend, x0)
    ∇f!_auto(g, x_) = DI.gradient!(f, g, prep, backend, x_)
    return solve(f, ∇f!_auto, lmo, x0; cache=c, kwargs...)
end

# ------------------------------------------------------------------
# θ-parameterized variants (differentiable)
# ------------------------------------------------------------------

"""
    solve(f, ∇f!, lmo, x0, θ; backend=DEFAULT_BACKEND, kwargs...) -> (x, Result)

Solve

```math
\\min_{x \\in C} f(x, \\theta)
```

with parameters ``\\theta``.

Here `f(x, θ)` and `∇f!(g, x, θ)` accept ``\\theta`` as the second argument.
A `ChainRulesCore.rrule` is defined for this signature, enabling
``\\partial x^* / \\partial \\theta`` via implicit differentiation.

# Differentiation keyword arguments
These are consumed by the rrule backward pass, not the forward solve:
- `backend`: AD backend for first-order gradients (default: `DEFAULT_BACKEND`)
- `hvp_backend`: AD backend for Hessian-vector products (default: `SECOND_ORDER_BACKEND`)
- `diff_cg_maxiter::Int=50`: max CG iterations for the Hessian solve
- `diff_cg_tol::Real=1e-6`: CG convergence tolerance
- `diff_λ::Real=1e-4`: Tikhonov regularization for the Hessian
"""
function solve(f::F, ∇f!::Function, lmo::L, x0::AbstractVector, θ;
               backend=DEFAULT_BACKEND,
               hvp_backend=SECOND_ORDER_BACKEND,
               diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_λ::Real=1e-4,
               kwargs...) where {F, L}
    # backend, hvp_backend, diff_cg_* consumed here to prevent leaking to inner solve;
    # they are used by the rrule (diff_rules.jl) for the backward pass
    fθ(x) = f(x, θ)
    ∇fθ!(g, x) = ∇f!(g, x, θ)
    return solve(fθ, ∇fθ!, lmo, x0; kwargs...)
end

"""
    solve(f, lmo, x0, θ; backend=DEFAULT_BACKEND, kwargs...) -> (x, Result)

Auto-gradient + parameterized variant. Uses `backend` for first-order gradients
and `hvp_backend` for Hessian-vector products in the implicit differentiation.

# Differentiation keyword arguments
- `backend`: AD backend for first-order gradients (default: `DEFAULT_BACKEND`)
- `hvp_backend`: AD backend for Hessian-vector products (default: `SECOND_ORDER_BACKEND`)
- `diff_cg_maxiter::Int=50`: max CG iterations for the Hessian solve
- `diff_cg_tol::Real=1e-6`: CG convergence tolerance
- `diff_λ::Real=1e-4`: Tikhonov regularization for the Hessian
"""
function solve(f::F, lmo::L, x0::AbstractVector, θ;
               backend=DEFAULT_BACKEND,
               hvp_backend=SECOND_ORDER_BACKEND,
               diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_λ::Real=1e-4,
               kwargs...) where {F, L}
    # hvp_backend, diff_cg_* consumed here; used by rrule for the backward pass
    fθ(x) = f(x, θ)
    return solve(fθ, lmo, x0; backend=backend, kwargs...)
end

# ------------------------------------------------------------------
# ParametricOracle variants (differentiable constraint sets)
# ------------------------------------------------------------------

"""
    solve(f, ∇f!, plmo::ParametricOracle, x0, θ; kwargs...) -> (x, Result)

Solve

```math
\\min_{x \\in C(\\theta)} f(x, \\theta)
```

with parameterized constraints.

Materializes `plmo` at ``\\theta``, then delegates to the standard solver.
A `ChainRulesCore.rrule` is defined for this signature, enabling
``\\partial x^* / \\partial \\theta`` via KKT adjoint differentiation through both
the objective and constraint set.

# Differentiation keyword arguments
- `backend`: AD backend for first-order gradients (default: `DEFAULT_BACKEND`)
- `hvp_backend`: AD backend for Hessian-vector products (default: `SECOND_ORDER_BACKEND`)
- `diff_cg_maxiter::Int=50`: max CG iterations for the KKT adjoint solve
- `diff_cg_tol::Real=1e-6`: CG convergence tolerance
- `diff_λ::Real=1e-4`: Tikhonov regularization
"""
function solve(f::F, ∇f!::Function, plmo::ParametricOracle, x0::AbstractVector, θ;
               backend=DEFAULT_BACKEND,
               hvp_backend=SECOND_ORDER_BACKEND,
               diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_λ::Real=1e-4,
               kwargs...) where F
    lmo = materialize(plmo, θ)
    fθ(x) = f(x, θ)
    ∇fθ!(g, x) = ∇f!(g, x, θ)
    return solve(fθ, ∇fθ!, lmo, x0; kwargs...)
end

"""
    solve(f, plmo::ParametricOracle, x0, θ; kwargs...) -> (x, Result)

Auto-gradient + parameterized constraints variant.
"""
function solve(f::F, plmo::ParametricOracle, x0::AbstractVector, θ;
               backend=DEFAULT_BACKEND,
               hvp_backend=SECOND_ORDER_BACKEND,
               diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_λ::Real=1e-4,
               kwargs...) where F
    lmo = materialize(plmo, θ)
    fθ(x) = f(x, θ)
    return solve(fθ, lmo, x0; backend=backend, kwargs...)
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
