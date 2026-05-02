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

# ── Cotangent unwrap ─────────────────────────────────────────────────

_unwrap_dy(dy::Tuple) = dy[1]
_unwrap_dy(dy::BatchSolveResult) = dy.X
_unwrap_dy(dy::ChainRulesCore.Tangent) = dy.X
_unwrap_dy(dy) = dy

# ── Constraint sensitivity dispatch ──────────────────────────────────

_face_multipliers(::AbstractOracle, _fθ, _grad, _x, _θ, _as, _backend) = nothing
function _face_multipliers(lmo::ParametricOracle, fθ_b, grad_b, x_b, θ, as, backend)
    _primal_face_multipliers(fθ_b, grad_b, x_b, θ, as, backend)
end

_add_constraint_dθ!(dθ_accum, ::AbstractOracle, _θ, _x, _u, _μb, _μe, _λ, _as, _backend) = dθ_accum
function _add_constraint_dθ!(dθ_accum, lmo::ParametricOracle, θ, x_b, u_b, μ_bound, μ_eq, λ_data_b, as, backend)
    λ_bound, λ_eq = λ_data_b
    dθ_con = _constraint_pullback(lmo, θ, x_b, u_b, μ_bound, μ_eq, λ_bound, λ_eq, as, backend)
    dθ_accum .+= dθ_con
end

# ── Per-column closures ──────────────────────────────────────────────

_make_fθ_per_col(expr::BatchedExpression, b::Int) =
    let b_ = b; (x, θ_) -> expr.f_per_col(x, θ_, b_) end

_make_grad_per_col(expr::BatchedExpression, b::Int) =
    expr.grad_per_col! === nothing ? nothing :
        let b_ = b; (g, x, θ_) -> expr.grad_per_col!(g, x, θ_, b_) end

# ── Cross derivative for a single column ─────────────────────────────

# Three paths, in priority order:
#   1. user-supplied analytic `cross_hvp` -- O(1) call into user code.
#   2. manual gradient -- AD over θ → ⟨∇ₓf(x*, θ), u⟩.
#   3. joint HVP on (x, θ) along (u, 0), take the θ block.
function _cross_dθ_per_col(expr::BatchedExpression, x_star_b, θ, u_b, b::Int, backend, hvp_backend)
    if expr.cross_hvp !== nothing
        out = similar(θ, promote_type(eltype(θ), eltype(u_b)))
        fill!(out, zero(eltype(out)))
        expr.cross_hvp(out, x_star_b, θ, u_b, b)
        return out
    end
    if expr.grad_per_col! !== nothing
        grad_b! = _make_grad_per_col(expr, b)
        ∇ₓf_of_θ = _make_∇ₓf_of_θ(grad_b!, x_star_b)
        return _cross_derivative_manual(∇ₓf_of_θ, u_b, θ, backend)
    end
    return _cross_derivative_hvp(_make_fθ_per_col(expr, b), x_star_b, θ, u_b, hvp_backend)
end

# ── Batched rrule ────────────────────────────────────────────────────

"""
Implicit differentiation rule for `batch_solve(expr::BatchedExpression, lmo, X0, θ; ...)`.

Solves `B` independent forward problems, builds per-problem
[`_PullbackState`](@ref) objects (one-time), then returns a pullback
closure that computes ``\\partial\\theta`` by summing per-problem KKT
adjoint contributions.

Per pullback work is `O(n·B)`: one `expr.grad_per_col!` invocation per
problem on a length-`n` column, contracted with the cotangent. No
intermediate `(n, B)` buffer round trip.

The returned pullback accepts `dY::AbstractMatrix` (the cotangent of
the `(n, B)` solution matrix) and returns a 5-tuple with `dθ` in the
last position (matching the 5 positional arguments:
`batch_solve, expr, lmo, X0, θ`).
"""
function ChainRulesCore.rrule(::typeof(batch_solve), expr::BatchedExpression, lmo, X0, θ;
                              config::BatchSolveConfig=BatchSolveConfig(),
                              cache::Union{BatchCache, Nothing}=nothing,
                              kwargs...)
    cfg = _merge_config(config, kwargs)
    X_star_mat, result = batch_solve(expr, lmo, X0, θ; config=cfg, cache=cache)
    T = eltype(X_star_mat)
    n, B = size(X_star_mat)
    m = length(θ)

    as_tol = min(cfg.tol, cfg.active_set_tol)
    if !all(result.converged)
        n_unc = count(.!result.converged)
        max_gap = maximum(result.gaps)
        if max_gap > 10 * as_tol
            @warn "rrule(batch_solve): $n_unc problems not converged (max gap=$max_gap >> active set tol=$as_tol); differentiation may be inaccurate" maxlog=3
        end
    end

    oracle = _to_oracle(lmo, θ)

    # Pull each column to CPU once for the pullback pipeline.
    X_star_host = adapt(Array, X_star_mat)

    # Per-column pullback states. Built once; refactored on demand if the
    # residual-driven Tikhonov retry kicks in.
    states = [
        let f_b = _make_fθ_per_col(expr, b),
            grad_b = _make_grad_per_col(expr, b),
            x_b = X_star_host[:, b]
            _build_pullback_state(f_b, cfg.hvp_backend, x_b, θ, oracle, cfg.tol;
                                   assume_interior=cfg.assume_interior,
                                   grad=grad_b,
                                   backend=cfg.backend_ad,
                                   diff_lambda=cfg.diff_lambda,
                                   refine_active_set=cfg.refine_active_set)
        end
        for b in 1:B
    ]

    λ_data = if lmo isa ParametricOracle
        [_face_multipliers(lmo, _make_fθ_per_col(expr, b),
                           _make_grad_per_col(expr, b),
                           X_star_host[:, b], θ, states[b].as, cfg.backend_ad) for b in 1:B]
    else
        nothing
    end

    function batch_solve_pullback(dy)
        dX = _unwrap_dy(dy)
        if dX isa ChainRulesCore.AbstractZero
            return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end
        # If dX is on a device, pull it to CPU for the diff pipeline.
        dX_host = adapt(Array, dX)

        dθ_accum = zeros(T, m)
        empty!(result.diagnostics)
        sizehint!(result.diagnostics, B)

        for b in 1:B
            dx_b = @view(dX_host[:, b])
            x_b = @view(X_star_host[:, b])
            state = states[b]

            u_b, μ_bound, μ_eq, _cg = _kkt_adjoint_solve_cached(state, dx_b)
            dθ_b = _cross_dθ_per_col(expr, x_b, θ, u_b, b, cfg.backend_ad, cfg.hvp_backend)
            dθ_accum .+= dθ_b

            if λ_data !== nothing
                _add_constraint_dθ!(dθ_accum, lmo, θ, x_b,
                                    u_b, μ_bound, μ_eq, λ_data[b], state.as, cfg.backend_ad)
            end

            push!(result.diagnostics, BatchPullbackDiagnostic(
                state.last_residual_rel, state.last_cg_iters,
                state.reduced_dim, state.lambda))
        end

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), dθ_accum
    end

    return BatchSolveResult(X_star_mat, result), batch_solve_pullback
end

"""
    batch_solution_jacobian(expr::BatchedExpression, lmo, X0, θ; kwargs...) -> (J, BatchSolveResult)

Compute the Jacobian ``\\partial X^*/\\partial\\theta \\in \\mathbb{R}^{nB \\times m}``
for the batched parametric solve.

The Jacobian is organized as a `(n*B, m)` matrix where rows `(b-1)*n+1 : b*n`
correspond to problem `b`. Uses cached reduced Hessian factorization per
problem for efficiency.
"""
function batch_solution_jacobian(expr::BatchedExpression, lmo, X0, θ;
                                  config::BatchSolveConfig=BatchSolveConfig(),
                                  cache::Union{BatchCache, Nothing}=nothing,
                                  kwargs...)
    cfg = _merge_config(config, kwargs)
    X_star, result = batch_solve(expr, lmo, X0, θ; config=cfg, cache=cache)
    T = eltype(X_star)
    n, B = size(X_star)
    m = length(θ)
    oracle = _to_oracle(lmo, θ)

    X_star_host = adapt(Array, X_star)
    J = zeros(T, n * B, m)

    for b in 1:B
        x_b = X_star_host[:, b]
        f_b = _make_fθ_per_col(expr, b)
        grad_b = _make_grad_per_col(expr, b)

        state = _build_pullback_state(f_b, cfg.hvp_backend, x_b, θ, oracle, cfg.tol;
                                       assume_interior=cfg.assume_interior,
                                       grad=grad_b, backend=cfg.backend_ad,
                                       diff_lambda=cfg.diff_lambda,
                                       refine_active_set=cfg.refine_active_set)
        J_b = @view(J[(b-1)*n+1 : b*n, :])
        _solution_jacobian_impl!(J_b, state, f_b, grad_b, x_b, θ, cfg.backend_ad, cfg.hvp_backend)

        if lmo isa ParametricOracle
            λ_bound, λ_eq = _primal_face_multipliers(f_b, grad_b, x_b, θ, state.as, cfg.backend_ad)
            _add_constraint_jacobian!(J_b, lmo, state, x_b, θ, λ_bound, λ_eq, cfg.backend_ad, cfg.hvp_backend)
        end
    end

    return J, BatchSolveResult(X_star, result)
end
