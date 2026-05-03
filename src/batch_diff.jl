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

# Batched differentiation. The forward solve runs as a fused KA kernel
# launch per FW iter; the backward pass is B independent scalar pullbacks
# on the host (no batched HVP, no batched factorization). The convenience
# wrapper here shares `BatchSolveConfig` and the device pull with
# `batch_solve`, but the per-column work is `_build_pullback_state` plus
# `_kkt_adjoint_solve_cached` from `diff_rules.jl`.

# ── Cotangent unwrap ─────────────────────────────────────────────────

_unwrap_dy(dy::Tuple) = isempty(dy) ? nothing : dy[1]
_unwrap_dy(dy::BatchSolveResult) = dy.X
_unwrap_dy(dy::ChainRulesCore.Tangent{<:BatchSolveResult}) = dy.X
_unwrap_dy(dy::ChainRulesCore.Tangent) = throw(ArgumentError(
    "rrule(batch_solve): expected Tangent{<:BatchSolveResult} cotangent; got $(typeof(dy))"))
_unwrap_dy(dy) = dy

# ── Batched rrule ────────────────────────────────────────────────────

"""
Implicit differentiation rule for `batch_solve(expr, lmo, X0, θ; ...)`.

Solves `B` independent forward problems via the batched (KA-kernelled)
solve, then differentiates each column on the host with the scalar
pullback machinery (`_build_pullback_state` + `_kkt_adjoint_solve_cached`).
There is no batched HVP or batched factorization in the backward pass.

Returns a 5-tuple cotangent `(NoTangent(), NoTangent(), NoTangent(),
NoTangent(), dθ)` matching the 5 positional arguments
`batch_solve, expr, lmo, X0, θ`.
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

    if !all(result.converged)
        n_unc = count(.!result.converged)
        max_gap = maximum(result.gaps)
        as_tol = min(cfg.tol, ACTIVE_SET_TOL_CEILING)
        if !isfinite(max_gap) || max_gap > 10 * as_tol
            @warn "rrule(batch_solve): $n_unc problems not converged (max gap=$max_gap); differentiation may be inaccurate" maxlog=3
        end
    end

    oracle = _to_oracle(lmo, θ)
    X_star_host = adapt(Array, X_star_mat)

    # Pre-build per-column closures and pullback states. The closures are
    # captured by reference inside the pullback so we don't rebuild them on
    # each pullback call. States are refactored on demand if the
    # residual-driven Tikhonov retry kicks in inside _kkt_adjoint_solve_cached.
    grad_per_col = [_make_grad_per_col(expr, b) for b in 1:B]
    f_per_col_θ  = [_make_fθ_per_col(expr, b)   for b in 1:B]
    states = [_build_pullback_state(f_per_col_θ[b], cfg.hvp_backend,
                                     X_star_host[:, b], θ, oracle, cfg.tol;
                                     assume_interior=cfg.assume_interior,
                                     grad=grad_per_col[b], backend=cfg.backend_ad,
                                     diff_lambda=cfg.diff_lambda)
              for b in 1:B]
    λ_data = if lmo isa ParametricOracle
        [_primal_face_multipliers(f_per_col_θ[b], grad_per_col[b],
                                   X_star_host[:, b], θ, states[b].as, cfg.backend_ad)
         for b in 1:B]
    else
        nothing
    end

    function batch_solve_pullback(dy)
        dX_unwrapped = _unwrap_dy(dy)
        if dX_unwrapped === nothing || dX_unwrapped isa ChainRulesCore.AbstractZero
            return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end
        dX_host = adapt(Array, dX_unwrapped)

        dθ_accum = zeros(T, m)
        for b in 1:B
            state = states[b]
            x_b = view(X_star_host, :, b)
            dx_b = view(dX_host, :, b)

            u_b, μ_bound, μ_eq, _ = _kkt_adjoint_solve_cached(state, dx_b)

            ∇ₓf_of_θ_b = _make_∇ₓf_of_θ(grad_per_col[b], x_b)
            dθ_b = _cross_derivative_manual(∇ₓf_of_θ_b, u_b, θ, cfg.backend_ad)
            dθ_accum .+= dθ_b

            if λ_data !== nothing
                λ_bound, λ_eq = λ_data[b]
                dθ_con = _constraint_pullback(lmo, θ, x_b, u_b, μ_bound, μ_eq,
                                              λ_bound, λ_eq, state.as, cfg.backend_ad)
                dθ_accum .+= dθ_con
            end
        end

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), dθ_accum
    end

    return BatchSolveResult(X_star_mat, result), batch_solve_pullback
end

"""
    batch_solution_jacobian(expr, lmo, X0, θ; kwargs...) -> (J, BatchSolveResult)

Compute the Jacobian ``\\partial X^*/\\partial\\theta \\in \\mathbb{R}^{nB \\times m}``
for the batched parametric solve.

The Jacobian is organized as a `(n*B, m)` matrix where rows
`(b-1)*n+1 : b*n` correspond to problem `b`. Each column is differentiated
independently using the scalar `_build_pullback_state` plus a cached
reduced Hessian factorization.
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
        grad_b! = _make_grad_per_col(expr, b)

        state = _build_pullback_state(f_b, cfg.hvp_backend, x_b, θ, oracle, cfg.tol;
                                       assume_interior=cfg.assume_interior,
                                       grad=grad_b!, backend=cfg.backend_ad,
                                       diff_lambda=cfg.diff_lambda)
        J_b = @view(J[(b-1)*n+1 : b*n, :])
        _solution_jacobian_impl!(J_b, state, f_b, grad_b!, x_b, θ, cfg.backend_ad, cfg.hvp_backend)

        if lmo isa ParametricOracle
            λ_bound, λ_eq = _primal_face_multipliers(f_b, grad_b!, x_b, θ, state.as, cfg.backend_ad)
            _add_constraint_jacobian!(J_b, lmo, state, x_b, θ, λ_bound, λ_eq, cfg.backend_ad, cfg.hvp_backend)
        end
    end

    return J, BatchSolveResult(X_star, result)
end
