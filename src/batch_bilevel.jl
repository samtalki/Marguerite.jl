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

# Batched bilevel: same shape as `batch_diff.jl`. Forward solve runs
# batched on the device; backward pass is B independent scalar
# `_kkt_adjoint_solve` calls on the host (single-shot CG, no cached
# factorization).

"""
    batch_bilevel_solve(outer, inner, lmo, X0, θ; config=BatchSolveConfig(), kwargs...) -> (X, dθ, cg_results)

Solve `B` independent inner problems and compute the summed gradient of
``\\sum_b L_b(x^*_b(\\theta))`` w.r.t. the shared parameters ``\\theta``.

Each inner problem is:
```math
x^*_b(\\theta) = \\arg\\min_{x \\in C(\\theta)} f_b(x, \\theta)
```

If `lmo` is a [`ParametricOracle`](@ref), gradients flow through both the
objective and constraint set via KKT adjoint differentiation.

# Arguments
- `outer::BatchedExpression`: outer loss carrying `f_per_col(x_b, θ, b)`
  and `grad_per_col!(g, x_b, θ, b)`.
- `inner::BatchedExpression`: inner objective with the same per-column shape.
- `lmo`: oracle (applied per column). If `ParametricOracle`, materialized at `θ`.
- `X0`: `(n, B)` initial point matrix.
- `θ`: shared parameter vector.

Per-call kwargs override matching fields of `config`.
"""
function batch_bilevel_solve(outer::BatchedExpression, inner::BatchedExpression,
                             lmo, X0::AbstractMatrix, θ;
                             config::BatchSolveConfig=BatchSolveConfig(),
                             cache::Union{BatchCache, Nothing}=nothing,
                             kwargs...)
    cfg = _merge_config(config, kwargs)
    oracle = _to_oracle(lmo, θ)

    X_star, inner_result = batch_solve(inner, lmo, X0, θ; config=cfg, cache=cache)
    if !all(inner_result.converged)
        n_unc = count(.!inner_result.converged)
        max_gap = maximum(inner_result.gaps)
        as_tol = min(cfg.tol, ACTIVE_SET_TOL_CEILING)
        if !isfinite(max_gap) || max_gap > 10 * as_tol
            @warn "batch_bilevel_solve: $n_unc/$(length(inner_result.converged)) inner problems did not converge (max gap=$max_gap); bilevel gradients may be inaccurate" maxlog=3
        end
    end

    T = eltype(X_star)
    n, B = size(X_star)
    m = length(θ)
    X_star_host = adapt(Array, X_star)
    dθ_total = zeros(T, m)
    cg_results = Vector{CGResult{T}}(undef, B)
    as_tol = min(cfg.tol, ACTIVE_SET_TOL_CEILING)

    for b in 1:B
        x_b = X_star_host[:, b]

        as_b = _active_set_for_diff(oracle, x_b;
                                     tol=as_tol,
                                     assume_interior=cfg.assume_interior,
                                     caller="batch_bilevel_solve")

        outer_grad_b! = _make_grad_per_col(outer, b)
        dx_b = similar(x_b)
        outer_grad_b!(dx_b, x_b, θ)

        inner_b = _make_fθ_per_col(inner, b)
        inner_grad_b! = _make_grad_per_col(inner, b)

        u_b, μ_bound, μ_eq, cg_b = _kkt_adjoint_solve(inner_b, cfg.hvp_backend, x_b, θ, dx_b, as_b;
                                                     cg_maxiter=cfg.diff_cg_maxiter,
                                                     cg_tol=cfg.diff_cg_tol,
                                                     cg_λ=cfg.diff_lambda,
                                                     grad=inner_grad_b!,
                                                     backend=cfg.backend_ad)
        cg_results[b] = cg_b

        ∇ₓf_of_θ = _make_∇ₓf_of_θ(inner_grad_b!, x_b)
        dθ_b = _cross_derivative_manual(∇ₓf_of_θ, u_b, θ, cfg.backend_ad)
        dθ_total .+= dθ_b

        if lmo isa ParametricOracle
            λ_bound, λ_eq = _primal_face_multipliers(inner_b, inner_grad_b!, x_b, θ, as_b, cfg.backend_ad)
            dθ_con = _constraint_pullback(lmo, θ, x_b, u_b, μ_bound, μ_eq,
                                          λ_bound, λ_eq, as_b, cfg.backend_ad)
            dθ_total .+= dθ_con
        end
    end

    nc = count(c -> c.converged, cg_results)
    if nc < B
        @warn "batch_bilevel_solve: $(B - nc)/$B CG solves did not converge; dθ may be inaccurate" maxlog=3
    end

    return BatchBilevelResult(X_star_host, dθ_total, cg_results)
end

"""
    batch_bilevel_gradient(outer, inner, lmo, X0, θ; kwargs...) -> dθ

Convenience wrapper: returns only ``\\nabla_\\theta \\sum_b L_b(x^*_b(\\theta))``.
See [`batch_bilevel_solve`](@ref) for full documentation.
"""
function batch_bilevel_gradient(outer::BatchedExpression, inner::BatchedExpression, lmo, X0, θ; kwargs...)
    _, dθ, _ = batch_bilevel_solve(outer, inner, lmo, X0, θ; kwargs...)
    return dθ
end
