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
    batch_bilevel_solve(outer, inner, lmo, X0, θ; config=BatchSolveConfig(), kwargs...) -> (X, dθ, cg_results)

Solve `B` independent inner problems and compute the summed gradient of
``\\sum_b L_b(x^*_b(\\theta))`` w.r.t. the shared parameters ``\\theta``.

Each inner problem is:
```math
x^*_b(\\theta) = \\arg\\min_{x \\in C(\\theta)} f_b(x, \\theta)
```

The outer losses ``L_b`` are evaluated per problem and their hypergradients
are accumulated.

If `lmo` is a [`ParametricOracle`](@ref), gradients flow through both the
objective and constraint set via KKT adjoint differentiation.

# Arguments
- `outer::BatchedExpression`: outer loss carrying `f_per_col(x_b, θ, b)` and
  `grad_per_col!(g, x_b, θ, b)`. The cross derivative is not used (outer is
  evaluated at fixed `θ`).
- `inner::BatchedExpression`: inner objective with the same per-column shape;
  may also carry an analytic `cross_hvp` for the inner cross derivative.
- `lmo`: oracle (applied column-wise). If `ParametricOracle`, materialized at `θ`.
- `X0`: `(n, B)` initial point matrix
- `θ`: shared parameter vector

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
        @warn "batch_bilevel_solve: $n_unc/$(length(inner_result.converged)) inner problems did not converge; bilevel gradients may be inaccurate" maxlog=3
    end

    T = eltype(X_star)
    n, B = size(X_star)
    m = length(θ)

    X_star_host = adapt(Array, X_star)
    dθ_total = zeros(T, m)
    cg_results = Vector{CGResult{T}}(undef, B)
    as_tol = min(cfg.tol, cfg.active_set_tol)

    for b in 1:B
        x_b = X_star_host[:, b]

        # Active set for problem b
        as_b = _active_set_for_diff(oracle, x_b;
                                     tol=as_tol,
                                     assume_interior=cfg.assume_interior,
                                     caller="batch_bilevel_solve")

        # Outer loss gradient at the inner optimum
        outer_grad = similar(x_b)
        if outer.grad_per_col! !== nothing
            outer.grad_per_col!(outer_grad, x_b, θ, b)
            dx_b = outer_grad
        else
            outer_b = let b_=b; x -> outer.f_per_col(x, θ, b_) end
            prep_outer = DI.prepare_gradient(outer_b, cfg.backend_ad, x_b)
            dx_b = DI.gradient(outer_b, prep_outer, cfg.backend_ad, x_b)
        end

        # Inner per-column closures
        inner_b = _make_fθ_per_col(inner, b)
        grad_b = _make_grad_per_col(inner, b)

        # KKT adjoint solve (single-use CG path)
        u_b, μ_bound, μ_eq, cg_b = _kkt_adjoint_solve(inner_b, cfg.hvp_backend, x_b, θ, dx_b, as_b;
                                                     cg_maxiter=cfg.diff_cg_maxiter,
                                                     cg_tol=cfg.diff_cg_tol,
                                                     cg_λ=cfg.diff_lambda,
                                                     grad=grad_b,
                                                     backend=cfg.backend_ad)
        cg_results[b] = cg_b

        # Cross derivative for the inner objective
        dθ_b = _cross_dθ_per_col(inner, x_b, θ, u_b, b, cfg.backend_ad, cfg.hvp_backend)
        dθ_total .+= dθ_b

        # Constraint sensitivity for ParametricOracle
        λ_b = _face_multipliers(lmo, inner_b, grad_b, x_b, θ, as_b, cfg.backend_ad)
        if λ_b !== nothing
            _add_constraint_dθ!(dθ_total, lmo, θ, x_b, u_b, μ_bound, μ_eq, λ_b, as_b, cfg.backend_ad)
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
