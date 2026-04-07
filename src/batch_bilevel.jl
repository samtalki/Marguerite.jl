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
    batch_bilevel_solve(outer_batch, inner_batch, lmo, X0, θ; kwargs...) -> (X, dθ, cg_results)

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
- `outer_batch(X::AbstractMatrix) -> AbstractVector`: per-problem outer losses (length `B`)
- `inner_batch(X::AbstractMatrix, θ) -> AbstractVector`: per-problem inner objectives
- `lmo`: oracle (applied column-wise). If `ParametricOracle`, materialized at `θ`.
- `X0`: `(n, B)` initial point matrix
- `θ`: shared parameter vector

# Keyword Arguments
- `grad_batch`: in-place gradient `grad_batch(G, X, θ)`. If `nothing`, auto-computed.
- `cross_deriv_batch`: manual cross-derivative `cross_deriv_batch(u, θ, b) -> dθ_b` per problem.
  Must return ``-(\\partial^2 f_b / \\partial\\theta\\partial x)^T u``.
- `backend`: AD backend for first-order gradients (default: `DEFAULT_BACKEND`)
- `hvp_backend`: AD backend for Hessian-vector products (default: `SECOND_ORDER_BACKEND`)
- `diff_cg_maxiter::Int=50`: max CG iterations per problem
- `diff_cg_tol::Real=1e-6`: CG convergence tolerance
- `diff_lambda::Real=1e-4`: Tikhonov regularization
- `assume_interior::Bool=false`: use interior approximation for custom oracles
- `tol::Real=1e-4`: inner solve convergence tolerance

All other kwargs are forwarded to `batch_solve`.
"""
function batch_bilevel_solve(outer_batch, inner_batch, lmo, X0::AbstractMatrix, θ;
                             grad_batch=nothing,
                             cross_deriv_batch=nothing,
                             backend=DEFAULT_BACKEND,
                             hvp_backend=SECOND_ORDER_BACKEND,
                             diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_lambda::Real=1e-4,
                             assume_interior::Bool=false,
                             tol::Real=1e-4,
                             kwargs...)
    oracle = _to_oracle(lmo, θ)

    X_star, inner_result = batch_solve(inner_batch, lmo, X0, θ;
                                        grad_batch=grad_batch, backend=backend,
                                        assume_interior=assume_interior,
                                        tol=tol, kwargs...)
    if !all(inner_result.converged)
        n_unc = count(.!inner_result.converged)
        @warn "batch_bilevel_solve: $n_unc/$( length(inner_result.converged)) inner problems did not converge; bilevel gradients may be inaccurate" maxlog=3
    end

    T = eltype(X_star)
    n, B = size(X_star)
    m = length(θ)

    dθ_total = zeros(T, m)
    cg_results = Vector{CGResult{T}}(undef, B)

    as_tol = min(tol, ACTIVE_SET_TOL_CEILING)

    for b in 1:B
        x_b = X_star[:, b]

        # Per-problem scalar inner loss
        inner_b(x, θ_) = inner_batch(_col_to_batch(x, b, n, B), θ_)[b]

        # Active set for problem b
        as_b = _active_set_for_diff(oracle, x_b;
                                     tol=as_tol,
                                     assume_interior=assume_interior,
                                     caller="batch_bilevel_solve")

        # Outer loss gradient for problem b
        outer_b(x) = outer_batch(_col_to_batch(x, b, n, B))[b]
        prep_outer = DI.prepare_gradient(outer_b, backend, x_b)
        dx_b = DI.gradient(outer_b, prep_outer, backend, x_b)

        # Per-problem manual gradient closure
        grad_b = grad_batch !== nothing ? _make_batch_col_grad(grad_batch, b, n, B) : nothing

        # KKT adjoint solve (CG path — single-use per problem)
        u_b, μ_bound, μ_eq, cg_b = _kkt_adjoint_solve(inner_b, hvp_backend, x_b, θ, dx_b, as_b;
                                                         cg_maxiter=diff_cg_maxiter,
                                                         cg_tol=diff_cg_tol,
                                                         cg_λ=diff_lambda,
                                                         grad=grad_b,
                                                         backend=backend)
        cg_results[b] = cg_b

        # Cross-derivative
        if cross_deriv_batch !== nothing
            dθ_b = cross_deriv_batch(u_b, θ, b)
        elseif grad_b !== nothing
            ∇ₓf_of_θ = _make_∇ₓf_of_θ(grad_b, x_b)
            dθ_b = _cross_derivative_manual(∇ₓf_of_θ, u_b, θ, backend)
        else
            dθ_b = _cross_derivative_hvp(inner_b, x_b, θ, u_b, hvp_backend)
        end

        # Constraint sensitivity for ParametricOracle
        if lmo isa ParametricOracle
            λ_bound, λ_eq = _primal_face_multipliers(inner_b, grad_b, x_b, θ, as_b, backend)
            dθ_con = _constraint_pullback(lmo, θ, x_b, u_b, μ_bound, μ_eq, λ_bound, λ_eq, as_b, backend)
            dθ_total .+= dθ_b .+ dθ_con
        else
            dθ_total .+= dθ_b
        end
    end

    nc = count(c -> c.converged, cg_results)
    if nc < B
        @warn "batch_bilevel_solve: $(B - nc)/$B CG solves did not converge; dθ may be inaccurate" maxlog=3
    end

    return BatchBilevelResult(Matrix(X_star), dθ_total, cg_results)
end

"""
    batch_bilevel_gradient(outer_batch, inner_batch, lmo, X0, θ; kwargs...) -> dθ

Convenience wrapper: returns only ``\\nabla_\\theta \\sum_b L_b(x^*_b(\\theta))``.
See [`batch_bilevel_solve`](@ref) for full documentation.
"""
function batch_bilevel_gradient(outer_batch, inner_batch, lmo, X0, θ; kwargs...)
    _, dθ, _ = batch_bilevel_solve(outer_batch, inner_batch, lmo, X0, θ; kwargs...)
    return dθ
end
