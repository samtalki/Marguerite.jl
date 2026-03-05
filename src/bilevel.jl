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

# ------------------------------------------------------------------
# Shared bilevel core: active_set → outer gradient → KKT solve → θ̄
# ------------------------------------------------------------------

"""
    _bilevel_core(outer_loss, f, x_star, θ, lmo, cross_deriv_fn; kwargs...)

Compute the bilevel parameter gradient after the inner solve.

Shared by all `bilevel_solve` variants. Computes the outer-loss gradient at
`x_star`, solves the KKT adjoint system, and combines objective and (optional)
constraint sensitivity into ``\\bar{\\theta}``.
"""
function _bilevel_core(outer_loss, f, x_star, θ, lmo, cross_deriv_fn;
                        plmo=nothing, backend, hvp_backend,
                        tol, diff_cg_maxiter, diff_cg_tol, diff_λ)
    as = active_set(lmo, x_star; tol=max(tol, _ACTIVE_SET_MIN_TOL))
    x̄ = DI.gradient(outer_loss, backend, x_star)

    u, μ_bound, μ_eq, cg_result = _kkt_adjoint_solve(
        f, hvp_backend, x_star, θ, x̄, as;
        cg_maxiter=diff_cg_maxiter, cg_tol=diff_cg_tol, cg_λ=diff_λ)

    θ̄ = cross_deriv_fn(u)

    if plmo !== nothing
        θ̄_con = _constraint_pullback(plmo, θ, x_star, μ_bound, μ_eq, as, backend)
        θ̄ = θ̄ .+ θ̄_con
    end

    if !cg_result.converged
        @warn "bilevel_solve: CG did not converge (residual=$(cg_result.residual_norm), iters=$(cg_result.iterations)): θ̄ may be inaccurate" maxlog=3
    end

    return θ̄, cg_result
end

# ------------------------------------------------------------------
# bilevel_solve: standard oracle (manual + auto gradient)
# ------------------------------------------------------------------

"""
    bilevel_solve(outer_loss, f, ∇f!, lmo, x0, θ; kwargs...) -> (x_star, θ_grad, cg_result)

Solve the inner problem and compute the gradient of ``L(x^*(\\theta))`` w.r.t. ``\\theta``.

Returns `(x_star, θ_grad, cg_result)` where `x_star` is the inner solution, `θ_grad` is
``\\nabla_\\theta L(x^*(\\theta))``, and `cg_result::CGResult` contains CG solver diagnostics.

`outer_loss(x) -> Real` takes only the inner solution. If the user's outer loss
depends on ``\\theta`` directly, close over it and add the direct gradient manually.

# Differentiation keyword arguments
- `backend`: AD backend for first-order gradients (default: `DEFAULT_BACKEND`)
- `hvp_backend`: AD backend for Hessian-vector products (default: `SECOND_ORDER_BACKEND`)
- `diff_cg_maxiter::Int=50`: max CG iterations for the Hessian solve
- `diff_cg_tol::Real=1e-6`: CG convergence tolerance
- `diff_λ::Real=1e-4`: Tikhonov regularization for the Hessian
- `tol::Real=1e-7`: inner solve convergence tolerance (also used for active-set identification)

All other kwargs are forwarded to `solve`.
"""
function bilevel_solve(outer_loss, f, ∇f!::Function, lmo, x0, θ;
                       backend=DEFAULT_BACKEND,
                       hvp_backend=SECOND_ORDER_BACKEND,
                       diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_λ::Real=1e-4,
                       tol::Real=1e-7,
                       kwargs...)
    x_star, inner_result = solve(f, ∇f!, lmo, x0, θ; backend=backend, tol=tol, kwargs...)
    if !inner_result.converged
        @warn "inner solve did not converge (gap=$(inner_result.gap), iters=$(inner_result.iterations)): bilevel gradient may be inaccurate" maxlog=3
    end

    ∇_x_f_of_θ = _make_∇_x_f_of_θ(∇f!, x_star)
    cross_deriv_fn = u -> _cross_derivative_manual(∇_x_f_of_θ, u, θ, backend)
    θ̄, cg_result = _bilevel_core(outer_loss, f, x_star, θ, lmo, cross_deriv_fn;
                                   plmo=nothing, backend, hvp_backend,
                                   tol, diff_cg_maxiter, diff_cg_tol, diff_λ)
    return x_star, θ̄, cg_result
end

"""
    bilevel_solve(outer_loss, f, lmo, x0, θ; kwargs...) -> (x_star, θ_grad, cg_result)

Auto-gradient variant. Uses a joint HVP on the concatenated ``[x;\\, \\theta]``
space to compute the cross-derivative without nested AD.

Accepts the same differentiation keyword arguments as the manual-gradient variant:
`backend`, `hvp_backend`, `diff_cg_maxiter`, `diff_cg_tol`, `diff_λ`.
"""
function bilevel_solve(outer_loss, f, lmo, x0, θ;
                       backend=DEFAULT_BACKEND,
                       hvp_backend=SECOND_ORDER_BACKEND,
                       diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_λ::Real=1e-4,
                       tol::Real=1e-7,
                       kwargs...)
    x_star, inner_result = solve(f, lmo, x0, θ; backend=backend, tol=tol, kwargs...)
    if !inner_result.converged
        @warn "inner solve did not converge (gap=$(inner_result.gap), iters=$(inner_result.iterations)): bilevel gradient may be inaccurate" maxlog=3
    end

    cross_deriv_fn = u -> _cross_derivative_hvp(f, x_star, θ, u, hvp_backend)
    θ̄, cg_result = _bilevel_core(outer_loss, f, x_star, θ, lmo, cross_deriv_fn;
                                   plmo=nothing, backend, hvp_backend,
                                   tol, diff_cg_maxiter, diff_cg_tol, diff_λ)
    return x_star, θ̄, cg_result
end

"""
    bilevel_gradient(outer_loss, f, ∇f!, lmo, x0, θ; kwargs...) -> θ_grad

Convenience wrapper: returns only the parameter gradient `∇_θ L(x*(θ))`.
See [`bilevel_solve`](@ref) for full documentation.
"""
function bilevel_gradient(outer_loss, f, ∇f!::Function, lmo, x0, θ; kwargs...)
    _, θ̄, _ = bilevel_solve(outer_loss, f, ∇f!, lmo, x0, θ; kwargs...)
    return θ̄
end

"""
    bilevel_gradient(outer_loss, f, lmo, x0, θ; kwargs...) -> θ_grad

Auto-gradient variant. Returns only the parameter gradient.
"""
function bilevel_gradient(outer_loss, f, lmo, x0, θ; kwargs...)
    _, θ̄, _ = bilevel_solve(outer_loss, f, lmo, x0, θ; kwargs...)
    return θ̄
end

# ------------------------------------------------------------------
# ParametricOracle bilevel methods
# ------------------------------------------------------------------

"""
    bilevel_solve(outer_loss, f, ∇f!, plmo::ParametricOracle, x0, θ; kwargs...) -> (x_star, θ_grad, cg_result)

Bilevel solve with parameterized constraints. Computes gradients through both
the objective and constraint set via KKT adjoint differentiation.

Accepts the same differentiation keyword arguments as the manual-gradient [`bilevel_solve`](@ref).
"""
function bilevel_solve(outer_loss, f, ∇f!::Function, plmo::ParametricOracle, x0, θ;
                       backend=DEFAULT_BACKEND,
                       hvp_backend=SECOND_ORDER_BACKEND,
                       diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_λ::Real=1e-4,
                       tol::Real=1e-7,
                       kwargs...)
    x_star, inner_result = solve(f, ∇f!, plmo, x0, θ; backend=backend, tol=tol, kwargs...)
    if !inner_result.converged
        @warn "inner solve did not converge (gap=$(inner_result.gap), iters=$(inner_result.iterations)): bilevel gradient may be inaccurate" maxlog=3
    end

    lmo = materialize(plmo, θ)
    ∇_x_f_of_θ = _make_∇_x_f_of_θ(∇f!, x_star)
    cross_deriv_fn = u -> _cross_derivative_manual(∇_x_f_of_θ, u, θ, backend)
    θ̄, cg_result = _bilevel_core(outer_loss, f, x_star, θ, lmo, cross_deriv_fn;
                                   plmo, backend, hvp_backend,
                                   tol, diff_cg_maxiter, diff_cg_tol, diff_λ)
    return x_star, θ̄, cg_result
end

"""
    bilevel_solve(outer_loss, f, plmo::ParametricOracle, x0, θ; kwargs...) -> (x_star, θ_grad, cg_result)

Auto-gradient bilevel solve with parameterized constraints.

Accepts the same differentiation keyword arguments as the manual-gradient [`bilevel_solve`](@ref).
"""
function bilevel_solve(outer_loss, f, plmo::ParametricOracle, x0, θ;
                       backend=DEFAULT_BACKEND,
                       hvp_backend=SECOND_ORDER_BACKEND,
                       diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_λ::Real=1e-4,
                       tol::Real=1e-7,
                       kwargs...)
    x_star, inner_result = solve(f, plmo, x0, θ; backend=backend, tol=tol, kwargs...)
    if !inner_result.converged
        @warn "inner solve did not converge (gap=$(inner_result.gap), iters=$(inner_result.iterations)): bilevel gradient may be inaccurate" maxlog=3
    end

    lmo = materialize(plmo, θ)
    cross_deriv_fn = u -> _cross_derivative_hvp(f, x_star, θ, u, hvp_backend)
    θ̄, cg_result = _bilevel_core(outer_loss, f, x_star, θ, lmo, cross_deriv_fn;
                                   plmo, backend, hvp_backend,
                                   tol, diff_cg_maxiter, diff_cg_tol, diff_λ)
    return x_star, θ̄, cg_result
end

"""
    bilevel_gradient(outer_loss, f, ∇f!, plmo::ParametricOracle, x0, θ; kwargs...) -> θ_grad

Bilevel gradient with parameterized constraints.
"""
function bilevel_gradient(outer_loss, f, ∇f!::Function, plmo::ParametricOracle, x0, θ; kwargs...)
    _, θ̄, _ = bilevel_solve(outer_loss, f, ∇f!, plmo, x0, θ; kwargs...)
    return θ̄
end

"""
    bilevel_gradient(outer_loss, f, plmo::ParametricOracle, x0, θ; kwargs...) -> θ_grad

Auto-gradient bilevel gradient with parameterized constraints.
"""
function bilevel_gradient(outer_loss, f, plmo::ParametricOracle, x0, θ; kwargs...)
    _, θ̄, _ = bilevel_solve(outer_loss, f, plmo, x0, θ; kwargs...)
    return θ̄
end
