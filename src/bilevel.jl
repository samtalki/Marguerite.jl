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

All other kwargs are forwarded to `solve`.
"""
function bilevel_solve(outer_loss, f, ∇f!::Function, lmo, x0, θ;
                       backend=DEFAULT_BACKEND,
                       hvp_backend=SECOND_ORDER_BACKEND,
                       diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_λ::Real=1e-4,
                       kwargs...)
    x_star, inner_result = solve(f, ∇f!, lmo, x0, θ; backend=backend, kwargs...)
    if !inner_result.converged
        @warn "inner solve did not converge (gap=$(inner_result.gap), iters=$(inner_result.iterations)): bilevel gradient may be inaccurate" maxlog=3
    end

    as = active_set(lmo, x_star)
    x̄ = DI.gradient(outer_loss, backend, x_star)

    ∇_x_f_of_θ(θ_) = begin
        T = promote_type(eltype(x_star), eltype(θ_))
        g = similar(x_star, T)
        ∇f!(g, x_star, θ_)
        return g
    end

    θ̄, cg_result = if isempty(as.bound_indices) && isempty(as.eq_normals)
        _implicit_pullback(f, ∇_x_f_of_θ, x_star, θ, x̄, backend, hvp_backend;
                          cg_maxiter=diff_cg_maxiter, cg_tol=diff_cg_tol, cg_λ=diff_λ)
    else
        θ̄_obj, _, _, _, cg_res = _kkt_implicit_pullback(f, ∇_x_f_of_θ, x_star, θ, x̄, as,
                                                         backend, hvp_backend;
                                                         cg_maxiter=diff_cg_maxiter, cg_tol=diff_cg_tol, cg_λ=diff_λ)
        θ̄_obj, cg_res
    end
    if !cg_result.converged
        @warn "bilevel_solve: CG did not converge (residual=$(cg_result.residual_norm), iters=$(cg_result.iterations)): θ̄ may be inaccurate" maxlog=3
    end
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
                       kwargs...)
    x_star, inner_result = solve(f, lmo, x0, θ; backend=backend, kwargs...)
    if !inner_result.converged
        @warn "inner solve did not converge (gap=$(inner_result.gap), iters=$(inner_result.iterations)): bilevel gradient may be inaccurate" maxlog=3
    end

    as = active_set(lmo, x_star)
    x̄ = DI.gradient(outer_loss, backend, x_star)

    θ̄, cg_result = if isempty(as.bound_indices) && isempty(as.eq_normals)
        _implicit_pullback_hvp(f, x_star, θ, x̄, hvp_backend;
                              cg_maxiter=diff_cg_maxiter, cg_tol=diff_cg_tol, cg_λ=diff_λ)
    else
        θ̄_obj, _, _, _, cg_res = _kkt_implicit_pullback_hvp(f, x_star, θ, x̄, as, hvp_backend;
                                                             cg_maxiter=diff_cg_maxiter, cg_tol=diff_cg_tol, cg_λ=diff_λ)
        θ̄_obj, cg_res
    end
    if !cg_result.converged
        @warn "bilevel_solve: CG did not converge (residual=$(cg_result.residual_norm), iters=$(cg_result.iterations)): θ̄ may be inaccurate" maxlog=3
    end
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
                       kwargs...)
    x_star, inner_result = solve(f, ∇f!, plmo, x0, θ; backend=backend, kwargs...)
    if !inner_result.converged
        @warn "inner solve did not converge (gap=$(inner_result.gap), iters=$(inner_result.iterations)): bilevel gradient may be inaccurate" maxlog=3
    end

    lmo = materialize(plmo, θ)
    as = active_set(lmo, x_star)
    x̄ = DI.gradient(outer_loss, backend, x_star)

    ∇_x_f_of_θ(θ_) = begin
        T = promote_type(eltype(x_star), eltype(θ_))
        g = similar(x_star, T)
        ∇f!(g, x_star, θ_)
        return g
    end

    θ̄_obj, u, μ_bound, μ_eq, cg_result = _kkt_implicit_pullback(
        f, ∇_x_f_of_θ, x_star, θ, x̄, as, backend, hvp_backend;
        cg_maxiter=diff_cg_maxiter, cg_tol=diff_cg_tol, cg_λ=diff_λ)

    if !cg_result.converged
        @warn "bilevel_solve: CG did not converge (residual=$(cg_result.residual_norm), iters=$(cg_result.iterations)): θ̄ may be inaccurate" maxlog=3
    end

    θ̄_con = _constraint_pullback(plmo, θ, x_star, μ_bound, μ_eq, as, backend)
    θ̄ = θ̄_obj .+ θ̄_con

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
                       kwargs...)
    x_star, inner_result = solve(f, plmo, x0, θ; backend=backend, kwargs...)
    if !inner_result.converged
        @warn "inner solve did not converge (gap=$(inner_result.gap), iters=$(inner_result.iterations)): bilevel gradient may be inaccurate" maxlog=3
    end

    lmo = materialize(plmo, θ)
    as = active_set(lmo, x_star)
    x̄ = DI.gradient(outer_loss, backend, x_star)

    θ̄_obj, u, μ_bound, μ_eq, cg_result = _kkt_implicit_pullback_hvp(
        f, x_star, θ, x̄, as, hvp_backend;
        cg_maxiter=diff_cg_maxiter, cg_tol=diff_cg_tol, cg_λ=diff_λ)

    if !cg_result.converged
        @warn "bilevel_solve: CG did not converge (residual=$(cg_result.residual_norm), iters=$(cg_result.iterations)): θ̄ may be inaccurate" maxlog=3
    end

    θ̄_con = _constraint_pullback(plmo, θ, x_star, μ_bound, μ_eq, as, backend)
    θ̄ = θ̄_obj .+ θ̄_con

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
