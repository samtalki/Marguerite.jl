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
    bilevel_solve(outer_loss, inner_loss, lmo, x0, θ; grad=nothing, cross_deriv=nothing, kwargs...) -> (x_star, θ_grad, cg_result)

Solve the inner problem and compute the gradient of ``L(x^*(\\theta))`` w.r.t. ``\\theta``.

If `lmo` is a [`ParametricOracle`](@ref), computes gradients through both the
objective and constraint set via KKT adjoint differentiation.

Returns `(x_star, θ_grad, cg_result)` where `x_star` is the inner solution, `θ_grad` is
``\\nabla_\\theta L(x^*(\\theta))``, and `cg_result::CGResult` contains CG solver diagnostics.

`outer_loss(x) -> Real` takes only the inner solution. If the user's outer loss
depends on ``\\theta`` directly, close over it and add the direct gradient manually.

# Keyword Arguments
- `grad`: in-place gradient `grad(g, x, θ)`. If `nothing` (default), auto-computed.
- `cross_deriv`: manual cross-derivative `cross_deriv(u, θ) -> dθ`. If provided,
  computes ``-(\\partial^2 f / \\partial\\theta\\partial x)^T u`` directly, bypassing
  AD-based cross-derivative computation. This can be much faster when the
  cross-derivative has closed-form structure (e.g. linear in ``u``).
- `backend`: AD backend for first-order gradients (default: `DEFAULT_BACKEND`)
- `hvp_backend`: AD backend for Hessian-vector products (default: `SECOND_ORDER_BACKEND`)
- `diff_cg_maxiter::Int=50`: max CG iterations for the Hessian solve
- `diff_cg_tol::Real=1e-6`: CG convergence tolerance
- `diff_lambda::Real=1e-4`: Tikhonov regularization for the Hessian
- `assume_interior::Bool=false`: for custom oracles lacking [`active_set`](@ref),
  error by default; when `true`, use the interior active-set approximation
- `tol::Real=1e-4`: inner solve convergence tolerance (also used for active-set identification)

All other kwargs are forwarded to `solve`.
"""
function bilevel_solve(outer_loss, inner_loss, lmo, x0, θ;
                       grad=nothing,
                       cross_deriv=nothing,
                       backend=DEFAULT_BACKEND,
                       hvp_backend=SECOND_ORDER_BACKEND,
                       diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_lambda::Real=1e-4,
                       assume_interior::Bool=false,
                       tol::Real=1e-4,
                       kwargs...)
    if lmo isa ParametricOracle
        oracle = materialize(lmo, θ)
    else
        oracle = lmo
    end
    x_star, inner_result = solve(inner_loss, lmo, x0, θ;
                                 grad=grad, backend=backend,
                                 assume_interior=assume_interior,
                                 tol=tol, kwargs...)
    if !inner_result.converged
        @warn "inner solve did not converge (gap=$(inner_result.gap), iters=$(inner_result.iterations)): bilevel gradient may be inaccurate" maxlog=3
    end

    as = _active_set_for_diff(oracle, x_star;
                               tol=min(tol, 1e-6),
                               assume_interior=assume_interior,
                               caller="bilevel_solve")
    dx = DI.gradient(outer_loss, backend, x_star)
    λ_bound = eltype(x_star)[]
    λ_eq = eltype(x_star)[]
    if lmo isa ParametricOracle
        λ_bound, λ_eq = _primal_face_multipliers(inner_loss, grad, x_star, θ, as, backend)
    end

    # KKT adjoint solve (shared by all cross-derivative paths)
    u, μ_bound, μ_eq, cg_result = _kkt_adjoint_solve(inner_loss, hvp_backend, x_star, θ, dx, as;
                                                       cg_maxiter=diff_cg_maxiter,
                                                       cg_tol=diff_cg_tol,
                                                       cg_λ=diff_lambda,
                                                       grad=grad,
                                                       backend=backend)

    # Cross-derivative: dispatch to manual, AD-with-grad, or AD-with-HVP
    if cross_deriv !== nothing
        dθ_obj = cross_deriv(u, θ)
    elseif grad !== nothing
        ∇ₓf_of_θ = _make_∇ₓf_of_θ(grad, x_star)
        dθ_obj = _cross_derivative_manual(∇ₓf_of_θ, u, θ, backend)
    else
        dθ_obj = _cross_derivative_hvp(inner_loss, x_star, θ, u, hvp_backend)
    end

    if lmo isa ParametricOracle
        dθ_con = _constraint_pullback(lmo, θ, x_star, u, μ_bound, μ_eq, λ_bound, λ_eq, as, backend)
        dθ = dθ_obj .+ dθ_con
    else
        dθ = dθ_obj
    end

    if !cg_result.converged
        @warn "bilevel_solve: CG did not converge (residual=$(cg_result.residual_norm), iters=$(cg_result.iterations)): dθ may be inaccurate" maxlog=3
    end
    return BilevelResult(x_star, dθ, cg_result)
end

"""
    bilevel_gradient(outer_loss, inner_loss, lmo, x0, θ; kwargs...) -> θ_grad

Convenience wrapper: returns only `∇_θ L(x*(θ))`.
See [`bilevel_solve`](@ref) for full documentation.
"""
function bilevel_gradient(outer_loss, inner_loss, lmo, x0, θ; kwargs...)
    _, dθ, _ = bilevel_solve(outer_loss, inner_loss, lmo, x0, θ; kwargs...)
    return dθ
end
