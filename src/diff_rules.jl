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
    _cg_solve(hvp_fn, rhs; maxiter=50, tol=1e-6, λ=1e-4)

Conjugate gradient solver for ``(H + \\lambda I) u = \\text{rhs}`` where ``H`` is accessed
only via Hessian-vector products `hvp_fn(d) -> Hd`.

Tikhonov regularization ``\\lambda`` ensures well-conditioned systems near
singular Hessians (e.g. on boundary of feasible set).
"""
function _cg_solve(hvp_fn, rhs::AbstractVector{T};
                   maxiter::Int=50, tol::Real=1e-6, λ::Real=1e-4) where T
    n = length(rhs)
    u = zeros(T, n)
    r = copy(rhs)     # r = rhs - (H + λI)u = rhs (since u=0)
    p = copy(r)
    r_dot_r = dot(r, r)
    converged = false
    iters = 0

    for k in 1:maxiter
        iters = k
        Hp = hvp_fn(p)
        Hp .+= λ .* p  # (H + λI)p
        pHp = dot(p, Hp)
        if pHp ≤ eps(T)
            @warn "CG encountered near-zero curvature (pHp=$pHp): Hessian may be singular. Consider increasing diff_λ." maxlog=3
            break
        end
        α = r_dot_r / pHp
        u .+= α .* p
        r .-= α .* Hp
        r_dot_r_new = dot(r, r)
        if sqrt(r_dot_r_new) < tol
            converged = true
            break
        end
        β = r_dot_r_new / r_dot_r
        p .= r .+ β .* p
        r_dot_r = r_dot_r_new
    end
    residual = sqrt(dot(r, r))
    converged = converged || residual < tol
    if !converged
        @warn "CG solve did not converge: residual=$residual after $iters iterations" maxlog=3
    end
    return u, CGResult(iters, residual, converged)
end

"""
    _hessian_cg_solve(f, hvp_backend, x_star, θ, x̄; cg_maxiter=50, cg_tol=1e-6, cg_λ=1e-4)

Solve ``(\\nabla^2_{xx} f + \\lambda I)\\, u = \\bar{x}`` via CG with HVPs.

Shared Hessian-solve step used by both [`_implicit_pullback`](@ref) and [`_implicit_pullback_hvp`](@ref).
"""
function _hessian_cg_solve(f, hvp_backend, x_star, θ, x̄;
                            cg_maxiter::Int=50, cg_tol::Real=1e-6, cg_λ::Real=1e-4)
    fθ = x_ -> f(x_, θ)
    prep_hvp = DI.prepare_hvp(fθ, hvp_backend, x_star, (x̄,))
    hvp_fn = d -> DI.hvp(fθ, prep_hvp, hvp_backend, x_star, (d,))[1]
    return _cg_solve(hvp_fn, x̄ isa AbstractVector ? x̄ : collect(x̄);
                     maxiter=cg_maxiter, tol=cg_tol, λ=cg_λ)
end

# ------------------------------------------------------------------
# KKT adjoint solve (constrained implicit differentiation)
# ------------------------------------------------------------------

"""
    _null_project(w, eq_normals, free_indices)

Project vector `w` (in free-variable space) onto the null space of the
equality constraint normals: ``P(w) = w - \\sum_j (a_j^T w / \\|a_j\\|^2) a_j``
where each ``a_j`` is restricted to free indices.
"""
function _null_project(w::AbstractVector{T}, eq_normals::Vector{Vector{T}}, free_indices::Vector{Int}) where T
    out = copy(w)
    for a_full in eq_normals
        a_free = a_full[free_indices]
        a_norm_sq = dot(a_free, a_free)
        if a_norm_sq > eps(T)
            out .-= (dot(a_free, out) / a_norm_sq) .* a_free
        end
    end
    return out
end

"""
    _kkt_adjoint_solve(f, hvp_backend, x_star, θ, x̄, as::ActiveSet;
                        cg_maxiter=50, cg_tol=1e-6, cg_λ=1e-4)

Solve the KKT adjoint system at ``x^*`` with active constraints.

The KKT system is:
```math
\\begin{bmatrix} \\nabla^2 f & G^T \\\\ G & 0 \\end{bmatrix}
\\begin{bmatrix} u \\\\ \\mu \\end{bmatrix} =
\\begin{bmatrix} \\bar{x} \\\\ 0 \\end{bmatrix}
```

Solved via reduced Hessian CG on the null space of ``G`` (active face):
1. Set ``u[\\text{bound}] = 0``, work only in free-variable subspace
2. Project ``\\bar{x}_{\\text{free}}`` to null(eq\\_normals)
3. CG solve: ``(P H_{\\text{free}} P + \\lambda I) w = P \\bar{x}_{\\text{free}}``
4. Recover ``\\mu`` from KKT residual
"""
function _kkt_adjoint_solve(f, hvp_backend, x_star, θ, x̄, as::ActiveSet{AT};
                             cg_maxiter::Int=50, cg_tol::Real=1e-6, cg_λ::Real=1e-4) where AT
    T = promote_type(AT, eltype(x_star))
    n = length(x_star)
    x̄_vec = x̄ isa AbstractVector ? x̄ : collect(x̄)

    # If no active constraints, fall back to unconstrained Hessian solve
    if isempty(as.bound_indices) && isempty(as.eq_normals)
        u, cg_result = _hessian_cg_solve(f, hvp_backend, x_star, θ, x̄_vec;
                                          cg_maxiter, cg_tol, cg_λ)
        μ_bound = T[]
        μ_eq = T[]
        return u, μ_bound, μ_eq, cg_result
    end

    free = as.free_indices
    bound = as.bound_indices
    n_free = length(free)

    # If no free variables, u = 0
    if n_free == 0
        u = zeros(T, n)
        # μ_bound from residual: G^T μ = x̄, for bound constraints e_i^T μ_i = x̄_i
        μ_bound = T[x̄_vec[i] for i in bound]
        μ_eq = T[]
        cg_result = CGResult(0, zero(T), true)
        return u, μ_bound, μ_eq, cg_result
    end

    # Prepare HVP on f(·, θ)
    fθ = x_ -> f(x_, θ)
    prep_hvp = DI.prepare_hvp(fθ, hvp_backend, x_star, (x̄_vec,))

    # Reduced HVP: expand w_free to full space → HVP → extract free → null-project
    function reduced_hvp(w_free)
        w_full = zeros(T, n)
        @inbounds for (j, idx) in enumerate(free)
            w_full[idx] = w_free[j]
        end
        Hw_full = DI.hvp(fθ, prep_hvp, hvp_backend, x_star, (w_full,))[1]
        Hw_free = Hw_full[free]
        return _null_project(Hw_free, as.eq_normals, free)
    end

    # RHS: project x̄_free onto null(eq_normals)
    x̄_free = x̄_vec[free]
    rhs = _null_project(x̄_free, as.eq_normals, free)

    # CG solve in reduced space
    u_free, cg_result = _cg_solve(reduced_hvp, rhs; maxiter=cg_maxiter, tol=cg_tol, λ=cg_λ)

    # Null-project the CG result for consistency
    u_free = _null_project(u_free, as.eq_normals, free)

    # Assemble full u
    u = zeros(T, n)
    @inbounds for (j, idx) in enumerate(free)
        u[idx] = u_free[j]
    end

    # Compute Hu for multiplier recovery
    Hu = DI.hvp(fθ, prep_hvp, hvp_backend, x_star, (u,))[1]
    residual = x̄_vec .- Hu

    # μ_bound: for bound constraint on index i, μ_i = residual[i]
    μ_bound = T[residual[i] for i in bound]

    # μ_eq: for equality constraint aᵀx = b,
    # μ_eq_j = (a_free' residual_free) / (a_free' a_free)
    μ_eq = T[]
    for a_full in as.eq_normals
        a_free = a_full[free]
        a_norm_sq = dot(a_free, a_free)
        if a_norm_sq > eps(T)
            push!(μ_eq, dot(a_free, residual[free]) / a_norm_sq)
        else
            push!(μ_eq, zero(T))
        end
    end

    return u, μ_bound, μ_eq, cg_result
end

# ------------------------------------------------------------------
# Implicit pullback (objective contribution to θ̄)
# ------------------------------------------------------------------

"""
    _implicit_pullback(f, ∇_x_f_of_θ, x_star, θ, x̄, backend, hvp_backend; cg_maxiter=50, cg_tol=1e-6, cg_λ=1e-4)

Shared pullback logic for implicit differentiation of `solve`.

1. Solves ``(\\nabla^2_{xx} f + \\lambda I)\\, u = \\bar{x}`` via CG with HVPs using `hvp_backend`.
2. Computes ``\\bar{\\theta} = -(\\partial(\\nabla_x f)/\\partial\\theta)^\\top u`` via the gradient of ``\\theta \\mapsto \\langle \\nabla_x f(\\theta), u \\rangle`` using `backend`.

`backend` handles the cross-derivative gradient; `hvp_backend` handles
second-order Hessian-vector products (``\\nabla^2_{xx} f``).

See [Implicit Differentiation](@ref) for the full derivation.
"""
function _implicit_pullback(f, ∇_x_f_of_θ, x_star, θ, x̄, backend, hvp_backend;
                            cg_maxiter::Int=50, cg_tol::Real=1e-6, cg_λ::Real=1e-4)
    u, cg_result = _hessian_cg_solve(f, hvp_backend, x_star, θ, x̄;
                                      cg_maxiter, cg_tol, cg_λ)

    ∇f_dot_u = θ_ -> dot(∇_x_f_of_θ(θ_), u)
    prep_g = DI.prepare_gradient(∇f_dot_u, backend, θ)
    θ̄ = -DI.gradient(∇f_dot_u, prep_g, backend, θ)
    return θ̄, cg_result
end

"""
    _implicit_pullback_hvp(f, x_star, θ, x̄, hvp_backend; cg_maxiter=50, cg_tol=1e-6, cg_λ=1e-4)

Auto-gradient variant of [`_implicit_pullback`](@ref) that avoids nested AD.

Computes the cross-derivative via a single HVP on the joint function
``g(z) = f(z_{1:n},\\, z_{n+1:\\text{end}})`` where ``z = [x;\\, \\theta]``.
The identity ``\\nabla^2 g \\cdot [u;\\, 0] = [\\nabla^2_{xx} f \\cdot u;\\, \\nabla^2_{\\theta x} f \\cdot u]``
extracts the cross-derivative as the last ``m`` entries.
"""
function _implicit_pullback_hvp(f, x_star, θ, x̄, hvp_backend;
                                 cg_maxiter::Int=50, cg_tol::Real=1e-6, cg_λ::Real=1e-4)
    u, cg_result = _hessian_cg_solve(f, hvp_backend, x_star, θ, x̄;
                                      cg_maxiter, cg_tol, cg_λ)

    # Cross-derivative via joint HVP (no nested AD)
    # g(z) = f(z[1:n], z[n+1:end])  where z = [x; θ]
    # ∇²g · [u; 0] = [∇²_{xx}f · u; ∇²_{θx}f · u]
    # θ̄ = -∇²_{θx}f · u
    n = length(x_star)
    m = length(θ)
    g = z -> f(z[1:n], z[n+1:end])
    z = vcat(x_star, θ)
    v = vcat(u, zeros(eltype(u), m))
    prep_cross = DI.prepare_hvp(g, hvp_backend, z, (v,))
    cross_hvp = DI.hvp(g, prep_cross, hvp_backend, z, (v,))[1]
    θ̄ = -cross_hvp[n+1:end]

    return θ̄, cg_result
end

# ------------------------------------------------------------------
# KKT-based implicit pullbacks (with active set)
# ------------------------------------------------------------------

"""
    _kkt_implicit_pullback(f, ∇_x_f_of_θ, x_star, θ, x̄, as, backend, hvp_backend; kwargs...)

KKT adjoint pullback with manual gradient. Solves the KKT system on the active
face, then computes ``\\bar{\\theta}_{\\text{obj}} = -(\\partial(\\nabla_x f)/\\partial\\theta)^T u``.
"""
function _kkt_implicit_pullback(f, ∇_x_f_of_θ, x_star, θ, x̄, as::ActiveSet,
                                 backend, hvp_backend;
                                 cg_maxiter::Int=50, cg_tol::Real=1e-6, cg_λ::Real=1e-4)
    u, μ_bound, μ_eq, cg_result = _kkt_adjoint_solve(f, hvp_backend, x_star, θ, x̄, as;
                                                       cg_maxiter, cg_tol, cg_λ)

    ∇f_dot_u = θ_ -> dot(∇_x_f_of_θ(θ_), u)
    prep_g = DI.prepare_gradient(∇f_dot_u, backend, θ)
    θ̄_obj = -DI.gradient(∇f_dot_u, prep_g, backend, θ)

    return θ̄_obj, u, μ_bound, μ_eq, cg_result
end

"""
    _kkt_implicit_pullback_hvp(f, x_star, θ, x̄, as, hvp_backend; kwargs...)

KKT adjoint pullback with auto-gradient (joint HVP, no nested AD).
"""
function _kkt_implicit_pullback_hvp(f, x_star, θ, x̄, as::ActiveSet, hvp_backend;
                                     cg_maxiter::Int=50, cg_tol::Real=1e-6, cg_λ::Real=1e-4)
    u, μ_bound, μ_eq, cg_result = _kkt_adjoint_solve(f, hvp_backend, x_star, θ, x̄, as;
                                                       cg_maxiter, cg_tol, cg_λ)

    # Cross-derivative via joint HVP
    n = length(x_star)
    m = length(θ)
    g = z -> f(z[1:n], z[n+1:end])
    z = vcat(x_star, θ)
    v = vcat(u, zeros(eltype(u), m))
    prep_cross = DI.prepare_hvp(g, hvp_backend, z, (v,))
    cross_hvp = DI.hvp(g, prep_cross, hvp_backend, z, (v,))[1]
    θ̄_obj = -cross_hvp[n+1:end]

    return θ̄_obj, u, μ_bound, μ_eq, cg_result
end

# ------------------------------------------------------------------
# Constraint pullback (constraint sensitivity contribution to θ̄)
# ------------------------------------------------------------------

"""
    _constraint_scalar(plmo::ParameterizedOracle, θ, x_star, u, μ_bound, μ_eq, as)

Compute the scalar function ``\\Phi(\\theta)`` whose gradient gives the
constraint sensitivity contribution to ``\\bar{\\theta}``.

``\\Phi(\\theta) = \\mu^T h(\\theta)`` where ``h(\\theta)`` are the active constraint RHS values.
"""
function _constraint_scalar end

function _constraint_scalar(plmo::ParameterizedBox, θ, x_star, u, μ_bound, μ_eq, as::ActiveSet{T}) where T
    # For box constraints: active constraints are x_i = lb_i or x_i = ub_i
    # Φ(θ) = ∑_i μ_i · bound_value_i(θ)
    lb = plmo.lb_fn(θ)
    ub = plmo.ub_fn(θ)
    s = zero(eltype(θ))
    for (k, i) in enumerate(as.bound_indices)
        bv = as.bound_values[k]
        # Determine if this is a lower or upper bound
        if abs(bv - lb[i]) ≤ eps(T) * 10
            s += μ_bound[k] * lb[i]
        else
            s += μ_bound[k] * ub[i]
        end
    end
    return s
end

function _constraint_scalar(plmo::ParameterizedSimplex{R, true}, θ, x_star, u, μ_bound, μ_eq, as::ActiveSet) where R
    # Probability simplex: x_i ≥ 0 (bounds, no θ-dependence) and ∑x_i = r(θ)
    # Φ(θ) = μ_budget · r(θ)
    # Bound constraints on x_i ≥ 0 don't depend on θ, so no contribution
    r = plmo.r_fn(θ)
    if !isempty(μ_eq)
        return μ_eq[1] * r
    end
    return zero(eltype(θ))
end

function _constraint_scalar(plmo::ParameterizedSimplex{R, false}, θ, x_star, u, μ_bound, μ_eq, as::ActiveSet) where R
    # Capped simplex: same structure but budget may not be active
    r = plmo.r_fn(θ)
    if !isempty(μ_eq)
        return μ_eq[1] * r
    end
    return zero(eltype(θ))
end

function _constraint_scalar(plmo::ParameterizedWeightedSimplex, θ, x_star, u, μ_bound, μ_eq, as::ActiveSet{T}) where T
    # Weighted simplex: x_i ≥ lb_i(θ) (bounds) and ⟨α(θ), x⟩ ≤ β(θ)
    # Φ(θ) = ∑_{bound} μ_i · lb(θ)[i] + μ_budget · β(θ)
    #       - μ_budget · ⟨α(θ), lb(θ)⟩  (from WeightedSimplex's shifted formulation)
    # Actually the constraints are:
    #   bound: x_i = lb_i(θ)  → contribution: μ_bound_i · lb_i(θ)
    #   budget: ⟨α(θ), x⟩ = β(θ) → contribution: μ_eq · β(θ)
    # But the equality normal also depends on θ through α, so we need:
    #   full contribution = μ_bound' · lb(θ)[bound] + μ_eq · (β(θ) - ⟨α(θ), x*⟩)
    # Wait -- at x*, ⟨α(θ), x*⟩ = β(θ) so that's zero. The actual
    # sensitivity comes from: μ_eq · β(θ) for the RHS, and
    # the normal-variation term -μ_eq · ⟨∂α/∂θ · dθ, x*⟩ which
    # is captured by AD through the full scalar.
    lb = plmo.lb_fn(θ)
    α = plmo.α_fn(θ)
    β = plmo.β_fn(θ)
    s = zero(eltype(θ))
    # Bound contributions: μ_bound_i · lb_i(θ)
    for (k, i) in enumerate(as.bound_indices)
        s += μ_bound[k] * lb[i]
    end
    # Budget equality: μ_eq · (β(θ) - ⟨α(θ), x*⟩)
    # Note: at optimality this is 0, but its derivative w.r.t. θ is not.
    # We write it as μ_eq · β(θ) and separately -μ_eq · ⟨α(θ), x*⟩
    if !isempty(μ_eq)
        s += μ_eq[1] * (β - dot(α, x_star))
    end
    return s
end

# Default: no constraint sensitivity
_constraint_scalar(::ParameterizedOracle, θ, x_star, u, μ_bound, μ_eq, as) = zero(eltype(θ))

"""
    _constraint_pullback(plmo::ParameterizedOracle, θ, x_star, u, μ_bound, μ_eq, as, backend)

Compute ``\\bar{\\theta}_{\\text{constraint}}`` via AD through the constraint scalar function.
"""
function _constraint_pullback(plmo::ParameterizedOracle, θ, x_star, u, μ_bound, μ_eq, as, backend)
    Φ(θ_) = _constraint_scalar(plmo, θ_, x_star, u, μ_bound, μ_eq, as)
    prep = DI.prepare_gradient(Φ, backend, θ)
    return DI.gradient(Φ, prep, backend, θ)
end

# ------------------------------------------------------------------
# rrule: solve(f, ∇f!, lmo, x0, θ; ...) -- existing, upgraded to KKT
# ------------------------------------------------------------------

"""
Implicit differentiation rule for `solve(f, ∇f!, lmo, x0, θ; ...)`.

Uses KKT adjoint solve when the LMO implements `active_set`, which correctly
handles boundary solutions. Falls back to unconstrained Hessian solve when the
active set is empty (interior solution or custom oracle without `active_set`).

See [Implicit Differentiation](@ref) for the full mathematical derivation.
"""
function ChainRulesCore.rrule(::typeof(solve), f, ∇f!, lmo, x0, θ;
                              backend=DEFAULT_BACKEND,
                              hvp_backend=SECOND_ORDER_BACKEND,
                              diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_λ::Real=1e-4,
                              kwargs...)
    x_star, result = solve(f, ∇f!, lmo, x0, θ; backend=backend, kwargs...)

    function solve_pullback(ȳ)
        x̄ = ȳ[1]  # tangent of x

        if x̄ isa ChainRulesCore.AbstractZero
            return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end

        as = active_set(lmo, x_star)

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
            @warn "rrule pullback: CG did not converge (residual=$(cg_result.residual_norm), iters=$(cg_result.iterations)): θ̄ may be inaccurate" maxlog=10
        end
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), θ̄
    end

    return (x_star, result), solve_pullback
end

# rrule for auto-gradient + θ variant (uses joint HVP, no nested AD)
function ChainRulesCore.rrule(::typeof(solve), f, lmo, x0, θ;
                              backend=DEFAULT_BACKEND,
                              hvp_backend=SECOND_ORDER_BACKEND,
                              diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_λ::Real=1e-4,
                              kwargs...)
    x_star, result = solve(f, lmo, x0, θ; backend=backend, kwargs...)

    function solve_pullback(ȳ)
        x̄ = ȳ[1]

        if x̄ isa ChainRulesCore.AbstractZero
            return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end

        as = active_set(lmo, x_star)

        θ̄, cg_result = if isempty(as.bound_indices) && isempty(as.eq_normals)
            _implicit_pullback_hvp(f, x_star, θ, x̄, hvp_backend;
                                  cg_maxiter=diff_cg_maxiter, cg_tol=diff_cg_tol, cg_λ=diff_λ)
        else
            θ̄_obj, _, _, _, cg_res = _kkt_implicit_pullback_hvp(f, x_star, θ, x̄, as, hvp_backend;
                                                                 cg_maxiter=diff_cg_maxiter, cg_tol=diff_cg_tol, cg_λ=diff_λ)
            θ̄_obj, cg_res
        end

        if !cg_result.converged
            @warn "rrule pullback: CG did not converge (residual=$(cg_result.residual_norm), iters=$(cg_result.iterations)): θ̄ may be inaccurate" maxlog=10
        end
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), θ̄
    end

    return (x_star, result), solve_pullback
end

# ------------------------------------------------------------------
# rrule: solve(f, ∇f!, plmo::ParameterizedOracle, x0, θ; ...)
# ------------------------------------------------------------------

"""
Implicit differentiation rule for `solve(f, ∇f!, plmo::ParameterizedOracle, x0, θ; ...)`.

Computes ``\\bar{\\theta} = \\bar{\\theta}_{\\text{obj}} + \\bar{\\theta}_{\\text{constraint}}``
via KKT adjoint solve on the active face.
"""
function ChainRulesCore.rrule(::typeof(solve), f, ∇f!::Function, plmo::ParameterizedOracle, x0, θ;
                              backend=DEFAULT_BACKEND,
                              hvp_backend=SECOND_ORDER_BACKEND,
                              diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_λ::Real=1e-4,
                              kwargs...)
    x_star, result = solve(f, ∇f!, plmo, x0, θ; backend=backend, kwargs...)
    lmo = materialize(plmo, θ)

    function solve_pullback(ȳ)
        x̄ = ȳ[1]

        if x̄ isa ChainRulesCore.AbstractZero
            return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end

        as = active_set(lmo, x_star)

        ∇_x_f_of_θ(θ_) = begin
            T = promote_type(eltype(x_star), eltype(θ_))
            g = similar(x_star, T)
            ∇f!(g, x_star, θ_)
            return g
        end

        θ̄_obj, u, μ_bound, μ_eq, cg_result = _kkt_implicit_pullback(
            f, ∇_x_f_of_θ, x_star, θ, x̄, as, backend, hvp_backend;
            cg_maxiter=diff_cg_maxiter, cg_tol=diff_cg_tol, cg_λ=diff_λ)

        θ̄_con = _constraint_pullback(plmo, θ, x_star, u, μ_bound, μ_eq, as, backend)
        θ̄ = θ̄_obj .+ θ̄_con

        if !cg_result.converged
            @warn "rrule pullback: CG did not converge (residual=$(cg_result.residual_norm), iters=$(cg_result.iterations)): θ̄ may be inaccurate" maxlog=10
        end
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), θ̄
    end

    return (x_star, result), solve_pullback
end

# rrule for auto-gradient + ParameterizedOracle (joint HVP)
function ChainRulesCore.rrule(::typeof(solve), f, plmo::ParameterizedOracle, x0, θ;
                              backend=DEFAULT_BACKEND,
                              hvp_backend=SECOND_ORDER_BACKEND,
                              diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_λ::Real=1e-4,
                              kwargs...)
    x_star, result = solve(f, plmo, x0, θ; backend=backend, kwargs...)
    lmo = materialize(plmo, θ)

    function solve_pullback(ȳ)
        x̄ = ȳ[1]

        if x̄ isa ChainRulesCore.AbstractZero
            return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end

        as = active_set(lmo, x_star)

        θ̄_obj, u, μ_bound, μ_eq, cg_result = _kkt_implicit_pullback_hvp(
            f, x_star, θ, x̄, as, hvp_backend;
            cg_maxiter=diff_cg_maxiter, cg_tol=diff_cg_tol, cg_λ=diff_λ)

        θ̄_con = _constraint_pullback(plmo, θ, x_star, u, μ_bound, μ_eq, as, backend)
        θ̄ = θ̄_obj .+ θ̄_con

        if !cg_result.converged
            @warn "rrule pullback: CG did not converge (residual=$(cg_result.residual_norm), iters=$(cg_result.iterations)): θ̄ may be inaccurate" maxlog=10
        end
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), θ̄
    end

    return (x_star, result), solve_pullback
end
