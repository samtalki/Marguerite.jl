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
    _cg_solve(hvp_fn, rhs; maxiter=50, tol=1e-6, λ=1e-4)

Conjugate gradient solver for

```math
(H + \\lambda I)\\, u = \\text{rhs}
```

where ``H`` is accessed only via Hessian-vector products `hvp_fn(d) -> Hd`.

Tikhonov regularization ``\\lambda`` ensures well-conditioned systems near
singular Hessians (e.g. on boundary of feasible set).
"""
function _cg_solve(hvp_fn, rhs::AbstractVector{T};
                   maxiter::Int=50, tol::Real=1e-6, λ::Real=1e-4) where T
    λ_T = T(λ)
    tol_T = T(tol)
    n = length(rhs)
    u = zeros(T, n)
    r = copy(rhs)     # r = rhs - (H + λI)u = rhs (since u=0)
    p = copy(r)
    r_dot_r = dot(r, r)
    converged = false
    curvature_failure = false
    iters = 0

    # Early return: if rhs is already near-zero, u = 0 is the solution
    if sqrt(r_dot_r) < tol_T
        return u, CGResult(0, sqrt(r_dot_r), true)
    end

    for k in 1:maxiter
        iters = k
        Hp = hvp_fn(p)
        @. Hp += λ_T * p  # (H + λI)p
        pHp = dot(p, Hp)
        if pHp ≤ eps(T)
            @warn "CG encountered near-zero curvature (pHp=$pHp): Hessian may be singular. Consider increasing diff_lambda." maxlog=3
            curvature_failure = true
            break
        end
        α = r_dot_r / pHp
        @. u += α * p
        @. r -= α * Hp
        r_dot_r_new = dot(r, r)
        β = r_dot_r_new / r_dot_r
        r_dot_r = r_dot_r_new
        if sqrt(r_dot_r) < tol_T
            converged = true
            break
        end
        @. p = r + β * p
    end
    residual = sqrt(r_dot_r)
    converged = !curvature_failure && (converged || residual < tol_T)
    if !converged && !curvature_failure
        @warn "CG solve did not converge: residual=$residual after $iters iterations" maxlog=3
    end
    return u, CGResult(iters, residual, converged)
end

"""
    _hessian_cg_solve(f, hvp_backend, x_star, θ, dx; cg_maxiter=50, cg_tol=1e-6, cg_λ=1e-4)

Solve

```math
(\\nabla^2_{xx} f + \\lambda I)\\, u = dx
```

via CG with HVPs.

Shared Hessian-solve step used by [`_kkt_adjoint_solve`](@ref) (fast path when active set is empty)
and the KKT implicit pullback functions.
"""
function _hessian_cg_solve(f, hvp_backend, x_star, θ, dx;
                            cg_maxiter::Int=50, cg_tol::Real=1e-6, cg_λ::Real=1e-4)
    T = eltype(x_star)
    fθ = x_ -> f(x_, θ)
    prep_hvp = DI.prepare_hvp(fθ, hvp_backend, x_star, (dx,))
    hvp_buf = zeros(T, length(x_star))
    hvp_fn = d -> begin
        DI.hvp!(fθ, (hvp_buf,), prep_hvp, hvp_backend, x_star, (d,))
        hvp_buf
    end
    return _cg_solve(hvp_fn, dx isa AbstractVector ? dx : collect(dx);
                     maxiter=cg_maxiter, tol=cg_tol, λ=cg_λ)
end

# ------------------------------------------------------------------
# KKT adjoint solve (constrained implicit differentiation)
# ------------------------------------------------------------------

"""
    _orthogonalize!(a_vecs, a_norm_sqs)

Modified Gram-Schmidt orthogonalization of constraint normals in-place.
Updates both `a_vecs` and `a_norm_sqs` so that sequential projection
in `_null_project!` is correct for non-orthogonal normals.
"""
function _orthogonalize!(a_vecs::Vector{Vector{T}}, a_norm_sqs::Vector{T}) where T
    for j in 2:length(a_vecs)
        for i in 1:(j-1)
            if a_norm_sqs[i] > eps(T)
                coeff = dot(a_vecs[j], a_vecs[i]) / a_norm_sqs[i]
                @. a_vecs[j] -= coeff * a_vecs[i]
            end
        end
        a_norm_sqs[j] = dot(a_vecs[j], a_vecs[j])
    end
end

"""
    _recover_μ_eq(a_vecs, residual)

Recover equality-constraint multipliers by solving the small normal-equations
system ``(A A^\\top) \\mu = A \\, \\text{residual}`` where the rows of ``A`` are
the (original, non-orthogonalized) constraint normals.

For a single constraint this reduces to the familiar inner-product formula.
"""
function _recover_μ_eq(a_vecs::Vector{Vector{T}}, residual::AbstractVector{T}) where T
    k = length(a_vecs)
    k == 0 && return T[]
    G = zeros(T, k, k)
    b = zeros(T, k)
    for j in 1:k
        b[j] = dot(a_vecs[j], residual)
        for i in 1:k
            G[i, j] = dot(a_vecs[i], a_vecs[j])
        end
    end
    return T.(G \ b)
end

"""
    _null_project!(out, w, a_frees, a_norm_sqs)

Project `w` (in free-variable space) onto the null space of pre-computed
equality constraint normals. Writes result into `out` (may alias `w`).

Assumes `a_frees` have been orthogonalized via `_orthogonalize!`
so that sequential subtraction is exact.

```math
P(w) = w - \\sum_j \\frac{a_j^\\top w}{\\|a_j\\|^2}\\, a_j
```
"""
function _null_project!(out::AbstractVector{T}, w::AbstractVector{T},
                        a_frees::Vector{Vector{T}}, a_norm_sqs::Vector{T}) where T
    if out !== w
        copyto!(out, w)
    end
    for (j, (a_free, a_norm_sq)) in enumerate(zip(a_frees, a_norm_sqs))
        if a_norm_sq > eps(T)
            coeff = dot(a_free, out) / a_norm_sq
            @. out -= coeff * a_free
        else
            @warn "null-space projection: constraint normal $j has near-zero free-space norm (||a||²=$a_norm_sq); skipped" maxlog=3
        end
    end
    return out
end

"""
    _correct_bound_multipliers!(μ_bound, μ_eq, as::ActiveConstraints)

Subtract equality-constraint contributions from bound multipliers in place:

```math
\\mu_{\\text{bound},k} \\mathrel{-}= \\sum_j \\mu_{\\text{eq},j} \\, a_j[i_k]
```
"""
function _correct_bound_multipliers!(μ_bound, μ_eq, as::ActiveConstraints)
    for (j, a_full) in enumerate(as.eq_normals)
        for (k, i) in enumerate(as.bound_indices)
            μ_bound[k] -= μ_eq[j] * a_full[i]
        end
    end
    return μ_bound
end

"""
    _kkt_adjoint_solve(f, hvp_backend, x_star, θ, dx, as::ActiveConstraints;
                        cg_maxiter=50, cg_tol=1e-6, cg_λ=1e-4)

Solve the KKT adjoint system at ``x^*`` with active constraints.

The KKT system is:
```math
\\begin{bmatrix} \\nabla^2 f & G^T \\\\ G & 0 \\end{bmatrix}
\\begin{bmatrix} u \\\\ \\mu \\end{bmatrix} =
\\begin{bmatrix} dx \\\\ 0 \\end{bmatrix}
```

Solved via reduced Hessian CG on the null space of ``G`` (active face):
1. Set ``u[\\text{bound}] = 0``, work only in free-variable subspace
2. Project ``dx_{\\text{free}}`` to null(eq\\_normals)
3. CG solve: ``(P H_{\\text{free}} P + \\lambda I) w = P dx_{\\text{free}}``
4. Recover ``\\mu`` from KKT residual
"""
function _kkt_adjoint_solve(f, hvp_backend, x_star, θ, dx, as::ActiveConstraints{AT};
                             cg_maxiter::Int=50, cg_tol::Real=1e-6, cg_λ::Real=1e-4) where AT
    T = promote_type(AT, eltype(x_star))
    n = length(x_star)
    dx_vec = dx isa AbstractVector ? dx : collect(dx)

    # If no active constraints, fall back to unconstrained Hessian solve
    if isempty(as.bound_indices) && isempty(as.eq_normals)
        u, cg_result = _hessian_cg_solve(f, hvp_backend, x_star, θ, dx_vec;
                                          cg_maxiter, cg_tol, cg_λ)
        return u, T[], T[], cg_result
    end

    free = as.free_indices
    bound = as.bound_indices
    n_free = length(free)

    # If no free variables, u = 0; recover multipliers from dx directly
    if n_free == 0
        # Stationarity: dx = ∑_k μ_bound_k · e_{bound_k} + ∑_j μ_eq_j · a_j
        # Recover μ_eq via normal-equations solve (handles non-orthogonal normals)
        μ_eq = _recover_μ_eq(as.eq_normals, dx_vec)
        # μ_bound = dx[bound] - ∑_j μ_eq_j · a_j[bound]
        μ_bound = T[dx_vec[i] for i in bound]
        _correct_bound_multipliers!(μ_bound, μ_eq, as)
        return zeros(T, n), μ_bound, μ_eq, CGResult(0, zero(T), true)
    end

    # Pre-compute a_free vectors and their squared norms (reused by _null_project!)
    a_frees_orig = [T.(a_full[free]) for a_full in as.eq_normals]
    a_frees = [copy(a) for a in a_frees_orig]
    a_norm_sqs = T[dot(a, a) for a in a_frees]
    _orthogonalize!(a_frees, a_norm_sqs)

    # Prepare HVP on f(·, θ)
    fθ = x_ -> f(x_, θ)
    prep_hvp = DI.prepare_hvp(fθ, hvp_backend, x_star, (dx_vec,))

    # Pre-allocate buffers for the CG loop
    w_full = zeros(T, n)
    hvp_buf = zeros(T, n)
    Hw_buf = zeros(T, n_free)
    proj_buf = zeros(T, n_free)

    # Reduced HVP: expand w_free to full space → HVP → extract free → null-project
    # NOTE: returns proj_buf (shared mutable buffer) for zero-allocation.
    # Safe because _cg_solve consumes Hp = hvp_fn(p) before the next call.
    function reduced_hvp(w_free)
        fill!(w_full, zero(T))
        @inbounds for (j, idx) in enumerate(free)
            w_full[idx] = w_free[j]
        end
        DI.hvp!(fθ, (hvp_buf,), prep_hvp, hvp_backend, x_star, (w_full,))
        @inbounds for (j, idx) in enumerate(free)
            Hw_buf[j] = hvp_buf[idx]
        end
        return _null_project!(proj_buf, Hw_buf, a_frees, a_norm_sqs)
    end

    # RHS: project dx_free onto null(eq_normals)
    dx_free = @view(dx_vec[free])
    rhs = _null_project!(similar(dx_free, T, length(free)), dx_free, a_frees, a_norm_sqs)

    # CG solve in reduced space
    u_free, cg_result = _cg_solve(reduced_hvp, rhs; maxiter=cg_maxiter, tol=cg_tol, λ=cg_λ)

    # Null-project the CG result for consistency
    _null_project!(u_free, u_free, a_frees, a_norm_sqs)

    # Assemble full u
    u = zeros(T, n)
    @inbounds for (j, idx) in enumerate(free)
        u[idx] = u_free[j]
    end

    # Compute Hu for multiplier recovery; reuse w_full as residual buffer
    DI.hvp!(fθ, (hvp_buf,), prep_hvp, hvp_backend, x_star, (u,))
    @. w_full = dx_vec - hvp_buf

    # μ_eq: solve normal equations using original (non-orthogonalized) normals
    # (must recover μ_eq first, since μ_bound correction depends on it)
    residual_free = @view(w_full[free])
    μ_eq = _recover_μ_eq(a_frees_orig, residual_free)

    # μ_bound: residual at bound index, minus equality constraint contributions
    # Stationarity: residual[i] = μ_bound_k + ∑_j μ_eq_j · a_j[i]
    μ_bound = T[w_full[i] for i in bound]
    _correct_bound_multipliers!(μ_bound, μ_eq, as)

    return u, μ_bound, μ_eq, cg_result
end

# ------------------------------------------------------------------
# Cross-derivative helpers (shared by unconstrained and KKT pullbacks)
# ------------------------------------------------------------------

"""
    _make_∇ₓf_of_θ(∇f!, x_star)

Build the map ``\\theta \\mapsto \\nabla_x f(x^*, \\theta)`` from a mutating gradient
`∇f!(g, x, θ)` and a fixed solution `x_star`.

The returned closure allocates a type-promoted buffer so that forward-mode AD
through ``\\theta`` propagates correctly. It is consumed by
[`_cross_derivative_manual`](@ref) to compute the cross-derivative
``(\\partial \\nabla_x f / \\partial \\theta)^\\top u``.
"""
function _make_∇ₓf_of_θ(∇f!, x_star)
    return θ_ -> begin
        T = promote_type(eltype(x_star), eltype(θ_))
        g = similar(x_star, T)
        ∇f!(g, x_star, θ_)
        return g
    end
end

"""
    _cross_derivative_manual(∇ₓf_of_θ, u, θ, backend)

Compute ``d\\theta = -(\\partial(\\nabla_x f)/\\partial\\theta)^T u`` via AD
through the scalar ``\\theta \\mapsto \\langle \\nabla_x f(\\theta), u \\rangle``.
"""
function _cross_derivative_manual(∇ₓf_of_θ, u, θ, backend)
    ∇f_dot_u = θ_ -> dot(∇ₓf_of_θ(θ_), u)
    prep_g = DI.prepare_gradient(∇f_dot_u, backend, θ)
    return -DI.gradient(∇f_dot_u, prep_g, backend, θ)
end

"""
    _cross_derivative_hvp(f, x_star, θ, u, hvp_backend)

Compute ``d\\theta = -\\nabla^2_{\\theta x} f \\cdot u`` via a joint HVP on
``g(z) = f(z_{1:n}, z_{n+1:\\text{end}})`` with ``z = [x; \\theta]``.
"""
function _cross_derivative_hvp(f, x_star, θ, u, hvp_backend)
    n = length(x_star)
    m = length(θ)
    # @view avoids O(n+m) allocation per HVP call; requires the AD backend to
    # support SubArray differentiation (ForwardDiff and Mooncake do).
    g = z -> f(@view(z[1:n]), @view(z[n+1:end]))
    z = vcat(x_star, θ)
    v = vcat(u, zeros(eltype(u), m))
    prep_cross = DI.prepare_hvp(g, hvp_backend, z, (v,))
    cross_hvp = DI.hvp(g, prep_cross, hvp_backend, z, (v,))[1]
    dθ = cross_hvp[n+1:end]
    @. dθ = -dθ
    return dθ
end

# ------------------------------------------------------------------
# KKT-based implicit pullbacks (with active set)
# ------------------------------------------------------------------

"""
    _kkt_implicit_pullback(f, ∇ₓf_of_θ, x_star, θ, dx, as, backend, hvp_backend; kwargs...)

KKT adjoint pullback with manual gradient. Solves the KKT system on the active
face, then computes ``d\\theta_{\\text{obj}} = -(\\partial(\\nabla_x f)/\\partial\\theta)^T u``.
"""
function _kkt_implicit_pullback(f, ∇ₓf_of_θ, x_star, θ, dx, as::ActiveConstraints,
                                 backend, hvp_backend;
                                 cg_maxiter::Int=50, cg_tol::Real=1e-6, cg_λ::Real=1e-4)
    u, μ_bound, μ_eq, cg_result = _kkt_adjoint_solve(f, hvp_backend, x_star, θ, dx, as;
                                                       cg_maxiter, cg_tol, cg_λ)
    dθ_obj = _cross_derivative_manual(∇ₓf_of_θ, u, θ, backend)
    return dθ_obj, u, μ_bound, μ_eq, cg_result
end

"""
    _kkt_implicit_pullback_hvp(f, x_star, θ, dx, as, hvp_backend; kwargs...)

KKT adjoint pullback with auto-gradient (joint HVP, no nested AD).
"""
function _kkt_implicit_pullback_hvp(f, x_star, θ, dx, as::ActiveConstraints, hvp_backend;
                                     cg_maxiter::Int=50, cg_tol::Real=1e-6, cg_λ::Real=1e-4)
    u, μ_bound, μ_eq, cg_result = _kkt_adjoint_solve(f, hvp_backend, x_star, θ, dx, as;
                                                       cg_maxiter, cg_tol, cg_λ)
    dθ_obj = _cross_derivative_hvp(f, x_star, θ, u, hvp_backend)
    return dθ_obj, u, μ_bound, μ_eq, cg_result
end

# ------------------------------------------------------------------
# Constraint pullback (constraint sensitivity contribution to dθ)
# ------------------------------------------------------------------

"""
    _constraint_scalar(plmo::ParametricOracle, θ, x_star, μ_bound, μ_eq, as)

Compute the scalar function ``\\Phi(\\theta)`` whose gradient gives the
constraint sensitivity contribution to ``d\\theta``.

For simple RHS-parametric constraints, ``\\Phi(\\theta) = \\mu^T h(\\theta)`` where
``h(\\theta)`` are the active constraint RHS values. For constraints with
``\\theta``-dependent normals (e.g. [`ParametricWeightedSimplex`](@ref)), the scalar
also captures normal-variation sensitivity via AD through the full constraint expression.
"""
function _constraint_scalar end

function _constraint_scalar(plmo::ParametricBox, θ, x_star, μ_bound, μ_eq, as::ActiveConstraints{T}) where T
    # For box constraints: active constraints are x_i = lb_i or x_i = ub_i
    # Φ(θ) = ∑_i μ_i · bound_value_i(θ)
    lb = plmo.lb_fn(θ)
    ub = plmo.ub_fn(θ)
    s = zero(eltype(θ))
    for (k, i) in enumerate(as.bound_indices)
        if as.bound_is_lower[k]
            s += μ_bound[k] * lb[i]
        else
            s += μ_bound[k] * ub[i]
        end
    end
    return s
end

function _constraint_scalar(plmo::ParametricSimplex, θ, x_star, μ_bound, μ_eq, as::ActiveConstraints)
    # Φ(θ) = μ_budget · r(θ); bound constraints x_i ≥ 0 don't depend on θ
    r = plmo.r_fn(θ)
    if !isempty(μ_eq)
        return μ_eq[1] * r
    end
    return zero(eltype(θ))
end

function _constraint_scalar(plmo::ParametricWeightedSimplex, θ, x_star, μ_bound, μ_eq, as::ActiveConstraints{T}) where T
    # Φ(θ) = ∑_{bound} μ_i · lb_i(θ) + μ_eq · (β(θ) - ⟨α(θ), x*⟩)
    # The budget term is zero at optimality, but its θ-derivative is not;
    # AD through this scalar captures both RHS and normal-variation sensitivity.
    lb = plmo.lb_fn(θ)
    α = plmo.α_fn(θ)
    β = plmo.β_fn(θ)
    s = zero(eltype(θ))
    for (k, i) in enumerate(as.bound_indices)
        s += μ_bound[k] * lb[i]
    end
    if !isempty(μ_eq)
        s += μ_eq[1] * (β - dot(α, x_star))
    end
    return s
end

# Default: error for unimplemented ParametricOracle subtypes
function _constraint_scalar(plmo::ParametricOracle, θ, x_star, μ_bound, μ_eq, as)
    error("_constraint_scalar not implemented for $(typeof(plmo)). Implement Marguerite._constraint_scalar(...) to enable constraint sensitivity.")
end

"""
    _constraint_pullback(plmo::ParametricOracle, θ, x_star, μ_bound, μ_eq, as, backend)

Compute ``d\\theta_{\\text{constraint}}`` via AD through the constraint scalar function.
"""
function _constraint_pullback(plmo::ParametricOracle, θ, x_star, μ_bound, μ_eq, as, backend)
    Φ(θ_) = _constraint_scalar(plmo, θ_, x_star, μ_bound, μ_eq, as)
    prep = DI.prepare_gradient(Φ, backend, θ)
    return DI.gradient(Φ, prep, backend, θ)
end

# ------------------------------------------------------------------
# rrule: solve(f, lmo, x0, θ; grad=..., ...)
# ------------------------------------------------------------------

"""
Implicit differentiation rule for `solve(f, lmo, x0, θ; ...)`.

Handles all `lmo` types (plain functions, `AbstractOracle`, `ParametricOracle`)
and both manual and auto gradient via the `grad=` keyword.

Uses KKT adjoint solve via [`_kkt_implicit_pullback`](@ref), which correctly
handles both boundary solutions (active constraints) and interior solutions
(empty active set fast path).

See [Implicit Differentiation](@ref) for the full mathematical derivation.
"""
function ChainRulesCore.rrule(::typeof(solve), f, lmo, x0, θ;
                              grad=nothing,
                              backend=DEFAULT_BACKEND,
                              hvp_backend=SECOND_ORDER_BACKEND,
                              diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_lambda::Real=1e-4,
                              tol::Real=1e-4,
                              kwargs...)
    x_star, result = solve(f, lmo, x0, θ; grad=grad, backend=backend, tol=tol, kwargs...)
    if lmo isa ParametricOracle
        oracle = materialize(lmo, θ)
    else
        oracle = lmo isa AbstractOracle ? lmo : FunctionOracle(lmo)
    end

    function solve_pullback(dy)
        dx = dy isa Tuple ? dy[1] : dy.x

        if dx isa ChainRulesCore.AbstractZero
            return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end

        as = active_set(oracle, x_star; tol=min(tol, 1e-6))

        if grad !== nothing
            ∇ₓf_of_θ = _make_∇ₓf_of_θ(grad, x_star)
            dθ_obj, u, μ_bound, μ_eq, cg_result = _kkt_implicit_pullback(
                f, ∇ₓf_of_θ, x_star, θ, dx, as, backend, hvp_backend;
                cg_maxiter=diff_cg_maxiter, cg_tol=diff_cg_tol, cg_λ=diff_lambda)
        else
            dθ_obj, u, μ_bound, μ_eq, cg_result = _kkt_implicit_pullback_hvp(
                f, x_star, θ, dx, as, hvp_backend;
                cg_maxiter=diff_cg_maxiter, cg_tol=diff_cg_tol, cg_λ=diff_lambda)
        end

        if lmo isa ParametricOracle
            dθ_con = _constraint_pullback(lmo, θ, x_star, μ_bound, μ_eq, as, backend)
            dθ = dθ_obj .+ dθ_con
        else
            dθ = dθ_obj
        end

        if !cg_result.converged
            @warn "rrule pullback: CG did not converge (residual=$(cg_result.residual_norm), iters=$(cg_result.iterations)): dθ may be inaccurate" maxlog=10
        end
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), dθ
    end

    return SolveResult(x_star, result), solve_pullback
end
