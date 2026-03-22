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
        if pHp ≤ eps(T) * max(one(T), r_dot_r)
            @warn "CG encountered near-zero curvature (pHp=$pHp): Hessian may be singular. Consider increasing diff_lambda." maxlog=3
            curvature_failure = true
            break
        end
        α = r_dot_r / pHp
        @. u += α * p
        @. r -= α * Hp
        r_dot_r_new = dot(r, r)
        if r_dot_r < tol_T * tol_T
            r_dot_r = r_dot_r_new
            converged = true
            break
        end
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
    max_norm_sq = isempty(a_norm_sqs) ? one(T) : maximum(a_norm_sqs)
    thr = eps(T) * max_norm_sq
    for j in 2:length(a_vecs)
        for i in 1:(j-1)
            if a_norm_sqs[i] > thr
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
function _recover_μ_eq(a_vecs::AbstractVector{<:AbstractVector{T}}, residual::AbstractVector{T}) where T
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
    return T.(pinv(G, rtol=sqrt(eps(T))) * b)
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
    max_norm_sq = isempty(a_norm_sqs) ? one(T) : maximum(a_norm_sqs)
    thr = eps(T) * max_norm_sq
    for (j, (a_free, a_norm_sq)) in enumerate(zip(a_frees, a_norm_sqs))
        if a_norm_sq > thr
            coeff = dot(a_free, out) / a_norm_sq
            @. out -= coeff * a_free
        else
            @warn "null-space projection: constraint normal $j has near-zero free-space norm (||a||²=$a_norm_sq); skipped" maxlog=3
        end
    end
    return out
end

@inline function _spectraplex_trace_zero_dim(rank::Int)
    return rank == 0 ? 0 : rank * (rank + 1) ÷ 2 - 1
end

@inline function _spectraplex_tangent_dim(rank::Int, nullity::Int)
    return _spectraplex_trace_zero_dim(rank) + rank * nullity
end

function _spectraplex_pack_trace_zero!(out::AbstractVector{T}, M::AbstractMatrix{T}) where T
    k = size(M, 1)
    p = 1
    if k == 0
        return out
    end
    diag_ref = M[k, k]
    @inbounds for i in 1:(k - 1)
        out[p] = M[i, i] - diag_ref
        p += 1
    end
    @inbounds for j in 2:k
        for i in 1:(j - 1)
            out[p] = M[i, j] + M[j, i]
            p += 1
        end
    end
    return out
end

function _spectraplex_pack_mixed!(out::AbstractVector{T}, B::AbstractMatrix{T}, offset::Int) where T
    p = offset
    @inbounds for j in 1:size(B, 2)
        for i in 1:size(B, 1)
            out[p] = B[i, j]
            p += 1
        end
    end
    return out
end

function _spectraplex_unpack_trace_zero!(S::AbstractMatrix{T}, z::AbstractVector{T}) where T
    k = size(S, 1)
    fill!(S, zero(T))
    if k == 0
        return S
    end

    p = 1
    diag_sum = zero(T)
    @inbounds for i in 1:(k - 1)
        zi = z[p]
        S[i, i] = zi
        diag_sum += zi
        p += 1
    end
    S[k, k] = -diag_sum

    @inbounds for j in 2:k
        for i in 1:(j - 1)
            zij = z[p]
            S[i, j] = zij
            S[j, i] = zij
            p += 1
        end
    end
    return S
end

function _spectraplex_unpack_mixed!(B::AbstractMatrix{T}, z::AbstractVector{T}, offset::Int) where T
    fill!(B, zero(T))
    p = offset
    @inbounds for j in 1:size(B, 2)
        for i in 1:size(B, 1)
            B[i, j] = z[p]
            p += 1
        end
    end
    return B
end

function _spectraplex_expand!(out::AbstractVector{T}, z::AbstractVector{T},
                              U::AbstractMatrix{T}, V_perp::AbstractMatrix{T},
                              face_buf::AbstractMatrix{T}, mixed_buf::AbstractMatrix{T},
                              tmp_face_buf::AbstractMatrix{T}, tmp_mixed_buf::AbstractMatrix{T},
                              full_buf::AbstractMatrix{T}, cross_buf::AbstractMatrix{T}) where T
    face_dim = _spectraplex_trace_zero_dim(size(U, 2))

    _spectraplex_unpack_trace_zero!(face_buf, @view(z[1:face_dim]))
    fill!(full_buf, zero(T))

    if !isempty(U)
        mul!(tmp_face_buf, U, face_buf)
        mul!(full_buf, tmp_face_buf, transpose(U))
    end

    if !isempty(U) && !isempty(V_perp)
        _spectraplex_unpack_mixed!(mixed_buf, z, face_dim + 1)
        mul!(tmp_mixed_buf, U, mixed_buf)
        mul!(cross_buf, tmp_mixed_buf, transpose(V_perp))
        @inbounds for j in 1:size(full_buf, 2)
            for i in 1:size(full_buf, 1)
                full_buf[i, j] += cross_buf[i, j] + cross_buf[j, i]
            end
        end
    end

    n = size(full_buf, 1)
    @inbounds for j in 1:n
        off = (j - 1) * n
        for i in 1:n
            out[off + i] = full_buf[i, j]
        end
    end
    return out
end

function _spectraplex_compress!(out::AbstractVector{T}, x::AbstractVector{T},
                                U::AbstractMatrix{T}, V_perp::AbstractMatrix{T},
                                tmp_face_buf::AbstractMatrix{T}, tmp_null_buf::AbstractMatrix{T},
                                face_buf::AbstractMatrix{T}, mixed_buf::AbstractMatrix{T},
                                full_buf::AbstractMatrix{T}) where T
    n = size(full_buf, 1)
    X = reshape(x, n, n)
    @inbounds for j in 1:n
        for i in 1:n
            full_buf[i, j] = (X[i, j] + X[j, i]) / 2
        end
    end

    face_dim = _spectraplex_trace_zero_dim(size(U, 2))
    if !isempty(U)
        mul!(tmp_face_buf, full_buf, U)
        mul!(face_buf, transpose(U), tmp_face_buf)
        _spectraplex_pack_trace_zero!(@view(out[1:face_dim]), face_buf)
    end

    if !isempty(U) && !isempty(V_perp)
        mul!(tmp_null_buf, full_buf, V_perp)
        mul!(mixed_buf, transpose(U), tmp_null_buf)
        _spectraplex_pack_mixed!(out, mixed_buf, face_dim + 1)
    end
    return out
end

function _spectraplex_add_mixed_curvature!(out::AbstractVector{T}, z::AbstractVector{T},
                                           G_uu::AbstractMatrix{T}, G_vv::AbstractMatrix{T},
                                           mixed_buf::AbstractMatrix{T},
                                           mixed_curv_buf::AbstractMatrix{T}) where T
    if isempty(mixed_buf)
        return out
    end

    face_dim = _spectraplex_trace_zero_dim(size(G_uu, 1))
    _spectraplex_unpack_mixed!(mixed_buf, z, face_dim + 1)
    fill!(mixed_curv_buf, zero(T))

    # The mixed active/null block sees the linearized PSD curvature term
    # B * G_vv - G_uu * B at rank-deficient optima.
    mul!(mixed_curv_buf, mixed_buf, G_vv)            # mixed_curv_buf = B * G_vv
    mul!(mixed_curv_buf, G_uu, mixed_buf, -one(T), one(T))  # mixed_curv_buf -= G_uu * B

    p = face_dim + 1
    @inbounds for j in 1:size(mixed_curv_buf, 2)
        for i in 1:size(mixed_curv_buf, 1)
            out[p] += mixed_curv_buf[i, j]
            p += 1
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
    _objective_gradient(f, grad, x_star, θ, backend)

Evaluate ``\\nabla_x f(x^*, \\theta)`` using either the user's manual gradient or
the configured AD backend.
"""
function _objective_gradient(f, grad, x_star, θ, backend)
    if grad !== nothing
        T = promote_type(eltype(x_star), eltype(θ))
        g = similar(x_star, T)
        grad(g, x_star, θ)
        return g
    end
    fθ = x_ -> f(x_, θ)
    prep = DI.prepare_gradient(fθ, backend, x_star)
    return DI.gradient(fθ, prep, backend, x_star)
end

"""
    _recover_face_multipliers(residual, as::ActiveConstraints)

Recover multipliers for the active face from a residual of the form
``G^\\top \\lambda = \\text{residual}``, where `G` contains the active face normals.

This is used both for adjoint multipliers (with `residual = dx - Hu`) and for
primal multipliers (with `residual = -∇f(x*, θ)`).
"""
function _recover_face_multipliers(residual::AbstractVector, as::ActiveConstraints{AT}) where AT
    T = promote_type(AT, eltype(residual))

    if isempty(as.bound_indices) && isempty(as.eq_normals)
        return T[], T[]
    end

    residual_vec = T.(residual)
    if isempty(as.eq_normals)
        λ_eq = T[]
    elseif isempty(as.free_indices)
        λ_eq = _recover_μ_eq([T.(a_full) for a_full in as.eq_normals], residual_vec)
    else
        a_frees = [T.(a_full[as.free_indices]) for a_full in as.eq_normals]
        λ_eq = _recover_μ_eq(a_frees, @view(residual_vec[as.free_indices]))
    end

    λ_bound = T[residual_vec[i] for i in as.bound_indices]
    _correct_bound_multipliers!(λ_bound, λ_eq, as)
    return λ_bound, λ_eq
end

function _recover_face_multipliers(residual::AbstractVector,
                                   as::ActiveConstraints{AT, <:SpectraplexEqNormals}) where AT
    T = promote_type(AT, eltype(residual))
    return T[], T[]
end

"""
    _primal_face_multipliers(f, grad, x_star, θ, as, backend)

Recover the primal active-face multipliers from stationarity

```math
\\nabla_x f(x^*; \\theta) + G(\\theta)^T \\lambda = 0.
```

When the active normals depend on ``\\theta``, these multipliers contribute to the
implicit gradient through the term ``-\\lambda^T (\\partial_\\theta G) u``.
"""
function _primal_face_multipliers(f, grad, x_star, θ, as::ActiveConstraints, backend)
    ∇ₓf = _objective_gradient(f, grad, x_star, θ, backend)
    residual = -∇ₓf
    return _recover_face_multipliers(residual, as)
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
                             cg_maxiter::Int=50, cg_tol::Real=1e-6, cg_λ::Real=1e-4,
                             grad=nothing, backend=DEFAULT_BACKEND) where AT
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

function _kkt_adjoint_solve(f, hvp_backend, x_star, θ, dx,
                            as::ActiveConstraints{AT, <:SpectraplexEqNormals};
                            cg_maxiter::Int=50, cg_tol::Real=1e-6, cg_λ::Real=1e-4,
                            grad=nothing, backend=DEFAULT_BACKEND) where AT
    T = promote_type(AT, eltype(x_star))
    m = length(x_star)
    dx_vec = dx isa AbstractVector ? dx : collect(dx)
    eq = as.eq_normals
    U = T.(eq.U)
    V_perp = T.(eq.V_perp)
    rank = size(U, 2)
    nullity = size(V_perp, 2)
    d = _spectraplex_tangent_dim(rank, nullity)

    if d == 0
        return zeros(T, m), T[], T[], CGResult(0, zero(T), true)
    end

    fθ = x_ -> f(x_, θ)
    prep_hvp = DI.prepare_hvp(fθ, hvp_backend, x_star, (dx_vec,))

    rhs = zeros(T, d)
    Hw = zeros(T, d)
    reg = zeros(T, d)
    w_full = zeros(T, m)
    hvp_buf = zeros(T, m)
    tmp_face_buf = zeros(T, size(eq.U, 1), rank)
    tmp_null_buf = zeros(T, size(eq.U, 1), nullity)
    face_buf = zeros(T, rank, rank)
    mixed_buf = zeros(T, rank, nullity)
    mixed_curv_buf = zeros(T, rank, nullity)
    full_buf = zeros(T, size(eq.U, 1), size(eq.U, 1))
    cross_buf = zeros(T, size(eq.U, 1), size(eq.U, 1))
    G = reshape(_objective_gradient(f, grad, x_star, θ, backend), size(eq.U, 1), size(eq.U, 1))
    G_sym = Symmetric((G .+ G') ./ T(2))
    G_uu = Matrix{T}(transpose(U) * G_sym * U)
    G_vv = Matrix{T}(transpose(V_perp) * G_sym * V_perp)
    λ_T = T(cg_λ)

    _spectraplex_compress!(rhs, dx_vec, U, V_perp,
                           tmp_face_buf, tmp_null_buf,
                           face_buf, mixed_buf, full_buf)

    function reduced_hvp(z)
        _spectraplex_expand!(w_full, z, U, V_perp,
                             face_buf, mixed_buf,
                             tmp_face_buf, tmp_null_buf,
                             full_buf, cross_buf)
        DI.hvp!(fθ, (hvp_buf,), prep_hvp, hvp_backend, x_star, (w_full,))
        _spectraplex_compress!(Hw, hvp_buf, U, V_perp,
                               tmp_face_buf, tmp_null_buf,
                               face_buf, mixed_buf, full_buf)
        if !iszero(λ_T)
            _spectraplex_compress!(reg, w_full, U, V_perp,
                                   tmp_face_buf, tmp_null_buf,
                                   face_buf, mixed_buf, full_buf)
            @. Hw += λ_T * reg
        end
        _spectraplex_add_mixed_curvature!(Hw, z, G_uu, G_vv, mixed_buf, mixed_curv_buf)
        return Hw
    end

    z, cg_result = _cg_solve(reduced_hvp, rhs; maxiter=cg_maxiter, tol=cg_tol, λ=zero(T))
    u = zeros(T, m)
    _spectraplex_expand!(u, z, U, V_perp,
                         face_buf, mixed_buf,
                         tmp_face_buf, tmp_null_buf,
                         full_buf, cross_buf)
    return u, T[], T[], cg_result
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
# Constraint pullback (constraint sensitivity contribution to dθ)
# ------------------------------------------------------------------

"""
    _constraint_scalar(plmo::ParametricOracle, θ, x_star, u, μ_bound, μ_eq, λ_bound, λ_eq, as)

Compute the scalar function ``\\Phi(\\theta)`` whose gradient gives the
constraint sensitivity contribution to ``d\\theta``.

For simple RHS-parametric constraints, ``\\Phi(\\theta) = \\mu^T h(\\theta)`` where
``h(\\theta)`` are the active constraint RHS values. For constraints with
``\\theta``-dependent normals, the scalar also includes the primal-multiplier term
``-\\lambda^T G(\\theta) u`` required by the full linear-face KKT pullback.
"""
function _constraint_scalar end

function _constraint_scalar(plmo::ParametricBox, θ, x_star, u, μ_bound, μ_eq, λ_bound, λ_eq,
                            as::ActiveConstraints{T}) where T
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

function _constraint_scalar(plmo::ParametricSimplex, θ, x_star, u, μ_bound, μ_eq, λ_bound, λ_eq,
                            as::ActiveConstraints)
    # bound constraints x_i ≥ 0 don't depend on θ
    r = plmo.r_fn(θ)
    if !isempty(μ_eq)
        return μ_eq[1] * r
    end
    return zero(eltype(θ))
end

function _constraint_scalar(plmo::ParametricWeightedSimplex, θ, x_star, u, μ_bound, μ_eq, λ_bound, λ_eq,
                            as::ActiveConstraints{T}) where T
    # Φ(θ) = ∑_{bound} μ_i · lb_i(θ) + μ_eq · (β(θ) - ⟨α(θ), x*⟩) - λ_eq · ⟨α(θ), u⟩
    # The final term is the missing normal-variation contribution from the
    # stationarity equation when α depends on θ.
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
    if !isempty(λ_eq)
        s -= λ_eq[1] * dot(α, u)
    end
    return s
end

# Default: error for unimplemented ParametricOracle subtypes
function _constraint_scalar(plmo::ParametricOracle, θ, x_star, u, μ_bound, μ_eq, λ_bound, λ_eq, as)
    error("_constraint_scalar not implemented for $(typeof(plmo)). Implement Marguerite._constraint_scalar(...) to enable constraint sensitivity.")
end

"""
    _constraint_pullback(plmo::ParametricOracle, θ, x_star, u, μ_bound, μ_eq, λ_bound, λ_eq, as, backend)

Compute ``d\\theta_{\\text{constraint}}`` via AD through the constraint scalar function.
"""
function _constraint_pullback(plmo::ParametricOracle, θ, x_star, u, μ_bound, μ_eq, λ_bound, λ_eq, as, backend)
    Φ(θ_) = _constraint_scalar(plmo, θ_, x_star, u, μ_bound, μ_eq, λ_bound, λ_eq, as)
    prep = DI.prepare_gradient(Φ, backend, θ)
    return DI.gradient(Φ, prep, backend, θ)
end

# ------------------------------------------------------------------
# Cached pullback state (rrule amortization across multiple pullbacks)
# ------------------------------------------------------------------

"""
    _PullbackState

Pre-computed state for the rrule pullback closure. Built once in the rrule body
and reused across all pullback calls (e.g. when computing a full Jacobian via
n pullback calls). Caches: active set, HVP preparation, orthogonalized constraint
normals, and CG buffers.
"""
struct _PullbackState{T, FT, HB, P, AS<:ActiveConstraints}
    fθ::FT
    x_star::Vector{T}
    hvp_backend::HB
    prep_hvp::P
    as::AS
    free::Vector{Int}
    bound::Vector{Int}
    n_free::Int
    a_frees::Vector{Vector{T}}
    a_frees_orig::Vector{Vector{T}}
    a_norm_sqs::Vector{T}
    w_full::Vector{T}
    hvp_buf::Vector{T}
    Hw_buf::Vector{T}
    proj_buf::Vector{T}
    rhs_buf::Vector{T}
    # Cross-derivative buffers (length n+m, reused by jacobian and rrule)
    cross_z::Vector{T}
    cross_v::Vector{T}
    cross_hvp::Vector{T}
end

struct _SpectraplexPullbackState{T, FT, HB, P, AS<:ActiveConstraints}
    fθ::FT
    x_star::Vector{T}
    hvp_backend::HB
    prep_hvp::P
    as::AS
    U::Matrix{T}
    V_perp::Matrix{T}
    G_uu::Matrix{T}
    G_vv::Matrix{T}
    reduced_dim::Int
    w_full::Vector{T}
    hvp_buf::Vector{T}
    rhs_buf::Vector{T}
    Hw_buf::Vector{T}
    reg_buf::Vector{T}
    tmp_face_buf::Matrix{T}
    tmp_null_buf::Matrix{T}
    face_buf::Matrix{T}
    mixed_buf::Matrix{T}
    mixed_curv_buf::Matrix{T}
    full_buf::Matrix{T}
    cross_buf::Matrix{T}
    cross_z::Vector{T}
    cross_v::Vector{T}
    cross_hvp::Vector{T}
end

@inline function _interior_active_constraints(x::AbstractVector{T}) where T
    n = length(x)
    return ActiveConstraints{T}(Int[], T[], BitVector(), collect(1:n), Vector{T}[], T[])
end

"""
    _has_active_set(oracle)

Trait: returns `true` if the oracle type has a specialized `active_set` method
beyond the generic fallback. Built-in oracles are enumerated; user-defined
oracle types should either override this trait or define `active_set` with a
typed first argument (which is detected automatically via method introspection).
"""
_has_active_set(::Simplex) = true
_has_active_set(::Knapsack) = true
_has_active_set(::MaskedKnapsack) = true
_has_active_set(::Box) = true
_has_active_set(::ScalarBox) = true
_has_active_set(::WeightedSimplex) = true
_has_active_set(::Spectraplex) = true
# Fallback: use method introspection to detect user-defined active_set methods
function _has_active_set(oracle)
    m = which(active_set, Tuple{typeof(oracle), Vector{Float64}})
    sig = Base.unwrap_unionall(m.sig)
    return sig.parameters[2] !== Any
end

function _active_set_for_diff(oracle, x::AbstractVector{T};
                              tol::Real=1e-8,
                              assume_interior::Bool=false,
                              caller::AbstractString="differentiate") where T
    if _has_active_set(oracle)
        return active_set(oracle, x; tol=tol)
    end

    oracle_type = typeof(oracle)
    if !assume_interior
        throw(ArgumentError(
            "$caller requires `Marguerite.active_set` support for $oracle_type to differentiate constrained solutions. " *
            "Define `Marguerite.active_set(::$(oracle_type), x; tol=...)` or pass `assume_interior=true` to use the interior approximation."
        ))
    end

    @warn "$caller: no active_set specialization for $oracle_type; assuming interior solution because assume_interior=true. Differentiated results may be inaccurate on boundary solutions." maxlog=3
    return _interior_active_constraints(x)
end

"""
    _build_pullback_state(f, hvp_backend, x_star, θ, oracle, tol)

 Build `_PullbackState` once in the rrule body. Performs active set
identification, HVP preparation, constraint orthogonalization, and buffer
allocation — all invariant across pullback calls.
"""
function _build_pullback_state(f, hvp_backend, x_star, θ, oracle, tol;
                                assume_interior::Bool=false,
                                grad=nothing,
                                backend=DEFAULT_BACKEND)
    as = _active_set_for_diff(oracle, x_star;
                               tol=min(tol, 1e-6),
                               assume_interior=assume_interior,
                               caller="rrule(solve)")
    T = promote_type(eltype(as.bound_values), eltype(x_star))
    n = length(x_star)
    fθ = x_ -> f(x_, θ)
    free = as.free_indices
    bound = as.bound_indices
    n_free = length(free)

    # Precompute orthogonalized constraint normals in free-variable space
    a_frees_orig = [T.(a[free]) for a in as.eq_normals]
    a_frees = [copy(a) for a in a_frees_orig]
    a_norm_sqs = T[dot(a, a) for a in a_frees]
    _orthogonalize!(a_frees, a_norm_sqs)

    # One-time HVP preparation (the expensive AD tape build)
    dx_dummy = zeros(T, n)
    prep_hvp = DI.prepare_hvp(fθ, hvp_backend, x_star, (dx_dummy,))

    # Pre-allocate buffers
    w_full = zeros(T, n)
    hvp_buf = zeros(T, n)
    Hw_buf = zeros(T, n_free)
    proj_buf = zeros(T, n_free)
    rhs_buf = zeros(T, n_free)
    m = length(θ)
    cross_z = zeros(T, n + m)
    cross_v = zeros(T, n + m)
    cross_hvp = zeros(T, n + m)

    return _PullbackState(fθ, x_star, hvp_backend, prep_hvp, as,
                          free, bound, n_free, a_frees, a_frees_orig, a_norm_sqs,
                          w_full, hvp_buf, Hw_buf, proj_buf, rhs_buf,
                          cross_z, cross_v, cross_hvp)
end

function _build_pullback_state(f, hvp_backend, x_star, θ, oracle::Spectraplex, tol;
                                assume_interior::Bool=false,
                                grad=nothing,
                                backend=DEFAULT_BACKEND)
    as = _active_set_for_diff(oracle, x_star;
                               tol=min(tol, 1e-6),
                               assume_interior=assume_interior,
                               caller="rrule(solve)")
    T = promote_type(eltype(as.bound_values), eltype(x_star))
    eq = as.eq_normals
    U = T.(eq.U)
    V_perp = T.(eq.V_perp)
    rank = size(U, 2)
    nullity = size(V_perp, 2)
    reduced_dim = _spectraplex_tangent_dim(rank, nullity)
    m = length(x_star)
    n = size(eq.U, 1)
    fθ = x_ -> f(x_, θ)
    G = reshape(_objective_gradient(f, grad, x_star, θ, backend), n, n)
    G_sym = Symmetric((G .+ G') ./ T(2))
    G_uu = Matrix{T}(transpose(U) * G_sym * U)
    G_vv = Matrix{T}(transpose(V_perp) * G_sym * V_perp)
    dx_dummy = zeros(T, m)
    prep_hvp = DI.prepare_hvp(fθ, hvp_backend, x_star, (dx_dummy,))

    m_θ = length(θ)
    return _SpectraplexPullbackState(
        fθ, x_star, hvp_backend, prep_hvp, as, U, V_perp, G_uu, G_vv, reduced_dim,
        zeros(T, m), zeros(T, m), zeros(T, reduced_dim), zeros(T, reduced_dim), zeros(T, reduced_dim),
        zeros(T, n, rank), zeros(T, n, nullity),
        zeros(T, rank, rank), zeros(T, rank, nullity), zeros(T, rank, nullity),
        zeros(T, n, n), zeros(T, n, n),
        zeros(T, m + m_θ), zeros(T, m + m_θ), zeros(T, m + m_θ))
end

"""
    _reduced_hvp!(state::_PullbackState, w_free)

Compute the reduced Hessian-vector product `P H P w` where `P` is the
null-space projector. Expands `w_free` to full space, computes HVP,
extracts free components, and null-projects. Result is in `state.proj_buf`.
"""
function _reduced_hvp!(state::_PullbackState{T}, w_free) where T
    fill!(state.w_full, zero(T))
    @inbounds for (j, idx) in enumerate(state.free)
        state.w_full[idx] = w_free[j]
    end
    DI.hvp!(state.fθ, (state.hvp_buf,), state.prep_hvp,
            state.hvp_backend, state.x_star, (state.w_full,))
    @inbounds for (j, idx) in enumerate(state.free)
        state.Hw_buf[j] = state.hvp_buf[idx]
    end
    return _null_project!(state.proj_buf, state.Hw_buf, state.a_frees, state.a_norm_sqs)
end

"""
    _build_reduced_hessian!(H_red, state::_PullbackState)

Form the reduced Hessian `H_red = Pᵀ H P` by computing `n_free` reduced HVPs.
"""
function _build_reduced_hessian!(H_red::AbstractMatrix{T}, state::_PullbackState{T}) where T
    for i in 1:state.n_free
        fill!(state.rhs_buf, zero(T))
        state.rhs_buf[i] = one(T)
        Hw = _reduced_hvp!(state, state.rhs_buf)
        H_red[:, i] .= Hw
    end
end

"""
    _build_cross_matrix!(C_red, state, f, grad, x_star, θ, backend, hvp_backend)

Build the `n_free × m` cross-derivative matrix `Pᵀ (∂²f/∂x∂θ)` in the free
subspace, null-projected.

Manual gradient: differentiates `θ → ∇ₓf(x*, θ)` via `DI.jacobian` to get
the `n × m` cross-Hessian, then extracts free rows and null-projects columns.

Auto gradient: uses joint HVPs on `g(z) = f(z[1:n], z[n+1:end])`.
"""
function _build_cross_matrix!(C_red::AbstractMatrix{T}, state::_PullbackState{T},
                               f, grad, x_star::AbstractVector{T}, θ,
                               backend, hvp_backend) where T
    n = length(x_star)
    m = length(θ)
    n_free = state.n_free

    if grad !== nothing
        ∇ₓf_of_θ = _make_∇ₓf_of_θ(grad, x_star)
        prep_jac = DI.prepare_jacobian(∇ₓf_of_θ, backend, θ)
        cross_hess = DI.jacobian(∇ₓf_of_θ, prep_jac, backend, θ)
        @inbounds for (j, idx) in enumerate(state.free)
            C_red[j, :] .= @view(cross_hess[idx, :])
        end
    else
        g = z -> f(@view(z[1:n]), @view(z[n+1:end]))
        @views state.cross_z[1:n] .= x_star
        @views state.cross_z[n+1:end] .= θ
        prep_cross = DI.prepare_hvp(g, hvp_backend, state.cross_z, (state.cross_v,))
        # One HVP per θ-component: v = [0; eⱼ] → Hzz·v gives cross-column
        for j in 1:m
            fill!(state.cross_v, zero(T))
            state.cross_v[n + j] = one(T)
            DI.hvp!(g, (state.cross_hvp,), prep_cross, hvp_backend, state.cross_z, (state.cross_v,))
            @inbounds for (k, idx) in enumerate(state.free)
                C_red[k, j] = state.cross_hvp[idx]
            end
        end
    end

    for j in 1:m
        col = @view(C_red[:, j])
        _null_project!(col, col, state.a_frees, state.a_norm_sqs)
    end
end

"""
    _kkt_adjoint_solve_cached(state::_PullbackState, dx; kwargs...)

 KKT adjoint solve using precomputed `_PullbackState`. Equivalent to
`_kkt_adjoint_solve` but reuses the HVP preparation, orthogonalized
normals, and buffers from `state` instead of recomputing them.
"""
function _kkt_adjoint_solve_cached(state::_PullbackState{T}, dx;
                                    cg_maxiter::Int=50, cg_tol::Real=1e-6, cg_λ::Real=1e-4) where T
    n = length(state.x_star)
    dx_vec = dx isa AbstractVector ? dx : collect(dx)

    # Fast path: no active constraints → CG with HVPs
    if isempty(state.bound) && isempty(state.as.eq_normals)
        hvp_fn = d -> begin
            DI.hvp!(state.fθ, (state.hvp_buf,), state.prep_hvp,
                    state.hvp_backend, state.x_star, (d,))
            state.hvp_buf
        end
        u, cg_result = _cg_solve(hvp_fn, dx_vec; maxiter=cg_maxiter, tol=cg_tol, λ=cg_λ)
        return u, T[], T[], cg_result
    end

    # No free variables → u = 0; recover multipliers from dx directly
    if state.n_free == 0
        μ_eq = _recover_μ_eq(state.as.eq_normals, dx_vec)
        μ_bound = T[dx_vec[i] for i in state.bound]
        _correct_bound_multipliers!(μ_bound, μ_eq, state.as)
        return zeros(T, n), μ_bound, μ_eq, CGResult(0, zero(T), true)
    end

    # Constrained path: matrix-free reduced Hessian CG using cached HVP prep
    reduced_hvp(w_free) = _reduced_hvp!(state, w_free)

    @inbounds for (j, idx) in enumerate(state.free)
        state.rhs_buf[j] = dx_vec[idx]
    end
    rhs = _null_project!(state.rhs_buf, state.rhs_buf, state.a_frees, state.a_norm_sqs)

    u_free, cg_result = _cg_solve(reduced_hvp, rhs; maxiter=cg_maxiter, tol=cg_tol, λ=cg_λ)
    _null_project!(u_free, u_free, state.a_frees, state.a_norm_sqs)

    # Assemble full u
    u = zeros(T, n)
    @inbounds for (j, idx) in enumerate(state.free)
        u[idx] = u_free[j]
    end

    # Multiplier recovery via one final HVP on the full-space adjoint
    DI.hvp!(state.fθ, (state.hvp_buf,), state.prep_hvp,
            state.hvp_backend, state.x_star, (u,))
    @. state.w_full = dx_vec - state.hvp_buf

    μ_eq = _recover_μ_eq(state.a_frees_orig, @view(state.w_full[state.free]))
    μ_bound = T[state.w_full[i] for i in state.bound]
    _correct_bound_multipliers!(μ_bound, μ_eq, state.as)

    return u, μ_bound, μ_eq, cg_result
end

function _kkt_adjoint_solve_cached(state::_SpectraplexPullbackState{T}, dx;
                                    cg_maxiter::Int=50, cg_tol::Real=1e-6, cg_λ::Real=1e-4) where T
    dx_vec = dx isa AbstractVector ? dx : collect(dx)
    λ_T = T(cg_λ)

    if state.reduced_dim == 0
        return zeros(T, length(state.x_star)), T[], T[], CGResult(0, zero(T), true)
    end

    _spectraplex_compress!(state.rhs_buf, dx_vec, state.U, state.V_perp,
                           state.tmp_face_buf, state.tmp_null_buf,
                           state.face_buf, state.mixed_buf, state.full_buf)

    function reduced_hvp(z)
        _spectraplex_expand!(state.w_full, z, state.U, state.V_perp,
                             state.face_buf, state.mixed_buf,
                             state.tmp_face_buf, state.tmp_null_buf,
                             state.full_buf, state.cross_buf)
        DI.hvp!(state.fθ, (state.hvp_buf,), state.prep_hvp,
                state.hvp_backend, state.x_star, (state.w_full,))
        _spectraplex_compress!(state.Hw_buf, state.hvp_buf, state.U, state.V_perp,
                               state.tmp_face_buf, state.tmp_null_buf,
                               state.face_buf, state.mixed_buf, state.full_buf)
        if !iszero(λ_T)
            _spectraplex_compress!(state.reg_buf, state.w_full, state.U, state.V_perp,
                                   state.tmp_face_buf, state.tmp_null_buf,
                                   state.face_buf, state.mixed_buf, state.full_buf)
            @. state.Hw_buf += λ_T * state.reg_buf
        end
        _spectraplex_add_mixed_curvature!(state.Hw_buf, z, state.G_uu, state.G_vv,
                                          state.mixed_buf, state.mixed_curv_buf)
        return state.Hw_buf
    end

    z, cg_result = _cg_solve(reduced_hvp, state.rhs_buf;
                             maxiter=cg_maxiter, tol=cg_tol, λ=zero(T))
    u = zeros(T, length(state.x_star))
    _spectraplex_expand!(u, z, state.U, state.V_perp,
                         state.face_buf, state.mixed_buf,
                         state.tmp_face_buf, state.tmp_null_buf,
                         state.full_buf, state.cross_buf)
    return u, T[], T[], cg_result
end

# ------------------------------------------------------------------
# rrule: solve(f, lmo, x0, θ; grad=..., ...)
# ------------------------------------------------------------------

"""
Implicit differentiation rule for `solve(f, lmo, x0, θ; ...)`.

Handles all `lmo` types (plain functions, `AbstractOracle`, `ParametricOracle`)
and both manual and auto gradient via the `grad=` keyword.

Uses `_kkt_adjoint_solve_cached` with precomputed `_PullbackState`
so that expensive setup (active set identification, HVP preparation, constraint
orthogonalization, buffer allocation) is performed once and amortized across
multiple pullback calls (e.g. when computing a full Jacobian).

See [Implicit Differentiation](@ref) for the full mathematical derivation.
"""
function ChainRulesCore.rrule(::typeof(solve), f, lmo, x0, θ;
                              grad=nothing,
                              backend=DEFAULT_BACKEND,
                              hvp_backend=SECOND_ORDER_BACKEND,
                              diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_lambda::Real=1e-4,
                              assume_interior::Bool=false,
                              tol::Real=1e-4,
                              kwargs...)
    x_star, result = solve(f, lmo, x0, θ;
                           grad=grad, backend=backend,
                           assume_interior=assume_interior,
                           tol=tol, kwargs...)
    if lmo isa ParametricOracle
        oracle = materialize(lmo, θ)
    else
        oracle = lmo
    end

    # ONE-TIME: build cached state for KKT adjoint solve
    state = _build_pullback_state(f, hvp_backend, x_star, θ, oracle, tol;
                                   assume_interior=assume_interior,
                                   grad=grad,
                                   backend=backend)

    _n = length(x_star)
    _m = length(θ)
    _T = eltype(x_star)
    λ_bound = _T[]
    λ_eq = _T[]
    if grad !== nothing
        ∇ₓf_of_θ = _make_∇ₓf_of_θ(grad, x_star)
        _cross_g = nothing
        _cross_prep = nothing
    else
        ∇ₓf_of_θ = nothing
        _cross_g = z -> f(@view(z[1:_n]), @view(z[_n+1:end]))
        @views state.cross_z[1:_n] .= x_star
        @views state.cross_z[_n+1:end] .= θ
        _cross_prep = DI.prepare_hvp(_cross_g, hvp_backend, state.cross_z, (state.cross_v,))
    end
    if lmo isa ParametricOracle
        λ_bound, λ_eq = _primal_face_multipliers(f, grad, x_star, θ, state.as, backend)
    end

    function solve_pullback(dy)
        dx = dy isa Tuple ? dy[1] : dy.x

        if dx isa ChainRulesCore.AbstractZero
            return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end

        u, μ_bound, μ_eq, cg_result = _kkt_adjoint_solve_cached(state, dx;
            cg_maxiter=diff_cg_maxiter, cg_tol=diff_cg_tol, cg_λ=diff_lambda)

        if ∇ₓf_of_θ !== nothing
            dθ_obj = _cross_derivative_manual(∇ₓf_of_θ, u, θ, backend)
        else
            @views begin
                state.cross_v[1:_n] .= u
                state.cross_v[_n+1:end] .= 0
            end
            DI.hvp!(_cross_g, (state.cross_hvp,), _cross_prep, hvp_backend,
                    state.cross_z, (state.cross_v,))
            dθ_obj = -copy(@view(state.cross_hvp[_n+1:end]))
        end

        if lmo isa ParametricOracle
            dθ_con = _constraint_pullback(lmo, θ, x_star, u, μ_bound, μ_eq, λ_bound, λ_eq, state.as, backend)
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

# ------------------------------------------------------------------
# Direct Jacobian computation (full ∂x*/∂θ via reduced Hessian)
# ------------------------------------------------------------------

"""
    jacobian!(J, f, lmo, x0, θ; kwargs...) -> (J, result)

In-place version of [`jacobian`](@ref). Writes the Jacobian
``\\partial x^*/\\partial\\theta`` into the pre-allocated matrix `J`.
"""
function jacobian!(J::AbstractMatrix, f, lmo, x0, θ;
                   grad=nothing, backend=DEFAULT_BACKEND,
                   hvp_backend=SECOND_ORDER_BACKEND,
                   diff_lambda::Real=1e-4, tol::Real=1e-4,
                   assume_interior::Bool=false, kwargs...)
    size(J) == (length(x0), length(θ)) ||
        throw(DimensionMismatch("J must be $(length(x0))×$(length(θ)), got $(size(J))"))
    x_star, result = solve(f, lmo, x0, θ; grad=grad, tol=tol, kwargs...)
    oracle = lmo isa ParametricOracle ? materialize(lmo, θ) : lmo

    state = _build_pullback_state(f, hvp_backend, x_star, θ, oracle, tol;
                                   assume_interior=assume_interior,
                                   grad=grad, backend=backend)
    n = length(x_star)
    m = length(θ)
    T = eltype(x_star)
    fill!(J, zero(T))

    if state.n_free == 0
        return J, SolveResult(x_star, result)
    end

    H_red = zeros(T, state.n_free, state.n_free)
    _build_reduced_hessian!(H_red, state)
    @inbounds for i in 1:state.n_free
        H_red[i, i] += T(diff_lambda)
    end
    F = cholesky(Symmetric(H_red))

    C_red = zeros(T, state.n_free, m)
    _build_cross_matrix!(C_red, state, f, grad, x_star, θ, backend, hvp_backend)

    U_red = F \ C_red

    @inbounds for (j, idx) in enumerate(state.free)
        for k in 1:m
            J[idx, k] = -U_red[j, k]
        end
    end

    return J, SolveResult(x_star, result)
end

"""
    jacobian(f, lmo, x0, θ; kwargs...) -> (J, result)

Compute the full Jacobian ``\\partial x^*/\\partial\\theta \\in \\mathbb{R}^{n \\times m}``
via direct reduced-Hessian factorization.

Forms the reduced Hessian ``P^\\top \\nabla^2 f\\, P`` explicitly (``n_{\\text{free}}``
HVPs), Cholesky-factors it once, then solves all ``m`` right-hand sides in one
shot. Much faster than ``m`` separate pullback calls for full Jacobians.

See [`jacobian!`](@ref) for the in-place version.

Returns `(J, result)` where `result` is a [`SolveResult`](@ref).
"""
function jacobian(f, lmo, x0, θ; kwargs...)
    n = length(x0)
    m = length(θ)
    J = zeros(Float64, n, m)
    return jacobian!(J, f, lmo, x0, θ; kwargs...)
end

