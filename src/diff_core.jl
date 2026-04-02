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
    ACTIVE_SET_TOL_CEILING

Maximum tolerance for active constraint identification during differentiation.
The active-set tolerance is `min(solver_tol, ACTIVE_SET_TOL_CEILING)` -- tight
enough to distinguish active from inactive constraints even when the solver
tolerance is loose.
"""
const ACTIVE_SET_TOL_CEILING = 1e-6

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
            @warn "CG encountered near-zero curvature (pHp=$pHp): Hessian may be singular. Consider increasing diff_lambda." maxlog=10
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
        @warn "CG solve did not converge: residual=$residual after $iters iterations" maxlog=10
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
    T = promote_type(eltype(x_star), eltype(θ))
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

# ── Orthogonalization and null space projection ─────────────────────

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
    # pinv handles near-parallel constraint normals gracefully (G can be rank deficient)
    return T.(pinv(G, rtol=sqrt(eps(T))) * b)
end

"""
    _null_project!(out, w, a_frees, a_norm_sqs)

Project `w` (in free variable space) onto the null space of pre-computed
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
            @warn "null space projection: constraint normal $j has near-zero free space norm (||a||²=$a_norm_sq); skipped" maxlog=3
        end
    end
    return out
end

"""
    _factor_reduced_hessian(H_red, diff_lambda) -> factorization

Regularize `H_red` with `diff_lambda`, then Cholesky-factor. Falls back to LU
if the Hessian is indefinite. Errors with an actionable message if both fail.
"""
function _factor_reduced_hessian(H_red::AbstractMatrix{T}, diff_lambda::Real) where T
    d = size(H_red, 1)
    @inbounds for i in 1:d
        H_red[i, i] += T(diff_lambda)
    end
    factor = cholesky(Symmetric(H_red); check=false)
    if issuccess(factor)
        return factor
    end
    @warn "Reduced Hessian is not positive definite; falling back to LU. Consider increasing diff_lambda (current: $diff_lambda)." maxlog=3
    factor = lu(H_red; check=false)
    if issuccess(factor)
        return factor
    end
    error("Reduced Hessian factorization failed (Cholesky and LU). The Hessian may be singular. Try increasing diff_lambda.")
end

# ── Spectraplex tangent space operations ────────────────────────────
#
# These functions implement a compressed coordinate system for the tangent
# space of the spectraplex at a rank deficient solution X*.
#
# The tangent space has two blocks:
#   1. Face block (rank × rank, trace zero, symmetric): perturbations within
#      the active eigenspace. Dimension: rank*(rank+1)/2 - 1.
#   2. Mixed block (rank × nullity): cross-perturbations between the active
#      and null eigenspaces. Dimension: rank * nullity.
#
# The pack/unpack functions convert between matrices and flat vectors.
# The compress/expand functions convert between full n²-vectors and
# tangent space coordinates using the eigenvector bases U and V_perp.

"""
    _spectraplex_trace_zero_dim(rank) -> Int

Number of free parameters in a `rank × rank` symmetric, trace zero matrix.
Equal to `rank*(rank+1)/2 - 1` (upper triangle minus the trace constraint).
"""
@inline function _spectraplex_trace_zero_dim(rank::Int)
    return rank == 0 ? 0 : rank * (rank + 1) ÷ 2 - 1
end

"""
    _spectraplex_tangent_dim(rank, nullity) -> Int

Total dimension of the spectraplex tangent space: face block (trace zero
symmetric, `rank*(rank+1)/2 - 1`) plus mixed block (`rank * nullity`).
"""
@inline function _spectraplex_tangent_dim(rank::Int, nullity::Int)
    return _spectraplex_trace_zero_dim(rank) + rank * nullity
end

"""
    _spectraplex_pack_trace_zero!(out, M) -> out

Encode a `k × k` symmetric trace zero matrix `M` as a flat vector.

Layout: first `k-1` entries are diagonal differences `M[i,i] - M[k,k]`,
then upper-triangle off-diagonal sums `M[i,j] + M[j,i]`. The last diagonal
is implicit via the trace zero constraint.

Inverse: [`_spectraplex_unpack_trace_zero!`](@ref).
"""
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

"""
    _spectraplex_pack_mixed!(out, B, offset) -> out

Pack the `rank × nullity` mixed block `B` into `out` starting at `offset`,
in column major order. Inverse: [`_spectraplex_unpack_mixed!`](@ref).
"""
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

"""
    _spectraplex_unpack_trace_zero!(S, z) -> S

Recover a `k × k` symmetric trace zero matrix `S` from its packed vector `z`.

Inverts [`_spectraplex_pack_trace_zero!`](@ref): recovers diagonals from the
stored differences plus the trace zero constraint (`S[k,k] = -∑ z[1:k-1] / k`),
then fills the symmetric off-diagonals from the stored sums (`z[p] / 2`).
"""
function _spectraplex_unpack_trace_zero!(S::AbstractMatrix{T}, z::AbstractVector{T}) where T
    k = size(S, 1)
    fill!(S, zero(T))
    k == 0 && return S

    # Invert pack's diagonal differences: z[i] = M[i,i] - M[k,k]
    # Using trace zero: M[k,k] = -sum(z) / k
    p = 1
    diag_diff_sum = zero(T)
    @inbounds for i in 1:(k - 1)
        diag_diff_sum += z[p]
        p += 1
    end
    diag_ref = -diag_diff_sum / T(k)

    p = 1
    @inbounds for i in 1:(k - 1)
        S[i, i] = z[p] + diag_ref
        p += 1
    end
    S[k, k] = diag_ref

    # Invert pack's off-diagonal sums: z[p] = M[i,j] + M[j,i] = 2·M[i,j]
    @inbounds for j in 2:k
        for i in 1:(j - 1)
            val = z[p] / T(2)
            S[i, j] = val
            S[j, i] = val
            p += 1
        end
    end
    return S
end

"""
    _spectraplex_unpack_mixed!(B, z, offset) -> B

Unpack the `rank × nullity` mixed block from `z` starting at `offset` into
matrix `B`, in column major order. Inverse: [`_spectraplex_pack_mixed!`](@ref).
"""
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

"""
    _spectraplex_expand!(out, z, U, V_perp, face, mixed, tmp_face, tmp_null, full, cross) -> out

Expand tangent space coordinates `z` to a full `n²` vector `out`.

Reconstructs the symmetric matrix perturbation:
``\\Delta X = U \\, S \\, U^\\top + U \\, B \\, V_\\perp^\\top + V_\\perp \\, B^\\top \\, U^\\top``
where `S` is the trace zero face block and `B` is the mixed block, both
unpacked from `z`. Then vectorizes column major into `out`.

Inverse: [`_spectraplex_compress!`](@ref).
"""
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

"""
    _spectraplex_compress!(out, x, U, V_perp, tmp_face, tmp_null, face, mixed, full) -> out

Compress a full `n²` vector `x` to tangent space coordinates `out`.

Symmetrizes `x` as an `n × n` matrix, then extracts:
- Face block: ``U^\\top \\operatorname{sym}(X) \\, U`` → packed trace zero
- Mixed block: ``U^\\top \\operatorname{sym}(X) \\, V_\\perp`` → packed column major

Inverse: [`_spectraplex_expand!`](@ref).
"""
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

"""
    _spectraplex_add_mixed_curvature!(out, z, G_uu, G_vv, mixed, mixed_curv) -> out

Add the PSD cone curvature correction to a reduced Hessian-vector product.

At a rank deficient optimum, perturbing in the active×null cross block `B`
(extracted from `z`) sees curvature from the cone boundary:
``\\Delta_{\\text{out}} \\mathrel{+}= B \\, G_{vv} - G_{uu} \\, B``
where `G_uu` and `G_vv` are the objective Hessian restricted to the active
and null eigenspaces. This term is zero when the solution is full-rank.
"""
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
    # B * G_vv - G_uu * B at rank deficient optima.
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

# ── Reduced Hessian factorization and multiplier recovery ───────────

"""
    _correct_bound_multipliers!(μ_bound, μ_eq, as::ActiveConstraints)

Correct raw bound multipliers by subtracting equality-constraint overlap.

The KKT residual at a bound index `iₖ` contains both the bound multiplier
and projections of equality multipliers onto that index. This function
removes the equality contributions so that `μ_bound` reflects only the
bound constraint force:

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

# ── KKT adjoint solve (non-cached, used by bilevel_solve) ───────────

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
1. Set ``u[\\text{bound}] = 0``, work only in free variable subspace
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
    n_vec = length(x_star)
    dx_vec = dx isa AbstractVector ? dx : collect(dx)
    eq = as.eq_normals
    U = convert(Matrix{T}, eq.U)
    V_perp = convert(Matrix{T}, eq.V_perp)
    rank = size(U, 2)
    nullity = size(V_perp, 2)
    d = _spectraplex_tangent_dim(rank, nullity)

    if d == 0
        if !iszero(eq.trace_rhs)
            @warn "Spectraplex tangent dimension is 0 for non-zero radius (r=$(eq.trace_rhs)): gradient will be zero. This may indicate degenerate rank detection." maxlog=3
        end
        return zeros(T, n_vec), T[], T[], CGResult(0, zero(T), true)
    end

    fθ = x_ -> f(x_, θ)
    prep_hvp = DI.prepare_hvp(fθ, hvp_backend, x_star, (dx_vec,))

    rhs = zeros(T, d)
    Hw = zeros(T, d)
    w_full = zeros(T, n_vec)
    hvp_buf = zeros(T, n_vec)
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
        # N.B. operation order is load-bearing: compress writes the mixed block
        # of Hw from mixed_buf before add_mixed_curvature! overwrites mixed_buf
        _spectraplex_compress!(Hw, hvp_buf, U, V_perp,
                               tmp_face_buf, tmp_null_buf,
                               face_buf, mixed_buf, full_buf)
        if !iszero(λ_T)
            @. Hw += λ_T * z
        end
        _spectraplex_add_mixed_curvature!(Hw, z, G_uu, G_vv, mixed_buf, mixed_curv_buf)
        return Hw
    end

    # λ=zero(T): regularization is inside reduced_hvp (alongside mixed curvature correction)
    z, cg_result = _cg_solve(reduced_hvp, rhs; maxiter=cg_maxiter, tol=cg_tol, λ=zero(T))
    u = zeros(T, n_vec)
    _spectraplex_expand!(u, z, U, V_perp,
                         face_buf, mixed_buf,
                         tmp_face_buf, tmp_null_buf,
                         full_buf, cross_buf)
    return u, T[], T[], cg_result
end

# ── Cross-derivative helpers ────────────────────────────────────────

"""
    _make_∇ₓf_of_θ(∇f!, x_star)

Build the map ``\\theta \\mapsto \\nabla_x f(x^*, \\theta)`` from a mutating gradient
`∇f!(g, x, θ)` and a fixed solution `x_star`.

The returned closure allocates a type promoted buffer so that forward mode AD
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

!!! note
    Uses `@view` slicing which requires the AD backend to support `SubArray`
    differentiation. ForwardDiff and Mooncake support this; other backends
    (e.g. Enzyme, Zygote) may not. If using an unsupported backend, provide
    a manual `grad` function to bypass this joint-HVP path.
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

# ── Constraint pullback (ParametricOracle) ──────────────────────────

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
