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

# ── Orthogonalization and null-space projection ─────────────────────

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
    # pinv handles near-parallel constraint normals gracefully (G can be rank-deficient)
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

# ── Spectraplex tangent-space operations ────────────────────────────
#
# These functions implement a compressed coordinate system for the tangent
# space of the spectraplex at a rank-deficient solution X*.
#
# The tangent space has two blocks:
#   1. Face block (rank × rank, trace-zero, symmetric): perturbations within
#      the active eigenspace. Dimension: rank*(rank+1)/2 - 1.
#   2. Mixed block (rank × nullity): cross-perturbations between the active
#      and null eigenspaces. Dimension: rank * nullity.
#
# The pack/unpack functions convert between matrices and flat vectors.
# The compress/expand functions convert between full n²-vectors and
# tangent-space coordinates using the eigenvector bases U and V_perp.

"""
    _spectraplex_trace_zero_dim(rank) -> Int

Number of free parameters in a `rank × rank` symmetric, trace-zero matrix.
Equal to `rank*(rank+1)/2 - 1` (upper triangle minus the trace constraint).
"""
@inline function _spectraplex_trace_zero_dim(rank::Int)
    return rank == 0 ? 0 : rank * (rank + 1) ÷ 2 - 1
end

"""
    _spectraplex_tangent_dim(rank, nullity) -> Int

Total dimension of the spectraplex tangent space: face block (trace-zero
symmetric, `rank*(rank+1)/2 - 1`) plus mixed block (`rank * nullity`).
"""
@inline function _spectraplex_tangent_dim(rank::Int, nullity::Int)
    return _spectraplex_trace_zero_dim(rank) + rank * nullity
end

"""
    _spectraplex_pack_trace_zero!(out, M) -> out

Encode a `k × k` symmetric trace-zero matrix `M` as a flat vector.

Layout: first `k-1` entries are diagonal differences `M[i,i] - M[k,k]`,
then upper-triangle off-diagonal sums `M[i,j] + M[j,i]`. The last diagonal
is implicit via the trace-zero constraint.

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
in column-major order. Inverse: [`_spectraplex_unpack_mixed!`](@ref).
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

Recover a `k × k` symmetric trace-zero matrix `S` from its packed vector `z`.

Inverts [`_spectraplex_pack_trace_zero!`](@ref): recovers diagonals from the
stored differences plus the trace-zero constraint (`S[k,k] = -∑ z[1:k-1] / k`),
then fills the symmetric off-diagonals from the stored sums (`z[p] / 2`).
"""
function _spectraplex_unpack_trace_zero!(S::AbstractMatrix{T}, z::AbstractVector{T}) where T
    k = size(S, 1)
    fill!(S, zero(T))
    k == 0 && return S

    # Invert pack's diagonal differences: z[i] = M[i,i] - M[k,k]
    # Using trace-zero: M[k,k] = -sum(z) / k
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
matrix `B`, in column-major order. Inverse: [`_spectraplex_pack_mixed!`](@ref).
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

Expand tangent-space coordinates `z` to a full `n²` vector `out`.

Reconstructs the symmetric matrix perturbation:
``\\Delta X = U \\, S \\, U^\\top + U \\, B \\, V_\\perp^\\top + V_\\perp \\, B^\\top \\, U^\\top``
where `S` is the trace-zero face block and `B` is the mixed block, both
unpacked from `z`. Then vectorizes column-major into `out`.

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

Compress a full `n²` vector `x` to tangent-space coordinates `out`.

Symmetrizes `x` as an `n × n` matrix, then extracts:
- Face block: ``U^\\top \\operatorname{sym}(X) \\, U`` → packed trace-zero
- Mixed block: ``U^\\top \\operatorname{sym}(X) \\, V_\\perp`` → packed column-major

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

At a rank-deficient optimum, perturbing in the active×null cross-block `B`
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

# ── TangentMap types and interface ───────────────────────────────────

"""
    _TangentMap{T}

Oracle-specific tangent-space geometry for implicit differentiation.

At a constrained optimum, only certain directions are "free" — the rest are
pinned by active constraints. A tangent map encodes how to move between the
full variable space and the reduced space of free directions (the "tangent
space" of the constraint surface).

All polyhedral oracles (Simplex, Box, Knapsack, etc.) share one tangent map
([`_PolyhedralTangentMap`](@ref)) because they all have the same structure:
some variables hit bounds, linear equality constraints restrict the rest.
The Spectraplex needs its own ([`_SpectralTangentMap`](@ref)) because its
constraint surface (the PSD cone) has curved geometry requiring
eigendecomposition-based operations.

Interface methods (dispatched on concrete subtype):
- [`_project_tangent!`](@ref): full space → tangent space
- [`_expand_tangent!`](@ref): tangent space → full space
- [`_tangent_correction!`](@ref): post-HVP correction for reduced Hessian
- [`_reduced_dim`](@ref): dimension of the tangent space
"""
abstract type _TangentMap{T} end

"""
    _PolyhedralTangentMap{T} <: _TangentMap{T}

Tangent map for polyhedral constraint sets (Simplex, Box, Knapsack, etc.).

At the solution, variables split into two groups:
- **Bound**: variables on a constraint boundary that cannot move.
- **Free**: interior variables that form the reduced search space.

Equality constraints (e.g. ``\\sum x_i = r`` for the Simplex) further restrict
the free variables to a subspace. The orthogonalized constraint normals
`a_frees` and their squared norms `a_norm_sqs` enable null-space projection:
removing components that would violate equality constraints.
"""
struct _PolyhedralTangentMap{T} <: _TangentMap{T}
    free::Vector{Int}
    bound::Vector{Int}
    a_frees::Vector{Vector{T}}
    a_frees_orig::Vector{Vector{T}}
    a_norm_sqs::Vector{T}
end

"""
    _SpectralTangentMap{T} <: _TangentMap{T}

Tangent map for the Spectraplex constraint (``X \\succeq 0``, ``\\operatorname{tr}(X) = r``).

The solution matrix ``X^*`` has an eigendecomposition with `rank` nonzero
eigenvalues. The eigenvectors split into:
- **U** (n × rank): eigenvectors with nonzero eigenvalues — the "active face"
  of the PSD cone, where ``X^*`` has support.
- **V_perp** (n × nullity): eigenvectors with zero eigenvalues — the boundary
  of the PSD cone, where ``X^*`` touches the cone edge.

The tangent space at ``X^*`` consists of symmetric perturbations that preserve
trace and PSD structure. Its dimension is `rank*(rank+1)/2 - 1` (trace-zero
face block) plus `rank * nullity` (active×null cross-block).

`G_uu` and `G_vv` are the objective's Hessian projected onto the active and
null eigenspaces. They appear in a curvature correction term: unlike polyhedral
constraints, moving along the active×null cross-block incurs additional
curvature `B·G_vv - G_uu·B` from the cone boundary itself.
"""
struct _SpectralTangentMap{T} <: _TangentMap{T}
    U::Matrix{T}
    V_perp::Matrix{T}
    G_uu::Matrix{T}
    G_vv::Matrix{T}
    tmp_face::Matrix{T}
    tmp_null::Matrix{T}
    face::Matrix{T}
    mixed::Matrix{T}
    mixed_curv::Matrix{T}
    full::Matrix{T}
    cross::Matrix{T}
end

"""
    _reduced_dim(tm::_TangentMap) -> Int

Working dimension for the reduced Hessian. For polyhedral, this is the number of
free variables (equality constraints are handled via null-space projection, not
dimensional reduction).
"""
@inline _reduced_dim(tm::_PolyhedralTangentMap) = length(tm.free)
@inline _reduced_dim(tm::_SpectralTangentMap) = _spectraplex_tangent_dim(size(tm.U, 2), size(tm.V_perp, 2))

"""
    _project_tangent!(out, v, tm::_TangentMap)

Project full-space vector `v` into the tangent space, writing to `out`
(length [`_reduced_dim(tm)`](@ref)).

Polyhedral: extracts free-variable components.
Spectral: compresses via [`_spectraplex_compress!`](@ref).
"""
function _project_tangent!(out::AbstractVector{T}, v::AbstractVector{T},
                            tm::_PolyhedralTangentMap{T}) where T
    @inbounds for (j, idx) in enumerate(tm.free)
        out[j] = v[idx]
    end
    return out
end

function _project_tangent!(out::AbstractVector{T}, v::AbstractVector{T},
                            tm::_SpectralTangentMap{T}) where T
    return _spectraplex_compress!(out, v, tm.U, tm.V_perp,
                                  tm.tmp_face, tm.tmp_null,
                                  tm.face, tm.mixed, tm.full)
end

"""
    _expand_tangent!(out, z, tm::_TangentMap)

Expand tangent-space vector `z` back to the full variable space, writing
to `out`.

Polyhedral: scatters into free-variable positions, zeros elsewhere.
Spectral: reconstructs via [`_spectraplex_expand!`](@ref).
"""
function _expand_tangent!(out::AbstractVector{T}, z::AbstractVector{T},
                           tm::_PolyhedralTangentMap{T}) where T
    fill!(out, zero(T))
    @inbounds for (j, idx) in enumerate(tm.free)
        out[idx] = z[j]
    end
    return out
end

function _expand_tangent!(out::AbstractVector{T}, z::AbstractVector{T},
                           tm::_SpectralTangentMap{T}) where T
    return _spectraplex_expand!(out, z, tm.U, tm.V_perp,
                                tm.face, tm.mixed,
                                tm.tmp_face, tm.tmp_null,
                                tm.full, tm.cross)
end

"""
    _tangent_correction!(out, z, tm::_TangentMap)

Post-HVP correction when building the reduced Hessian matrix.

Polyhedral: null-space projection — removes components along equality
constraint normals that are not truly free.
Spectral: adds the mixed curvature term from the PSD cone geometry.
At a rank-deficient optimum, perturbing in the active×null cross-block
sees additional curvature ``B G_{vv} - G_{uu} B`` from the cone boundary.
"""
function _tangent_correction!(out::AbstractVector{T}, ::AbstractVector{T},
                               tm::_PolyhedralTangentMap{T}) where T
    return _null_project!(out, out, tm.a_frees, tm.a_norm_sqs)
end

function _tangent_correction!(out::AbstractVector{T}, z::AbstractVector{T},
                               tm::_SpectralTangentMap{T}) where T
    return _spectraplex_add_mixed_curvature!(out, z, tm.G_uu, tm.G_vv,
                                             tm.mixed, tm.mixed_curv)
end

# ── Pullback state ──────────────────────────────────────────────────

"""
    _PullbackState{T, FT, HB, P, AS, HF, TM}

Pre-computed state for the rrule pullback closure. Built once in the rrule
body and reused across all pullback calls (e.g. when computing a full
Jacobian via ``n`` pullback calls).

Caches the active set, HVP preparation, tangent-space geometry
([`_TangentMap`](@ref)), pre-allocated work buffers, and the Cholesky
(or LU) factorization of the reduced Hessian for direct backsubstitution.
"""
struct _PullbackState{T, FT, HB, P, AS<:ActiveConstraints, HF, TM<:_TangentMap{T}}
    fθ::FT
    x_star::Vector{T}
    hvp_backend::HB
    prep_hvp::P
    as::AS
    tangent_map::TM
    reduced_dim::Int
    w_full::Vector{T}
    hvp_buf::Vector{T}
    Hw_buf::Vector{T}
    rhs_buf::Vector{T}
    cross_z::Vector{T}
    cross_v::Vector{T}
    cross_hvp::Vector{T}
    hessian_factor::HF
end

@inline function _interior_active_constraints(x::AbstractVector{T}) where T
    n = length(x)
    return ActiveConstraints{T}(Int[], T[], BitVector(), collect(1:n), Vector{T}[], T[])
end

"""
    _has_active_set(oracle)

Trait: returns `true` if the oracle type has a specialized `active_set` method.
Built-in oracles are enumerated. Custom oracle types should also define
`Marguerite._has_active_set(::MyOracle) = true` to enable differentiation.

The fallback checks whether a typed (non-generic) `active_set` method exists
for the oracle type via `which` signature inspection.
"""
_has_active_set(::Simplex) = true
_has_active_set(::Knapsack) = true
_has_active_set(::MaskedKnapsack) = true
_has_active_set(::Box) = true
_has_active_set(::ScalarBox) = true
_has_active_set(::WeightedSimplex) = true
_has_active_set(::Spectraplex) = true
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
    _build_tangent_map(T, oracle, as, f, grad, x_star, θ, backend) -> _TangentMap

Construct the oracle-specific tangent map from the active constraint set.
Polyhedral oracles get [`_PolyhedralTangentMap`](@ref), Spectraplex gets
[`_SpectralTangentMap`](@ref).
"""
function _build_tangent_map(::Type{T}, oracle, as::ActiveConstraints,
                            f, grad, x_star, θ, backend) where T
    free = as.free_indices
    bound = as.bound_indices
    a_frees_orig = [T.(a[free]) for a in as.eq_normals]
    a_frees = [copy(a) for a in a_frees_orig]
    a_norm_sqs = T[dot(a, a) for a in a_frees]
    _orthogonalize!(a_frees, a_norm_sqs)
    return _PolyhedralTangentMap{T}(free, bound, a_frees, a_frees_orig, a_norm_sqs)
end

function _build_tangent_map(::Type{T}, oracle::Spectraplex,
                            as::ActiveConstraints{<:Real, <:SpectraplexEqNormals},
                            f, grad, x_star, θ, backend) where T
    eq = as.eq_normals
    U = convert(Matrix{T}, eq.U)
    V_perp = convert(Matrix{T}, eq.V_perp)
    rank = size(U, 2)
    nullity = size(V_perp, 2)
    n = size(U, 1)
    if _spectraplex_tangent_dim(rank, nullity) == 0 && !iszero(oracle.r)
        @warn "Spectraplex tangent dimension is 0 for non-zero radius (r=$(oracle.r)): gradient will be zero. This may indicate degenerate rank detection." maxlog=3
    end
    G = reshape(_objective_gradient(f, grad, x_star, θ, backend), n, n)
    G_sym = Symmetric((G .+ G') ./ T(2))
    G_uu = Matrix{T}(transpose(U) * G_sym * U)
    G_vv = Matrix{T}(transpose(V_perp) * G_sym * V_perp)
    return _SpectralTangentMap{T}(U, V_perp, G_uu, G_vv,
                                  zeros(T, n, rank), zeros(T, n, nullity),
                                  zeros(T, rank, rank), zeros(T, rank, nullity),
                                  zeros(T, rank, nullity),
                                  zeros(T, n, n), zeros(T, n, n))
end

"""
    _build_pullback_state(f, hvp_backend, x_star, θ, oracle, tol; ...)

Build [`_PullbackState`](@ref) once in the rrule body. Performs active set
identification, tangent map construction, reduced Hessian factorization, and
buffer allocation — all invariant across pullback calls.
"""
function _build_pullback_state(f, hvp_backend, x_star, θ, oracle, tol;
                                assume_interior::Bool=false,
                                grad=nothing,
                                backend=DEFAULT_BACKEND,
                                diff_lambda::Real=1e-4)
    as = _active_set_for_diff(oracle, x_star;
                               tol=min(tol, 1e-6),
                               assume_interior=assume_interior,
                               caller="rrule(solve)")
    T = promote_type(eltype(as.bound_values), eltype(x_star))
    tm = _build_tangent_map(T, oracle, as, f, grad, x_star, θ, backend)
    d = _reduced_dim(tm)

    n = length(x_star)
    fθ = x_ -> f(x_, θ)
    dx_dummy = zeros(T, n)
    prep_hvp = DI.prepare_hvp(fθ, hvp_backend, x_star, (dx_dummy,))

    w_full = zeros(T, n)
    hvp_buf = zeros(T, n)
    Hw_buf = zeros(T, d)
    rhs_buf = zeros(T, d)
    m_θ = length(θ)
    cross_z = zeros(T, n + m_θ)
    cross_v = zeros(T, n + m_θ)
    cross_hvp_buf = zeros(T, n + m_θ)

    # Build and factor reduced Hessian via d HVPs through the tangent map
    if d > 0
        H_red = zeros(T, d, d)
        eᵢ = zeros(T, d)
        for i in 1:d
            fill!(eᵢ, zero(T))
            eᵢ[i] = one(T)
            _expand_tangent!(w_full, eᵢ, tm)
            DI.hvp!(fθ, (hvp_buf,), prep_hvp, hvp_backend, x_star, (w_full,))
            _project_tangent!(Hw_buf, hvp_buf, tm)
            _tangent_correction!(Hw_buf, eᵢ, tm)
            H_red[:, i] .= Hw_buf
        end
        hessian_factor = _factor_reduced_hessian(H_red, diff_lambda)
    else
        hessian_factor = nothing
    end

    return _PullbackState(fθ, x_star, hvp_backend, prep_hvp, as,
                          tm, d, w_full, hvp_buf, Hw_buf, rhs_buf,
                          cross_z, cross_v, cross_hvp_buf, hessian_factor)
end


"""
    _build_cross_matrix!(C_red, state::_PullbackState, f, grad, x_star, θ, backend, hvp_backend)

Build the `d × m` cross-derivative matrix ``P^\\top (\\partial^2 f / \\partial x \\partial \\theta)``
in the tangent space.

Manual gradient path: differentiates `θ → ∇ₓf(x*, θ)` via `DI.jacobian`,
then projects each column into the tangent space.
Auto gradient path: `m` joint HVPs with `v = [0; eⱼ]`, project each result.
"""
function _build_cross_matrix!(C_red::AbstractMatrix{T}, state::_PullbackState{T},
                               f, grad, x_star::AbstractVector{T}, θ,
                               backend, hvp_backend) where T
    n = length(x_star)
    m = length(θ)
    tm = state.tangent_map
    d = state.reduced_dim
    col_buf = zeros(T, d)

    if grad !== nothing
        ∇ₓf_of_θ = _make_∇ₓf_of_θ(grad, x_star)
        prep_jac = DI.prepare_jacobian(∇ₓf_of_θ, backend, θ)
        cross_hess = DI.jacobian(∇ₓf_of_θ, prep_jac, backend, θ)
        for j in 1:m
            _project_tangent!(col_buf, @view(cross_hess[:, j]), tm)
            C_red[:, j] .= col_buf
        end
    else
        g = z -> f(@view(z[1:n]), @view(z[n+1:end]))
        @views state.cross_z[1:n] .= x_star
        @views state.cross_z[n+1:end] .= θ
        prep_cross = DI.prepare_hvp(g, hvp_backend, state.cross_z, (state.cross_v,))
        for j in 1:m
            fill!(state.cross_v, zero(T))
            state.cross_v[n + j] = one(T)
            DI.hvp!(g, (state.cross_hvp,), prep_cross, hvp_backend, state.cross_z, (state.cross_v,))
            _project_tangent!(col_buf, @view(state.cross_hvp[1:n]), tm)
            C_red[:, j] .= col_buf
        end
    end

    # Polyhedral: null-project each column to remove equality-constraint components
    if tm isa _PolyhedralTangentMap
        for j in 1:m
            col = @view(C_red[:, j])
            _null_project!(col, col, tm.a_frees, tm.a_norm_sqs)
        end
    end
end

"""
    _kkt_adjoint_solve_cached(state::_PullbackState, dx)

KKT adjoint solve using precomputed [`_PullbackState`](@ref). Projects `dx`
into the tangent space, solves via the cached Hessian factorization, and
expands back to full space. For polyhedral oracles, also recovers KKT
multipliers via one HVP.
"""
function _kkt_adjoint_solve_cached(state::_PullbackState{T}, dx) where T
    n = length(state.x_star)
    dx_vec = dx isa AbstractVector ? dx : collect(dx)
    tm = state.tangent_map
    d = state.reduced_dim

    # Degenerate: zero tangent dimension
    if d == 0
        if tm isa _PolyhedralTangentMap
            μ_eq = _recover_μ_eq(state.as.eq_normals, dx_vec)
            μ_bound = T[dx_vec[i] for i in tm.bound]
            _correct_bound_multipliers!(μ_bound, μ_eq, state.as)
            return zeros(T, n), μ_bound, μ_eq, CGResult(0, zero(T), true)
        end
        return zeros(T, n), T[], T[], CGResult(0, zero(T), true)
    end

    # Unconstrained fast path (polyhedral with no active constraints)
    if tm isa _PolyhedralTangentMap && isempty(tm.bound) && isempty(state.as.eq_normals)
        u = state.hessian_factor \ dx_vec
        return u, T[], T[], CGResult(0, zero(T), true)
    end

    # Project dx into tangent space and solve
    _project_tangent!(state.rhs_buf, dx_vec, tm)
    if tm isa _PolyhedralTangentMap
        _null_project!(state.rhs_buf, state.rhs_buf, tm.a_frees, tm.a_norm_sqs)
    end

    u_red = state.hessian_factor \ state.rhs_buf
    if tm isa _PolyhedralTangentMap
        _null_project!(u_red, u_red, tm.a_frees, tm.a_norm_sqs)
    end

    # Expand to full space
    u = zeros(T, n)
    _expand_tangent!(u, u_red, tm)

    # Multiplier recovery (polyhedral only — spectraplex has no explicit multipliers)
    if tm isa _PolyhedralTangentMap
        DI.hvp!(state.fθ, (state.hvp_buf,), state.prep_hvp,
                state.hvp_backend, state.x_star, (u,))
        @. state.w_full = dx_vec - state.hvp_buf
        μ_eq = _recover_μ_eq(tm.a_frees_orig, @view(state.w_full[tm.free]))
        μ_bound = T[state.w_full[i] for i in tm.bound]
        _correct_bound_multipliers!(μ_bound, μ_eq, state.as)
    else
        μ_bound = T[]
        μ_eq = T[]
    end

    return u, μ_bound, μ_eq, CGResult(0, zero(T), true)
end

# ── rrule ───────────────────────────────────────────────────────────

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
    as_tol = min(tol, 1e-6)
    if !result.converged && result.gap > 10 * as_tol
        @warn "rrule(solve): solver gap ($(result.gap)) >> active-set tolerance ($as_tol); differentiation may be inaccurate. Consider tightening tol." maxlog=3
    end
    if lmo isa ParametricOracle
        oracle = materialize(lmo, θ)
    else
        oracle = lmo
    end

    # ONE-TIME: build cached state for KKT adjoint solve
    state = _build_pullback_state(f, hvp_backend, x_star, θ, oracle, tol;
                                   assume_interior=assume_interior,
                                   grad=grad,
                                   backend=backend,
                                   diff_lambda=diff_lambda)

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

        u, μ_bound, μ_eq, cg_result = _kkt_adjoint_solve_cached(state, dx)

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
            @warn "rrule pullback: CG did not converge (residual=$(cg_result.residual_norm), iters=$(cg_result.iterations)): dθ may be inaccurate" maxlog=100
        end
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), dθ
    end

    return SolveResult(x_star, result), solve_pullback
end

# ── solution_jacobian ───────────────────────────────────────────────

"""
    solution_jacobian!(J, f, lmo, x0, θ; kwargs...) -> (J, result)

In-place version of [`solution_jacobian`](@ref). Writes the Jacobian
``\\partial x^*/\\partial\\theta`` into the pre-allocated matrix `J`.
"""
function solution_jacobian!(J::AbstractMatrix, f, lmo, x0, θ;
                   grad=nothing, backend=DEFAULT_BACKEND,
                   hvp_backend=SECOND_ORDER_BACKEND,
                   diff_lambda::Real=1e-4, tol::Real=1e-4,
                   assume_interior::Bool=false, kwargs...)
    size(J) == (length(x0), length(θ)) ||
        throw(DimensionMismatch("J must be $(length(x0))×$(length(θ)), got $(size(J))"))
    if lmo isa ParametricOracle
        throw(ArgumentError(
            "solution_jacobian does not yet support ParametricOracle (constraint sensitivity is not included). " *
            "Use the rrule-based pullback approach instead: (_, pb) = rrule(solve, f, lmo, x0, θ; ...)."))
    end
    x_star, result = solve(f, lmo, x0, θ; grad=grad, backend=backend, tol=tol, kwargs...)
    if !result.converged
        @warn "solution_jacobian: inner solve did not converge (gap=$(result.gap)): Jacobian may be inaccurate" maxlog=3
    end
    oracle = lmo isa ParametricOracle ? materialize(lmo, θ) : lmo

    state = _build_pullback_state(f, hvp_backend, x_star, θ, oracle, tol;
                                   assume_interior=assume_interior,
                                   grad=grad, backend=backend,
                                   diff_lambda=diff_lambda)
    T = eltype(x_star)
    fill!(J, zero(T))
    _solution_jacobian_impl!(J, state, f, grad, x_star, θ, backend, hvp_backend)
    return J, SolveResult(x_star, result)
end

function _solution_jacobian_impl!(J, state::_PullbackState{T},
                                   f, grad, x_star, θ, backend, hvp_backend) where T
    m = length(θ)
    d = state.reduced_dim
    d == 0 && return J
    tm = state.tangent_map

    C_red = zeros(T, d, m)
    _build_cross_matrix!(C_red, state, f, grad, x_star, θ, backend, hvp_backend)
    U_red = state.hessian_factor \ C_red

    col_buf = zeros(T, length(x_star))
    z_buf = zeros(T, d)
    @inbounds for k in 1:m
        for i in 1:d
            z_buf[i] = U_red[i, k]
        end
        _expand_tangent!(col_buf, z_buf, tm)
        for i in eachindex(col_buf)
            J[i, k] = -col_buf[i]
        end
    end
    return J
end

"""
    solution_jacobian(f, lmo, x0, θ; kwargs...) -> (J, result)

Compute the full Jacobian ``\\partial x^*/\\partial\\theta \\in \\mathbb{R}^{n \\times m}``
via direct reduced-Hessian factorization.

Forms the reduced Hessian ``P^\\top \\nabla^2 f\\, P`` explicitly (``n_{\\text{free}}``
HVPs), Cholesky-factors it once, then solves all ``m`` right-hand sides in one
shot. Much faster than ``m`` separate pullback calls for full Jacobians.

See [`solution_jacobian!`](@ref) for the in-place version.

Returns `(J, result)` where `result` is a [`SolveResult`](@ref).
"""
function solution_jacobian(f, lmo, x0, θ; kwargs...)
    n = length(x0)
    m = length(θ)
    J = zeros(promote_type(eltype(x0), eltype(θ)), n, m)
    return solution_jacobian!(J, f, lmo, x0, θ; kwargs...)
end

