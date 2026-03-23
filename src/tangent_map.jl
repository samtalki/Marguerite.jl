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

# ── TangentMap types and interface ───────────────────────────────────

"""
    _TangentMap{T}

Oracle-specific tangent space geometry for implicit differentiation.

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
`a_frees` and their squared norms `a_norm_sqs` enable null space projection:
removing components that would violate equality constraints.

# Fields
- `free::Vector{Int}` -- indices of free (non-bound) variables
- `bound::Vector{Int}` -- indices of variables pinned to constraint boundaries
- `a_frees::Vector{Vector{T}}` -- orthogonalized equality constraint normals restricted to free variables
- `a_frees_orig::Vector{Vector{T}}` -- original (pre-orthogonalization) constraint normals, retained for multiplier recovery
- `a_norm_sqs::Vector{T}` -- squared norms of orthogonalized normals
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
trace and PSD structure. Its dimension is `rank*(rank+1)/2 - 1` (trace zero
face block) plus `rank * nullity` (active×null cross block).

`G_uu` and `G_vv` are the objective's Hessian projected onto the active and
null eigenspaces. They appear in a curvature correction term: unlike polyhedral
constraints, moving along the active×null cross block incurs additional
curvature `B·G_vv - G_uu·B` from the cone boundary itself.

Remaining fields (`tmp_face`, `tmp_null`, `face`, `mixed`, `mixed_curv`, `full`,
`cross`) are pre-allocated work buffers for compress/expand/curvature operations.
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
free variables (equality constraints are handled via null space projection, not
dimensional reduction).
"""
@inline _reduced_dim(tm::_PolyhedralTangentMap) = length(tm.free)
@inline _reduced_dim(tm::_SpectralTangentMap) = _spectraplex_tangent_dim(size(tm.U, 2), size(tm.V_perp, 2))

"""
    _project_tangent!(out, v, tm::_TangentMap)

Project full space vector `v` into the tangent space, writing to `out`
(length [`_reduced_dim(tm)`](@ref)).

Polyhedral: extracts free variable components.
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

Expand tangent space vector `z` back to the full variable space, writing
to `out`.

Polyhedral: scatters into free variable positions, zeros elsewhere.
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

Polyhedral: null space projection — removes components along equality
constraint normals that are not truly free.
Spectral: adds the mixed curvature term from the PSD cone geometry.
At a rank deficient optimum, perturbing in the active×null cross block
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

# ── Factory ────────────────────────────────────────────────────────

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
