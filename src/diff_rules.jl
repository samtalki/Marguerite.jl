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
    try
        m = which(active_set, Tuple{typeof(oracle), Vector{Float64}})
        sig = Base.unwrap_unionall(m.sig)
        return sig.parameters[2] !== Any
    catch e
        e isa MethodError || rethrow(e)
        return false
    end
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

        u, μ_bound, μ_eq, _ = _kkt_adjoint_solve_cached(state, dx)

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
