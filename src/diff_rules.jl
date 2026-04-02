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

Caches the active set, HVP preparation, tangent space geometry
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
                               tol=min(tol, ACTIVE_SET_TOL_CEILING),
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
    as_tol = min(tol, ACTIVE_SET_TOL_CEILING)
    if !result.converged && result.gap > 10 * as_tol
        @warn "rrule(solve): solver gap ($(result.gap)) >> active set tolerance ($as_tol); differentiation may be inaccurate. Consider tightening tol." maxlog=3
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

Supports [`ParametricOracle`](@ref) inputs: the constraint sensitivity
(how the feasible set changes with ``\\theta``) is included via bound shift
coupling and normal space equality displacement, requiring up to
``n_{\\text{bound}}`` additional HVPs (for oracle types with parametric bounds).
"""
function solution_jacobian!(J::AbstractMatrix, f, lmo, x0, θ;
                   grad=nothing, backend=DEFAULT_BACKEND,
                   hvp_backend=SECOND_ORDER_BACKEND,
                   diff_lambda::Real=1e-4, tol::Real=1e-4,
                   assume_interior::Bool=false, kwargs...)
    size(J) == (length(x0), length(θ)) ||
        throw(DimensionMismatch("J must be $(length(x0))×$(length(θ)), got $(size(J))"))
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

    # Constraint sensitivity for ParametricOracle
    if lmo isa ParametricOracle
        λ_bound, λ_eq = _primal_face_multipliers(f, grad, x_star, θ, state.as, backend)
        _add_constraint_jacobian!(J, lmo, state, x_star, θ, λ_bound, λ_eq, backend, hvp_backend)
    end

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

# ── Constraint Jacobian correction (ParametricOracle) ──────────────

"""
    _add_constraint_jacobian!(J, plmo, state, x_star, θ, λ_bound, λ_eq, backend, hvp_backend)

Add the constraint sensitivity contribution to the Jacobian `J` for a
`ParametricOracle`. Dispatches on the concrete oracle type to compute:

1. **Direct bound shifts**: ``J_{\\text{bound}_i, j} += \\partial b_i(\\theta)/\\partial\\theta_j``
2. **Coupling correction**: ``J -= P\\, H_{\\text{red}}^{-1}\\, \\Pi_\\perp(P^\\top H\\, B)``
   (bound shifts feed back into free variables through the Hessian;
   ``\\Pi_\\perp`` is the null-space projection removing equality-normal
   components, identity when no equality constraints are active)
3. **Normal-space equality displacement**: ``J_{\\text{free}, j} += (\\delta_j / \\|a\\|^2)\\, a``
   (constraint surface shift pushes ``x^*`` along the constraint normal)
"""
function _add_constraint_jacobian! end

"""
    _apply_bound_shifts!(J, B_bound, as)

Add direct bound shifts to `J`: ``J[i, j] += B_{\\text{bound}}[k, j]`` for each
active bound index ``i``.
"""
function _apply_bound_shifts!(J::AbstractMatrix{T}, B_bound::AbstractMatrix{T},
                               as::ActiveConstraints{T}) where T
    m = size(J, 2)
    @inbounds for (k, i) in enumerate(as.bound_indices)
        for j in 1:m
            J[i, j] += B_bound[k, j]
        end
    end
end

"""
    _coupling_correction!(J, C_con, state)

Null-project `C_con`, solve via the cached Hessian factorization, expand back
to full space and subtract from `J`. Shared by ParametricBox and
ParametricWeightedSimplex coupling correction.
"""
function _coupling_correction!(J::AbstractMatrix{T}, C_con::AbstractMatrix{T},
                                state::_PullbackState{T}) where T
    state.reduced_dim == 0 && return
    n, m = size(J)
    tm = state.tangent_map

    if tm isa _PolyhedralTangentMap
        for j in 1:m
            _null_project!(@view(C_con[:, j]), @view(C_con[:, j]),
                           tm.a_frees, tm.a_norm_sqs)
        end
    end

    ΔU = state.hessian_factor \ C_con

    col_buf = zeros(T, n)
    @inbounds for j in 1:m
        _expand_tangent!(col_buf, @view(ΔU[:, j]), tm)
        for i in 1:n
            J[i, j] -= col_buf[i]
        end
    end
end

"""
    _build_H_coupling(state, n_bound) -> H_coupling

Build the ``d \\times n_{\\text{bound}}`` matrix where column ``k`` is
``P^\\top H e_{\\text{bound}_k}`` (one HVP per active bound variable).
Shared by ParametricBox and ParametricWeightedSimplex.
"""
function _build_H_coupling(state::_PullbackState{T}, n_bound::Int) where T
    d = state.reduced_dim
    n = length(state.x_star)
    tm = state.tangent_map
    as = state.as

    H_coupling = zeros(T, d, n_bound)
    e_full = zeros(T, n)
    h_buf = zeros(T, n)
    proj_buf = zeros(T, d)
    @inbounds for (k, i) in enumerate(as.bound_indices)
        fill!(e_full, zero(T))
        e_full[i] = one(T)
        DI.hvp!(state.fθ, (h_buf,), state.prep_hvp, state.hvp_backend,
                state.x_star, (e_full,))
        _project_tangent!(proj_buf, h_buf, tm)
        H_coupling[:, k] .= proj_buf
    end
    return H_coupling
end

function _add_constraint_jacobian!(J::AbstractMatrix{T}, plmo::ParametricBox,
                                    state::_PullbackState{T}, _x_star, θ,
                                    _λ_bound, _λ_eq, backend, _hvp_backend) where T
    as = state.as
    d = state.reduced_dim
    n_bound = length(as.bound_indices)
    n_bound == 0 && return J

    m = size(J, 2)

    # Jacobians of bound functions w.r.t. θ
    prep_lb = DI.prepare_jacobian(plmo.lb_fn, backend, θ)
    lb_jac = DI.jacobian(plmo.lb_fn, prep_lb, backend, θ)
    prep_ub = DI.prepare_jacobian(plmo.ub_fn, backend, θ)
    ub_jac = DI.jacobian(plmo.ub_fn, prep_ub, backend, θ)

    # B_bound: shift of each active bound variable w.r.t. θ (n_bound × m)
    B_bound = zeros(T, n_bound, m)
    @inbounds for (k, i) in enumerate(as.bound_indices)
        if as.bound_is_lower[k]
            B_bound[k, :] .= @view(lb_jac[i, :])
        else
            B_bound[k, :] .= @view(ub_jac[i, :])
        end
    end

    _apply_bound_shifts!(J, B_bound, as)

    # Coupling correction in tangent space
    if d > 0
        H_coupling = _build_H_coupling(state, n_bound)
        C_con = H_coupling * B_bound
        _coupling_correction!(J, C_con, state)
    end

    return J
end

function _add_constraint_jacobian!(J::AbstractMatrix{T}, plmo::ParametricSimplex,
                                    state::_PullbackState{T}, _x_star, θ,
                                    _λ_bound, _λ_eq, backend, _hvp_backend) where T
    as = state.as
    tm = state.tangent_map

    # Only contributes when the budget equality is active
    isempty(as.eq_normals) && return J

    # ∇_θ r(θ)
    prep_r = DI.prepare_gradient(plmo.r_fn, backend, θ)
    dr = DI.gradient(plmo.r_fn, prep_r, backend, θ)

    # Normal-space displacement along equality normal in free subspace
    if tm isa _PolyhedralTangentMap
        a_free = tm.a_frees[1]
        a_norm_sq = tm.a_norm_sqs[1]
        max_norm_sq = maximum(tm.a_norm_sqs)
        if a_norm_sq < eps(T) * max_norm_sq
            @warn "constraint normal has near-zero free-space norm; skipping equality displacement" maxlog=3
            return J
        end
        @inbounds for j in 1:size(J, 2)
            coeff = dr[j] / a_norm_sq
            for (idx, i) in enumerate(tm.free)
                J[i, j] += coeff * a_free[idx]
            end
        end
    end

    return J
end

function _add_constraint_jacobian!(J::AbstractMatrix{T}, plmo::ParametricWeightedSimplex,
                                    state::_PullbackState{T}, x_star, θ,
                                    _λ_bound, λ_eq, backend, _hvp_backend) where T
    as = state.as
    tm = state.tangent_map
    d = state.reduced_dim
    n, m = size(J)
    n_bound = length(as.bound_indices)

    # B_bound: lower-bound shift (WeightedSimplex only has lower bounds)
    # Always defined so equality displacement can reference it safely.
    B_bound = zeros(T, n_bound, m)
    if n_bound > 0
        prep_lb = DI.prepare_jacobian(plmo.lb_fn, backend, θ)
        lb_jac = DI.jacobian(plmo.lb_fn, prep_lb, backend, θ)
        @inbounds for (k, i) in enumerate(as.bound_indices)
            B_bound[k, :] .= @view(lb_jac[i, :])
        end
        _apply_bound_shifts!(J, B_bound, as)
    end

    # Pre-compute α/β derivatives once (used by both stationarity and equality blocks).
    α_jac = zeros(T, 0, 0)
    dβ = T[]
    α_vals = T[]
    if !isempty(as.eq_normals)
        prep_α = DI.prepare_jacobian(plmo.α_fn, backend, θ)
        α_jac = DI.jacobian(plmo.α_fn, prep_α, backend, θ)
        prep_β = DI.prepare_gradient(plmo.β_fn, backend, θ)
        dβ = DI.gradient(plmo.β_fn, prep_β, backend, θ)
        α_vals = plmo.α_fn(θ)
    end

    # Coupling correction + stationarity correction in tangent space
    if d > 0
        C_con = zeros(T, d, m)

        # Coupling from bound shifts (n_bound HVPs)
        if n_bound > 0
            H_coupling = _build_H_coupling(state, n_bound)
            C_con .+= H_coupling * B_bound
        end

        # Stationarity correction from θ-dependent normals: λ_eq · P^T (∂α/∂θ_j)
        if !isempty(as.eq_normals) && !isempty(λ_eq)
            proj_buf_α = zeros(T, d)
            @inbounds for j in 1:m
                _project_tangent!(proj_buf_α, @view(α_jac[:, j]), tm)
                for i in 1:d
                    C_con[i, j] += λ_eq[1] * proj_buf_α[i]
                end
            end
        end

        _coupling_correction!(J, C_con, state)
    end

    # Normal-space equality displacement
    if !isempty(as.eq_normals) && tm isa _PolyhedralTangentMap
        a_free = tm.a_frees[1]
        a_norm_sq = tm.a_norm_sqs[1]
        max_norm_sq = maximum(tm.a_norm_sqs)
        if a_norm_sq < eps(T) * max_norm_sq
            @warn "constraint normal has near-zero free-space norm; skipping equality displacement" maxlog=3
            return J
        end
        @inbounds for j in 1:m
            # δ_j = ∂β/∂θ_j - ⟨∂α/∂θ_j, x*⟩ - ⟨α_bound, b_j_bound⟩
            δ_j = dβ[j] - dot(@view(α_jac[:, j]), x_star)
            for (k, i) in enumerate(as.bound_indices)
                δ_j -= α_vals[i] * B_bound[k, j]
            end
            coeff = δ_j / a_norm_sq
            for (idx, i) in enumerate(tm.free)
                J[i, j] += coeff * a_free[idx]
            end
        end
    end

    return J
end

# Fallback for unimplemented ParametricOracle subtypes
function _add_constraint_jacobian!(J, plmo::ParametricOracle, state, _x_star, θ,
                                    _λ_bound, _λ_eq, _backend, _hvp_backend)
    throw(ArgumentError(
        "_add_constraint_jacobian! not implemented for $(typeof(plmo)). " *
        "Implement Marguerite._add_constraint_jacobian!(...) to enable solution_jacobian with this oracle type."))
end

"""
    solution_jacobian(f, lmo, x0, θ; kwargs...) -> (J, result)

Compute the full Jacobian ``\\partial x^*/\\partial\\theta \\in \\mathbb{R}^{n \\times m}``
via direct reduced-Hessian factorization.

Forms the reduced Hessian ``P^\\top \\nabla^2 f\\, P`` explicitly (``n_{\\text{free}}``
HVPs), Cholesky-factors it once, then solves all ``m`` right-hand sides in one
shot. Much faster than ``m`` separate pullback calls for full Jacobians.

Supports [`ParametricOracle`](@ref) inputs: the constraint sensitivity is
included via bound shift coupling and normal space equality displacement.

See [`solution_jacobian!`](@ref) for the in-place version.

Returns `(J, result)` where `result` is a [`SolveResult`](@ref).
"""
function solution_jacobian(f, lmo, x0, θ; kwargs...)
    n = length(x0)
    m = length(θ)
    J = zeros(promote_type(eltype(x0), eltype(θ)), n, m)
    return solution_jacobian!(J, f, lmo, x0, θ; kwargs...)
end
