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

# ── Per-problem cross-derivative state ─────────────────────────────

struct _ManualCross{F}
    grad_fn::F
end

struct _HVPCross{G, T, P}
    g_joint::G
    z::Vector{T}
    v::Vector{T}
    prep::P
    hvp_out::Vector{T}
end

function _build_cross_manual(grad_b, x_b)
    _ManualCross(_make_∇ₓf_of_θ(grad_b, x_b))
end

function _build_cross_hvp(fθ_b, x_b::AbstractVector{T}, θ, hvp_backend) where T
    n, m = length(x_b), length(θ)
    g_joint = z -> fθ_b(@view(z[1:n]), @view(z[n+1:end]))
    z = zeros(T, n + m)
    @views z[1:n] .= x_b
    @views z[n+1:end] .= θ
    v = zeros(T, n + m)
    prep = DI.prepare_hvp(g_joint, hvp_backend, z, (v,))
    _HVPCross(g_joint, z, v, prep, zeros(T, n + m))
end

function _cross_dθ(c::_ManualCross, u_b, θ, backend, ::Any)
    _cross_derivative_manual(c.grad_fn, u_b, θ, backend)
end

function _cross_dθ(c::_HVPCross, u_b, θ, ::Any, hvp_backend)
    n = length(u_b)
    @views begin
        c.v[1:n] .= u_b
        c.v[n+1:end] .= 0
    end
    DI.hvp!(c.g_joint, (c.hvp_out,), c.prep, hvp_backend, c.z, (c.v,))
    return -copy(@view(c.hvp_out[n+1:end]))
end

# ── dY cotangent unwrap ────────────────────────────────────────────

_unwrap_dy(dy::Tuple) = dy[1]
_unwrap_dy(dy::BatchSolveResult) = dy.X
_unwrap_dy(dy) = dy

# ── Constraint sensitivity dispatch ────────────────────────────────

_face_multipliers(::AbstractOracle, _fθ, _grad, _x, _θ, _as, _backend) = nothing
function _face_multipliers(lmo::ParametricOracle, fθ_b, grad_b, x_b, θ, as, backend)
    _primal_face_multipliers(fθ_b, grad_b, x_b, θ, as, backend)
end

_add_constraint_dθ!(dθ_accum, ::AbstractOracle, _θ, _x, _u, _μb, _μe, _λ, _as, _backend) = dθ_accum
function _add_constraint_dθ!(dθ_accum, lmo::ParametricOracle, θ, x_b, u_b, μ_bound, μ_eq, λ_data_b, as, backend)
    λ_bound, λ_eq = λ_data_b
    dθ_con = _constraint_pullback(lmo, θ, x_b, u_b, μ_bound, μ_eq, λ_bound, λ_eq, as, backend)
    dθ_accum .+= dθ_con
end

# ── Batched rrule ──────────────────────────────────────────────────

"""
Implicit differentiation rule for `batch_solve(f_batch, lmo, X0, θ; ...)`.

Solves `B` independent forward problems, builds per-problem
[`_PullbackState`](@ref) objects (one-time), then returns a pullback
closure that computes ``\\partial\\theta`` by summing per-problem KKT
adjoint contributions.

The returned pullback accepts `dY::AbstractMatrix` (the cotangent of
the `(n, B)` solution matrix) and returns a 5-tuple with `dθ` in the
last position (matching the 5 positional arguments:
`batch_solve, f_batch, lmo, X0, θ`).
"""
function ChainRulesCore.rrule(::typeof(batch_solve), f_batch, lmo, X0, θ;
                              grad_batch=nothing,
                              backend=DEFAULT_BACKEND,
                              hvp_backend=SECOND_ORDER_BACKEND,
                              diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_lambda::Real=1e-4,
                              assume_interior::Bool=false,
                              tol::Real=1e-4,
                              kwargs...)
    X_star_mat, result = batch_solve(f_batch, lmo, X0, θ;
                                      grad_batch=grad_batch, backend=backend,
                                      assume_interior=assume_interior,
                                      tol=tol, kwargs...)
    T = eltype(X_star_mat)
    n, B = size(X_star_mat)
    m = length(θ)

    as_tol = min(tol, ACTIVE_SET_TOL_CEILING)
    if !all(result.converged)
        n_unc = count(.!result.converged)
        max_gap = maximum(result.gaps)
        if max_gap > 10 * as_tol
            @warn "rrule(batch_solve): $n_unc problems not converged (max gap=$max_gap >> active set tol=$as_tol); differentiation may be inaccurate" maxlog=3
        end
    end

    oracle = _to_oracle(lmo, θ)

    # Per-problem closures: build once, reuse across state, cross-data, λ-data.
    fθ_bs   = [_make_batch_col_fn_θ(f_batch, b, n, B; X_template=X_star_mat) for b in 1:B]
    grad_bs = grad_batch === nothing ? nothing :
              [_make_batch_col_grad(grad_batch, b, n, B; X_template=X_star_mat) for b in 1:B]

    states = [
        _build_pullback_state(fθ_bs[b], hvp_backend, X_star_mat[:, b], θ, oracle, tol;
                              assume_interior=assume_interior,
                              grad=(grad_bs === nothing ? nothing : grad_bs[b]),
                              backend=backend,
                              diff_lambda=diff_lambda)
        for b in 1:B
    ]

    crosses = if grad_bs !== nothing
        [_build_cross_manual(grad_bs[b], X_star_mat[:, b]) for b in 1:B]
    else
        [_build_cross_hvp(fθ_bs[b], X_star_mat[:, b], θ, hvp_backend) for b in 1:B]
    end

    λ_data = if lmo isa ParametricOracle
        [_face_multipliers(lmo, fθ_bs[b],
                           grad_bs === nothing ? nothing : grad_bs[b],
                           X_star_mat[:, b], θ, states[b].as, backend) for b in 1:B]
    else
        nothing
    end

    function batch_solve_pullback(dy)
        dX = _unwrap_dy(dy)
        if dX isa ChainRulesCore.AbstractZero
            return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end

        dθ_accum = zeros(T, m)

        for b in 1:B
            dx_b = @view(dX[:, b])
            state = states[b]

            u_b, μ_bound, μ_eq, _ = _kkt_adjoint_solve_cached(state, dx_b)

            dθ_b = _cross_dθ(crosses[b], u_b, θ, backend, hvp_backend)
            dθ_accum .+= dθ_b

            if λ_data !== nothing
                _add_constraint_dθ!(dθ_accum, lmo, θ, @view(X_star_mat[:, b]),
                                    u_b, μ_bound, μ_eq, λ_data[b], state.as, backend)
            end
        end

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), dθ_accum
    end

    return BatchSolveResult(X_star_mat, result), batch_solve_pullback
end

"""
    batch_solution_jacobian(f_batch, lmo, X0, θ; kwargs...) -> (J, BatchSolveResult)

Compute the Jacobian ``\\partial X^*/\\partial\\theta \\in \\mathbb{R}^{nB \\times m}``
for the batched parametric solve.

The Jacobian is organized as a `(n*B, m)` matrix where rows `(b-1)*n+1 : b*n`
correspond to problem `b`. Uses cached reduced Hessian factorization per
problem for efficiency.
"""
function batch_solution_jacobian(f_batch, lmo, X0, θ;
                                  grad_batch=nothing, backend=DEFAULT_BACKEND,
                                  hvp_backend=SECOND_ORDER_BACKEND,
                                  diff_lambda::Real=1e-4, tol::Real=1e-4,
                                  assume_interior::Bool=false, kwargs...)
    X_star, result = batch_solve(f_batch, lmo, X0, θ;
                                  grad_batch=grad_batch, backend=backend,
                                  tol=tol, kwargs...)
    T = eltype(X_star)
    n, B = size(X_star)
    m = length(θ)
    oracle = _to_oracle(lmo, θ)

    J = zeros(T, n * B, m)

    for b in 1:B
        x_b = X_star[:, b]
        fθ_b = _make_batch_col_fn_θ(f_batch, b, n, B; X_template=X_star)
        grad_b = grad_batch !== nothing ?
            _make_batch_col_grad(grad_batch, b, n, B; X_template=X_star) : nothing

        state = _build_pullback_state(fθ_b, hvp_backend, x_b, θ, oracle, tol;
                                       assume_interior=assume_interior,
                                       grad=grad_b, backend=backend,
                                       diff_lambda=diff_lambda)
        J_b = @view(J[(b-1)*n+1 : b*n, :])
        _solution_jacobian_impl!(J_b, state, fθ_b, grad_b, x_b, θ, backend, hvp_backend)

        if lmo isa ParametricOracle
            λ_bound, λ_eq = _primal_face_multipliers(fθ_b, grad_b, x_b, θ, state.as, backend)
            _add_constraint_jacobian!(J_b, lmo, state, x_b, θ, λ_bound, λ_eq, backend, hvp_backend)
        end
    end

    return J, BatchSolveResult(X_star, result)
end
