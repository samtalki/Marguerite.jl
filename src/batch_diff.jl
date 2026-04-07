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

# ── Batched rrule for batch_solve ──────────────────────────────────

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

    if lmo isa ParametricOracle
        oracle = materialize(lmo, θ)
    else
        oracle = lmo
    end

    # Build per-problem pullback states (one-time expensive setup)
    states = Vector{Any}(undef, B)
    for b in 1:B
        x_b = X_star_mat[:, b]
        # Per-problem scalar objective closure
        fθ_b = (x, θ_) -> f_batch(_col_to_batch(x, b, n, B), θ_)[b]
        grad_b = grad_batch !== nothing ? _make_batch_col_grad(grad_batch, b, n, B) : nothing
        states[b] = _build_pullback_state(fθ_b, hvp_backend, x_b, θ, oracle, tol;
                                           assume_interior=assume_interior,
                                           grad=grad_b,
                                           backend=backend,
                                           diff_lambda=diff_lambda)
    end

    # Pre-compute per-problem cross-derivative setup (manual or HVP path)
    _cross_data = Vector{Any}(undef, B)
    for b in 1:B
        x_b = X_star_mat[:, b]
        _n = n
        _m = m
        state = states[b]
        if grad_batch !== nothing
            grad_b = (g, x, θ_) -> begin
                Tg = promote_type(T, eltype(θ_))
                G_buf = zeros(Tg, _n, B)
                grad_batch(G_buf, _col_to_batch(x, b, _n, B), θ_)
                copyto!(g, @view(G_buf[:, b]))
            end
            _cross_data[b] = (:manual, _make_∇ₓf_of_θ(grad_b, x_b))
        else
            fθ_b = (x, θ_) -> f_batch(_col_to_batch(x, b, _n, B), θ_)[b]
            g_joint = z -> fθ_b(@view(z[1:_n]), @view(z[_n+1:end]))
            z = zeros(T, _n + _m)
            @views z[1:_n] .= x_b
            @views z[_n+1:end] .= θ
            v = zeros(T, _n + _m)
            _cross_data[b] = (:hvp, g_joint, z, v, DI.prepare_hvp(g_joint, hvp_backend, z, (v,)))
        end
    end

    # Pre-compute constraint multipliers for ParametricOracle
    λ_data = if lmo isa ParametricOracle
        map(1:B) do b
            x_b = X_star_mat[:, b]
            inner_b(x, θ_) = f_batch(_col_to_batch(x, b, n, B), θ_)[b]
            grad_b = if grad_batch !== nothing
                (g, x, θ_) -> begin
                    Tg = promote_type(T, eltype(θ_))
                    G_buf = zeros(Tg, n, B)
                    grad_batch(G_buf, _col_to_batch(x, b, n, B), θ_)
                    copyto!(g, @view(G_buf[:, b]))
                end
            else
                nothing
            end
            _primal_face_multipliers(inner_b, grad_b, x_b, θ, states[b].as, backend)
        end
    else
        nothing
    end

    function batch_solve_pullback(dy)
        dX = if dy isa Tuple
            dy[1]
        elseif dy isa BatchSolveResult
            dy.X
        else
            dy
        end

        if dX isa ChainRulesCore.AbstractZero
            return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end

        dθ_accum = zeros(T, m)

        for b in 1:B
            dx_b = dX isa AbstractMatrix ? dX[:, b] : zeros(T, n)
            state = states[b]

            u_b, μ_bound, μ_eq, _ = _kkt_adjoint_solve_cached(state, dx_b)

            # Cross-derivative for problem b
            cd = _cross_data[b]
            if cd[1] === :manual
                dθ_b = _cross_derivative_manual(cd[2], u_b, θ, backend)
            else
                # HVP path
                _, g_joint, z, v, prep = cd
                _n = n
                @views begin
                    v[1:_n] .= u_b
                    v[_n+1:end] .= 0
                end
                hvp_out = zeros(T, _n + m)
                DI.hvp!(g_joint, (hvp_out,), prep, hvp_backend, z, (v,))
                dθ_b = -copy(@view(hvp_out[_n+1:end]))
            end

            # Constraint sensitivity for ParametricOracle
            if lmo isa ParametricOracle && λ_data !== nothing
                x_b = X_star_mat[:, b]
                λ_bound, λ_eq = λ_data[b]
                dθ_con = _constraint_pullback(lmo, θ, x_b, u_b, μ_bound, μ_eq, λ_bound, λ_eq, state.as, backend)
                dθ_accum .+= dθ_b .+ dθ_con
            else
                dθ_accum .+= dθ_b
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
    oracle = lmo isa ParametricOracle ? materialize(lmo, θ) : lmo

    J = zeros(T, n * B, m)

    for b in 1:B
        x_b = X_star[:, b]
        fθ_b = (x, θ_) -> f_batch(_col_to_batch(x, b, n, B), θ_)[b]
        grad_b = grad_batch !== nothing ? _make_batch_col_grad(grad_batch, b, n, B) : nothing

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
