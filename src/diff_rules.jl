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

Conjugate gradient solver for `(H + λI)u = rhs` where `H` is accessed
only via Hessian-vector products `hvp_fn(d) -> Hd`.

Tikhonov regularization `λ` ensures well-conditioned systems near
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
    _implicit_pullback(f, ∇_x_f_of_θ, x_star, θ, x̄, backend, hvp_backend; cg_maxiter=50, cg_tol=1e-6, cg_λ=1e-4)

Shared pullback logic for implicit differentiation of `solve`.

1. Solves `(∇²ₓₓf + λI) u = x̄` via CG with HVPs using `hvp_backend`.
2. Computes `θ̄ = -(∂(∇_x f)/∂θ)ᵀ u` via the gradient of `θ ↦ ⟨∇_x f(θ), u⟩` using `backend`.

`backend` handles the cross-derivative gradient (∂/∂θ); `hvp_backend` handles
second-order Hessian-vector products (∇²ₓₓf). When `∇_x_f_of_θ` is computed by
a manual gradient (no internal AD), `backend` can be any mode. When `∇_x_f_of_θ`
uses reverse-mode AD internally (auto-gradient variant), the caller must ensure
`backend` doesn't create a reverse-over-reverse composition.

See [Implicit Differentiation](@ref) for the full derivation.
"""
function _implicit_pullback(f, ∇_x_f_of_θ, x_star, θ, x̄, backend, hvp_backend;
                            cg_maxiter::Int=50, cg_tol::Real=1e-6, cg_λ::Real=1e-4)
    fθ = x_ -> f(x_, θ)
    prep_hvp = DI.prepare_hvp(fθ, hvp_backend, x_star, (x̄,))
    hvp_fn = d -> DI.hvp(fθ, prep_hvp, hvp_backend, x_star, (d,))[1]
    u, cg_result = _cg_solve(hvp_fn, x̄ isa AbstractVector ? x̄ : collect(x̄);
                              maxiter=cg_maxiter, tol=cg_tol, λ=cg_λ)

    ∇f_dot_u = θ_ -> sum(∇_x_f_of_θ(θ_) .* u)
    prep_g = DI.prepare_gradient(∇f_dot_u, backend, θ)
    θ̄ = -DI.gradient(∇f_dot_u, prep_g, backend, θ)
    return θ̄, cg_result
end

"""
    _implicit_pullback_hvp(f, x_star, θ, x̄, hvp_backend; cg_maxiter=50, cg_tol=1e-6, cg_λ=1e-4)

Auto-gradient variant of `_implicit_pullback` that avoids nested AD.

Instead of differentiating through `∇_x f(x, θ)` (which would nest reverse-over-reverse),
computes the cross-derivative via a single HVP on the joint function `g(z) = f(z[1:n], z[n+1:end])`
where `z = [x; θ]`. The identity `∇²g · [u; 0] = [∇²_{xx}u; ∇²_{θx}u]` extracts the
cross-derivative as the last `m` entries.
"""
function _implicit_pullback_hvp(f, x_star, θ, x̄, hvp_backend;
                                 cg_maxiter::Int=50, cg_tol::Real=1e-6, cg_λ::Real=1e-4)
    # Step 1: CG solve  (H + λI)u = x̄  where H = ∇²_{xx} f(x*, θ)
    fθ = x_ -> f(x_, θ)
    prep_hvp = DI.prepare_hvp(fθ, hvp_backend, x_star, (x̄,))
    hvp_fn = d -> DI.hvp(fθ, prep_hvp, hvp_backend, x_star, (d,))[1]
    u, cg_result = _cg_solve(hvp_fn, x̄ isa AbstractVector ? x̄ : collect(x̄);
                              maxiter=cg_maxiter, tol=cg_tol, λ=cg_λ)

    # Step 2: Cross-derivative via joint HVP (no nested AD)
    # g(z) = f(z[1:n], z[n+1:end])  where z = [x; θ]
    # ∇²g · [u; 0] = [∇²_{xx}·u; ∇²_{θx}·u]
    # θ̄ = -∇²_{θx}·u
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
# rrule: solve(f, ∇f!, lmo, x0, θ; backend, kwargs...)
# ------------------------------------------------------------------

"""
Implicit differentiation rule for `solve(f, ∇f!, lmo, x0, θ; ...)`.

At convergence, `∂x*/∂θ = -[∇²ₓₓf]⁻¹ ∇²ₓθf` (implicit function theorem).

The pullback computes:
1. `u = [∇²ₓₓf + λI]⁻¹ x̄` via CG with HVPs (using `hvp_backend`)
2. `θ̄ = -(∂(∇_x f)/∂θ)ᵀ u` via AD (using `backend`)

# Keyword arguments
- `backend`: AD backend for first-order gradients (default: `DEFAULT_BACKEND`)
- `hvp_backend`: AD backend for Hessian-vector products (default: `SECOND_ORDER_BACKEND`)
- `diff_cg_maxiter`, `diff_cg_tol`, `diff_λ`: CG solver parameters

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

        ∇_x_f_of_θ(θ_) = begin
            T = promote_type(eltype(x_star), eltype(θ_))
            g = similar(x_star, T)
            ∇f!(g, x_star, θ_)
            return g
        end

        θ̄, _ = _implicit_pullback(f, ∇_x_f_of_θ, x_star, θ, x̄, backend, hvp_backend;
                                   cg_maxiter=diff_cg_maxiter, cg_tol=diff_cg_tol, cg_λ=diff_λ)
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

        θ̄, _ = _implicit_pullback_hvp(f, x_star, θ, x̄, hvp_backend;
                                       cg_maxiter=diff_cg_maxiter, cg_tol=diff_cg_tol, cg_λ=diff_λ)
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), θ̄
    end

    return (x_star, result), solve_pullback
end
