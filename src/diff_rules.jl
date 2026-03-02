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
    _implicit_pullback(f, ∇_x_f_of_θ, x_star, θ, x̄, backend; cg_maxiter=50, cg_tol=1e-6, cg_λ=1e-4)

Shared pullback logic for implicit differentiation of `solve`.

1. Solves ``(\\nabla^2_{xx} f + \\lambda I) u = \\bar{x}`` via CG with HVPs.
2. Computes ``\\bar{\\theta} = -(\\partial(\\nabla_x f)/\\partial \\theta)^\\top u`` via the gradient of ``\\theta \\mapsto \\langle \\nabla_x f(\\theta), u \\rangle``.
"""
function _implicit_pullback(f, ∇_x_f_of_θ, x_star, θ, x̄, backend;
                            cg_maxiter::Int=50, cg_tol::Real=1e-6, cg_λ::Real=1e-4)
    fθ = x_ -> f(x_, θ)
    prep_hvp = DI.prepare_hvp(fθ, backend, x_star, (x̄,))
    hvp_fn = d -> DI.hvp(fθ, prep_hvp, backend, x_star, (d,))[1]
    u, cg_result = _cg_solve(hvp_fn, x̄ isa AbstractVector ? x̄ : collect(x̄);
                              maxiter=cg_maxiter, tol=cg_tol, λ=cg_λ)

    ∇f_dot_u = θ_ -> dot(∇_x_f_of_θ(θ_), u)
    prep_g = DI.prepare_gradient(∇f_dot_u, backend, θ)
    θ̄ = -DI.gradient(∇f_dot_u, prep_g, backend, θ)
    return θ̄, cg_result
end

# ------------------------------------------------------------------
# rrule: solve(f, ∇f!, lmo, x0, θ; backend, kwargs...)
# ------------------------------------------------------------------

"""
Implicit differentiation rule for `solve(f, ∇f!, lmo, x0, θ; ...)`.

At convergence ``x^*``, the optimality condition ``\\nabla_x f(x^*; \\theta) \\approx 0``
on the active face gives (via the implicit function theorem):

```math
\\frac{\\partial x^*}{\\partial \\theta} = -[\\nabla^2_{xx} f]^{-1} \\nabla^2_{x\\theta} f
```

The pullback computes:
1. ``u = [\\nabla^2_{xx} f + \\lambda I]^{-1} \\bar{x}`` via CG with HVPs
2. ``\\bar{\\theta} = -(\\partial(\\nabla_x f)/\\partial \\theta)^\\top u`` via AD
"""
function ChainRulesCore.rrule(::typeof(solve), f, ∇f!, lmo, x0, θ;
                              backend=DEFAULT_BACKEND,
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

        θ̄, _ = _implicit_pullback(f, ∇_x_f_of_θ, x_star, θ, x̄, backend;
                                   cg_maxiter=diff_cg_maxiter, cg_tol=diff_cg_tol, cg_λ=diff_λ)
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), θ̄
    end

    return (x_star, result), solve_pullback
end

# rrule for auto-gradient + θ variant
function ChainRulesCore.rrule(::typeof(solve), f, lmo, x0, θ;
                              backend=DEFAULT_BACKEND,
                              diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_λ::Real=1e-4,
                              kwargs...)
    x_star, result = solve(f, lmo, x0, θ; backend=backend, kwargs...)

    function solve_pullback(ȳ)
        x̄ = ȳ[1]

        if x̄ isa ChainRulesCore.AbstractZero
            return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end

        ∇_x_f_of_θ(θ_) = begin
            f_of_x = x_ -> f(x_, θ_)
            return DI.gradient(f_of_x, backend, x_star)
        end

        θ̄, _ = _implicit_pullback(f, ∇_x_f_of_θ, x_star, θ, x̄, backend;
                                   cg_maxiter=diff_cg_maxiter, cg_tol=diff_cg_tol, cg_λ=diff_λ)
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), θ̄
    end

    return (x_star, result), solve_pullback
end
