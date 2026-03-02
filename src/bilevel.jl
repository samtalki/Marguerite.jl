"""
    bilevel_solve(outer_loss, f, ∇f!, lmo, x0, θ; kwargs...) -> (x_star, θ_grad)

Solve the inner problem and compute the gradient of `outer_loss(x*(θ))` w.r.t. `θ`.

Returns `(x_star, θ_grad)` where `x_star` is the inner solution and `θ_grad` is
``\\nabla_\\theta L(x^*(\\theta))``.

`outer_loss(x) -> Real` takes only the inner solution. If the user's outer loss
depends on `θ` directly, close over it and add the direct gradient manually.

# Differentiation keyword arguments
- `backend`: AD backend (default: `DEFAULT_BACKEND`)
- `diff_cg_maxiter::Int=50`: max CG iterations for the Hessian solve
- `diff_cg_tol::Real=1e-6`: CG convergence tolerance
- `diff_λ::Real=1e-4`: Tikhonov regularization for the Hessian

All other kwargs are forwarded to `solve`.
"""
function bilevel_solve(outer_loss, f, ∇f!::Function, lmo, x0, θ;
                       backend=DEFAULT_BACKEND,
                       diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_λ::Real=1e-4,
                       kwargs...)
    x_star, _ = solve(f, ∇f!, lmo, x0, θ; backend=backend, kwargs...)

    x̄ = DI.gradient(outer_loss, backend, x_star)

    ∇_x_f_of_θ(θ_) = begin
        T = promote_type(eltype(x_star), eltype(θ_))
        g = similar(x_star, T)
        ∇f!(g, x_star, θ_)
        return g
    end

    θ̄, _ = _implicit_pullback(f, ∇_x_f_of_θ, x_star, θ, x̄, backend;
                               cg_maxiter=diff_cg_maxiter, cg_tol=diff_cg_tol, cg_λ=diff_λ)
    return x_star, θ̄
end

"""
    bilevel_solve(outer_loss, f, lmo, x0, θ; kwargs...) -> (x_star, θ_grad)

Auto-gradient variant. Computes ``\\nabla_x f`` via AD.
"""
function bilevel_solve(outer_loss, f, lmo, x0, θ;
                       backend=DEFAULT_BACKEND,
                       diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_λ::Real=1e-4,
                       kwargs...)
    x_star, _ = solve(f, lmo, x0, θ; backend=backend, kwargs...)

    x̄ = DI.gradient(outer_loss, backend, x_star)

    ∇_x_f_of_θ(θ_) = begin
        f_of_x = x_ -> f(x_, θ_)
        return DI.gradient(f_of_x, backend, x_star)
    end

    θ̄, _ = _implicit_pullback(f, ∇_x_f_of_θ, x_star, θ, x̄, backend;
                               cg_maxiter=diff_cg_maxiter, cg_tol=diff_cg_tol, cg_λ=diff_λ)
    return x_star, θ̄
end

"""
    bilevel_gradient(outer_loss, f, ∇f!, lmo, x0, θ; kwargs...) -> θ_grad

Convenience wrapper: returns only the parameter gradient ``\\nabla_\\theta L(x^*(\\theta))``.
See [`bilevel_solve`](@ref) for full documentation.
"""
function bilevel_gradient(outer_loss, f, ∇f!::Function, lmo, x0, θ; kwargs...)
    _, θ̄ = bilevel_solve(outer_loss, f, ∇f!, lmo, x0, θ; kwargs...)
    return θ̄
end

"""
    bilevel_gradient(outer_loss, f, lmo, x0, θ; kwargs...) -> θ_grad

Auto-gradient variant. Returns only the parameter gradient.
"""
function bilevel_gradient(outer_loss, f, lmo, x0, θ; kwargs...)
    _, θ̄ = bilevel_solve(outer_loss, f, lmo, x0, θ; kwargs...)
    return θ̄
end
