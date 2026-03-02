"""
    bilevel_solve(outer_loss, f, ∇f!, lmo, x0, θ; kwargs...) -> (x_star, θ_grad, cg_result)

Solve the inner problem and compute the gradient of `outer_loss(x*(θ))` w.r.t. `θ`.

Returns `(x_star, θ_grad, cg_result)` where `x_star` is the inner solution, `θ_grad` is
`∇_θ L(x*(θ))`, and `cg_result::CGResult` contains CG solver diagnostics.

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
    x_star, inner_result = solve(f, ∇f!, lmo, x0, θ; backend=backend, kwargs...)
    if !inner_result.converged
        @warn "inner solve did not converge (gap=$(inner_result.gap), iters=$(inner_result.iterations)): bilevel gradient may be inaccurate" maxlog=3
    end

    x̄ = DI.gradient(outer_loss, backend, x_star)

    ∇_x_f_of_θ(θ_) = begin
        T = promote_type(eltype(x_star), eltype(θ_))
        g = similar(x_star, T)
        ∇f!(g, x_star, θ_)
        return g
    end

    θ̄, cg_result = _implicit_pullback(f, ∇_x_f_of_θ, x_star, θ, x̄, backend;
                               cg_maxiter=diff_cg_maxiter, cg_tol=diff_cg_tol, cg_λ=diff_λ)
    return x_star, θ̄, cg_result
end

"""
    bilevel_solve(outer_loss, f, lmo, x0, θ; kwargs...) -> (x_star, θ_grad, cg_result)

Auto-gradient variant. Computes `∇_x f` via AD.
"""
function bilevel_solve(outer_loss, f, lmo, x0, θ;
                       backend=DEFAULT_BACKEND,
                       diff_cg_maxiter::Int=50, diff_cg_tol::Real=1e-6, diff_λ::Real=1e-4,
                       kwargs...)
    x_star, inner_result = solve(f, lmo, x0, θ; backend=backend, kwargs...)
    if !inner_result.converged
        @warn "inner solve did not converge (gap=$(inner_result.gap), iters=$(inner_result.iterations)): bilevel gradient may be inaccurate" maxlog=3
    end

    x̄ = DI.gradient(outer_loss, backend, x_star)

    ∇_x_f_of_θ(θ_) = begin
        f_of_x = x_ -> f(x_, θ_)
        return DI.gradient(f_of_x, backend, x_star)
    end

    θ̄, cg_result = _implicit_pullback(f, ∇_x_f_of_θ, x_star, θ, x̄, backend;
                               cg_maxiter=diff_cg_maxiter, cg_tol=diff_cg_tol, cg_λ=diff_λ)
    return x_star, θ̄, cg_result
end

"""
    bilevel_gradient(outer_loss, f, ∇f!, lmo, x0, θ; kwargs...) -> θ_grad

Convenience wrapper: returns only the parameter gradient `∇_θ L(x*(θ))`.
See [`bilevel_solve`](@ref) for full documentation.
"""
function bilevel_gradient(outer_loss, f, ∇f!::Function, lmo, x0, θ; kwargs...)
    _, θ̄, _ = bilevel_solve(outer_loss, f, ∇f!, lmo, x0, θ; kwargs...)
    return θ̄
end

"""
    bilevel_gradient(outer_loss, f, lmo, x0, θ; kwargs...) -> θ_grad

Auto-gradient variant. Returns only the parameter gradient.
"""
function bilevel_gradient(outer_loss, f, lmo, x0, θ; kwargs...)
    _, θ̄, _ = bilevel_solve(outer_loss, f, lmo, x0, θ; kwargs...)
    return θ̄
end
