"""
    solve(f, ∇f!, lmo, x0; kwargs...) -> (x, Result)

Solve ``\\min_{x \\in \\mathcal{C}} f(x)`` via the Frank-Wolfe algorithm with
user-supplied gradient `∇f!(g, x)`.

# Arguments
- `f`: objective function `f(x) -> Real`
- `∇f!`: in-place gradient `∇f!(g, x)`, writing ``\\nabla f(x)`` into `g`
- `lmo`: linear minimization oracle (callable `lmo(v, g)` or `<: LinearOracle`)
- `x0`: initial feasible point (will be copied)

# Keyword Arguments
- `max_iters::Int = 1000`: maximum iterations
- `tol::Real = 1e-7`: convergence tolerance (gap ≤ tol * |f(x)|)
- `step_rule = MonotonicStepSize()`: step size rule (callable `t -> γ`)
- `monotonic::Bool = true`: reject non-improving updates
- `verbose::Bool = false`: print progress
- `cache::Union{Cache, Nothing} = nothing`: pre-allocated buffers

# Returns
`(x, result)` where `x` is the solution and `result::Result` holds diagnostics.
"""
function solve(f, ∇f!::Function, lmo, x0::AbstractVector;
               max_iters::Int=1000, tol::Real=1e-7,
               step_rule=MonotonicStepSize(), monotonic::Bool=true,
               verbose::Bool=false, cache::Union{Cache, Nothing}=nothing)
    x = copy(x0)
    T = eltype(x)
    n = length(x)
    c = something(cache, Cache{T}(n))

    obj = f(x)
    fw_gap = T(Inf)
    discards = 0
    converged = false
    reuse_grad = false
    final_iter = max_iters

    @inbounds for t in 0:(max_iters - 1)
        if !reuse_grad
            ∇f!(c.gradient, x)
        end

        lmo(c.vertex, c.gradient)

        # Frank-Wolfe gap: ⟨∇f, x - v⟩
        fw_gap = zero(T)
        for i in 1:n
            fw_gap += c.gradient[i] * (x[i] - c.vertex[i])
        end

        # Convergence check
        if fw_gap ≤ tol * abs(obj)
            converged = true
            final_iter = t
            break
        end

        γ = T(step_rule(t))

        # Trial point: x + γ(v - x)
        for i in 1:n
            c.x_trial[i] = x[i] + γ * (c.vertex[i] - x[i])
        end

        obj_trial = f(c.x_trial)

        if monotonic && obj_trial > obj + eps(T)
            reuse_grad = true
            discards += 1
            continue
        end

        for i in 1:n
            x[i] = c.x_trial[i]
        end
        obj = obj_trial
        reuse_grad = false

        if verbose && (t % 50 == 0 || t == max_iters - 1)
            println("t=$t | f(x)=$obj | gap=$fw_gap")
        end
    end

    return x, Result(obj, fw_gap, final_iter, converged, discards)
end

"""
    solve(f, lmo, x0; backend=DEFAULT_BACKEND, kwargs...) -> (x, Result)

Auto-gradient variant (no parameters). Computes ``\\nabla f`` via
`DifferentiationInterface.gradient!` using the specified `backend`.
"""
function solve(f, lmo, x0::AbstractVector;
               backend=DEFAULT_BACKEND,
               max_iters::Int=1000, tol::Real=1e-7,
               step_rule=MonotonicStepSize(), monotonic::Bool=true,
               verbose::Bool=false, cache::Union{Cache, Nothing}=nothing)
    x = copy(x0)
    T = eltype(x)
    n = length(x)
    c = something(cache, Cache{T}(n))
    prep = DI.prepare_gradient(f, backend, x)
    ∇f!_auto(g, x_) = DI.gradient!(f, g, prep, backend, x_)
    return solve(f, ∇f!_auto, lmo, x;
                 max_iters=max_iters, tol=tol, step_rule=step_rule,
                 monotonic=monotonic, verbose=verbose, cache=c)
end

# ------------------------------------------------------------------
# θ-parameterized variants (differentiable)
# ------------------------------------------------------------------

"""
    solve(f, ∇f!, lmo, x0, θ; kwargs...) -> (x, Result)

Solve ``\\min_{x \\in \\mathcal{C}} f(x, \\theta)`` with parameters `θ`.

Here `f(x, θ)` and `∇f!(g, x, θ)` accept θ as the second argument.
A `ChainRulesCore.rrule` is defined for this signature, enabling
``\\partial x^* / \\partial \\theta`` via implicit differentiation.
"""
function solve(f, ∇f!::Function, lmo, x0::AbstractVector, θ;
               backend=DEFAULT_BACKEND,
               max_iters::Int=1000, tol::Real=1e-7,
               step_rule=MonotonicStepSize(), monotonic::Bool=true,
               verbose::Bool=false, cache::Union{Cache, Nothing}=nothing)
    fθ(x) = f(x, θ)
    ∇fθ!(g, x) = ∇f!(g, x, θ)
    return solve(fθ, ∇fθ!, lmo, x0;
                 max_iters=max_iters, tol=tol, step_rule=step_rule,
                 monotonic=monotonic, verbose=verbose, cache=cache)
end

"""
    solve(f, lmo, x0, θ; backend=DEFAULT_BACKEND, kwargs...) -> (x, Result)

Auto-gradient + parameterized variant. Both the gradient and the implicit
differentiation use `backend`.
"""
function solve(f, lmo, x0::AbstractVector, θ;
               backend=DEFAULT_BACKEND,
               max_iters::Int=1000, tol::Real=1e-7,
               step_rule=MonotonicStepSize(), monotonic::Bool=true,
               verbose::Bool=false, cache::Union{Cache, Nothing}=nothing)
    fθ(x) = f(x, θ)
    x = copy(x0)
    T = eltype(x)
    n = length(x)
    c = something(cache, Cache{T}(n))
    prep = DI.prepare_gradient(fθ, backend, x)
    ∇fθ!_auto(g, x_) = DI.gradient!(fθ, g, prep, backend, x_)
    return solve(fθ, ∇fθ!_auto, lmo, x;
                 max_iters=max_iters, tol=tol, step_rule=step_rule,
                 monotonic=monotonic, verbose=verbose, cache=c)
end

# ------------------------------------------------------------------
# Adaptive step size logic
# ------------------------------------------------------------------

function (rule::AdaptiveStepSize)(t::Int, f, x, gradient, direction, obj)
    T = eltype(x)
    d_norm_sq = dot(direction, direction)
    if d_norm_sq < eps(T)
        return zero(T)
    end
    grad_dot_d = dot(gradient, direction)

    # Backtracking: find L such that sufficient decrease holds
    while true
        γ = clamp(-grad_dot_d / (rule.L * d_norm_sq), zero(T), one(T))
        x_trial = x .+ γ .* direction
        if f(x_trial) ≤ obj + γ * grad_dot_d + γ^2 * rule.L * d_norm_sq / 2
            break
        end
        rule.L *= rule.η
    end
    γ = clamp(-grad_dot_d / (rule.L * d_norm_sq), zero(T), one(T))
    rule.L = max(rule.L / rule.η, eps(T))  # relax for next iteration
    return γ
end
