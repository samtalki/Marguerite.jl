"""
    Result{T<:Real}

Immutable record of a Frank-Wolfe solve.

# Fields
- `objective::T` -- final objective value ``f(x^*)``
- `gap::T` -- final Frank-Wolfe duality gap
- `iterations::Int` -- iterations taken
- `converged::Bool` -- whether `gap ≤ tol ⋅ |f(x)|`
- `discards::Int` -- rejected non-improving updates (monotonic mode)
"""
struct Result{T<:Real}
    objective::T
    gap::T
    iterations::Int
    converged::Bool
    discards::Int
end

"""
    CGResult{T<:Real}

Diagnostics from the conjugate gradient linear solve in implicit differentiation.

# Fields
- `iterations::Int` -- CG iterations taken
- `residual_norm::T` -- final residual `‖r‖`
- `converged::Bool` -- whether residual dropped below tolerance
"""
struct CGResult{T<:Real}
    iterations::Int
    residual_norm::T
    converged::Bool
end

"""
    Cache{T<:Real}

Pre-allocated working buffers for the Frank-Wolfe inner loop.

Construct via `Cache{T}(n)` or let `solve` allocate one automatically.
"""
struct Cache{T<:Real}
    gradient::Vector{T}
    vertex::Vector{T}
    x_trial::Vector{T}
end

Cache{T}(n::Int) where {T<:Real} = Cache{T}(zeros(T, n), zeros(T, n), zeros(T, n))
Cache(n::Int) = Cache{Float64}(n)

"""
    MonotonicStepSize()

Step size ``γ_t = 2/(t+2)``, yielding ``O(1/t)`` convergence on generalized
self-concordant objectives (Carderera, Besançon & Pokutta, 2024).
"""
struct MonotonicStepSize end

(::MonotonicStepSize)(t::Int) = 2.0 / (t + 2)

"""
    AdaptiveStepSize(L0::Real=1.0; η=2.0)

Backtracking line-search step size with Lipschitz estimation.

Starting from estimate `L`, doubles `L` until sufficient decrease holds,
then sets `γ = min(⟨∇f, x - v⟩ / (L ‖d‖²), 1)`.
"""
mutable struct AdaptiveStepSize{T<:Real}
    L::T
    η::T
end

AdaptiveStepSize(L0::Real=1.0; η=2.0) = AdaptiveStepSize(promote(Float64(L0), Float64(η))...)
