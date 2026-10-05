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

# ------------------------------------------------------------------
# Backend dispatch via KernelAbstractions
# ------------------------------------------------------------------
#
# Marguerite dispatches CPU vs GPU paths through `KernelAbstractions.get_backend(x)`.
# CUDA.jl, Metal.jl, AMDGPU.jl register their own KA backends through their own
# package extensions; Marguerite picks up whichever is loaded. Wrapper arrays
# (views, reshapes) inherit the parent's backend through KA's own `get_backend`
# recursion.

"""
    Result{T<:Real}

Immutable record of a Frank-Wolfe solve.

# Fields
- `objective::T` -- final objective value ``f(x^*)``
- `gap::T` -- Frank-Wolfe duality gap ``\\langle \\nabla f(x), x - v \\rangle`` at the returned iterate
- `iterations::Int` -- iterations taken (accepted and rejected steps)
- `converged::Bool` -- whether ``\\mathrm{gap} \\le \\mathrm{tol} \\cdot (1 + |f(x)|)``
  or ``\\mathrm{gap} \\le \\mathrm{rel\\_tol} \\cdot |f(x)|``
- `discards::Int` -- rejected non-improving updates (monotonic mode)
- `lower_bound::T` -- running maximum of ``f(x_t) - \\mathrm{gap}_t`` over the iterates
  whose gap was evaluated; a lower bound on ``\\min_{x \\in C} f(x)`` when ``f`` is convex
- `elapsed::Float64` -- wall-clock seconds from the start of the solve to its return
- `drop_steps::Int` -- accepted pairwise steps whose step size hit the largest feasible
  step ``\\gamma_{\\max}`` (`variant=:pairwise`; always `0` for plain Frank-Wolfe)
- `fallback_steps::Int` -- accepted steps of the pairwise variant that used the
  Frank-Wolfe direction because the largest feasible pairwise step was below
  `pairwise_atol` (always `0` for plain Frank-Wolfe)

The five-argument constructor `Result(objective, gap, iterations, converged, discards)`
sets `lower_bound = objective - gap` (or `-Inf` when either is not finite),
`elapsed = 0.0`, `drop_steps = 0` and `fallback_steps = 0`.
"""
struct Result{T<:Real}
    objective::T
    gap::T
    iterations::Int
    converged::Bool
    discards::Int
    lower_bound::T
    elapsed::Float64
    drop_steps::Int
    fallback_steps::Int
end

function Result(objective::T, gap::T, iterations::Integer, converged::Bool,
                discards::Integer) where {T<:Real}
    lb = _lower_bound_update(T(-Inf), objective, gap)
    return Result{T}(objective, gap, Int(iterations), converged, Int(discards), lb, 0.0, 0, 0)
end

# Running lower bound max(lb, obj - gap). Non-finite values are skipped so a
# corrupted objective or gap cannot produce a spurious bound.
@inline function _lower_bound_update(lb::T, obj, gap) where {T<:Real}
    (isfinite(obj) && isfinite(gap)) || return lb
    return max(lb, T(obj - gap))
end

"""
    CGResult{T<:Real}

Diagnostics from the linear solve in implicit differentiation.

For the iterative `bilevel_solve` path, all three fields reflect the
inner CG. For the cached `rrule(solve)` direct-factorization path,
`iterations` is `0` and `residual_norm` carries the relative
factorization residual (``\\lambda \\|u\\| / \\|b\\|``); `converged` is
`true` if that residual is below the Tikhonov retry threshold.

# Fields
- `iterations::Int` -- CG iterations (0 for direct-solve path)
- `residual_norm::T` -- residual norm
- `converged::Bool` -- residual below the acceptance threshold
"""
struct CGResult{T<:Real}
    iterations::Int
    residual_norm::T
    converged::Bool
end

"""
    Cache{T<:Real}

Pre-allocated working buffers for the Frank-Wolfe inner loop.
Includes sparse vertex buffers used internally by fused LMO+gap computation,
`gradient_trial`, which holds the gradient at a trial point for step rules
that evaluate it ([`SecantLineSearch`](@ref)), and `away_vertex`, which
holds the away vertex of the pairwise variant (see [`away_vertex!`](@ref)).

Construct via `Cache{T}(n)` or let `solve` allocate one automatically.
"""
struct Cache{T<:Real, V<:AbstractVector{T}}
    gradient::V
    vertex::V
    x_trial::V
    direction::V
    vertex_nzind::Vector{Int}   # always CPU — scalar indexing in sparse vertex protocol
    vertex_nzval::Vector{T}     # always CPU — scalar indexing in sparse vertex protocol
    gradient_trial::V
    away_vertex::V

    function Cache{T,V}(gradient::V, vertex::V,
                        x_trial::V, direction::V,
                        vertex_nzind::Vector{Int}, vertex_nzval::Vector{T},
                        gradient_trial::V, away_vertex::V) where {T<:Real, V<:AbstractVector{T}}
        n = length(gradient)
        (length(vertex) == n && length(x_trial) == n && length(direction) == n &&
         length(gradient_trial) == n && length(away_vertex) == n) ||
            throw(DimensionMismatch(
                "Cache buffers must all have length $n (got $(length(gradient)), $(length(vertex)), $(length(x_trial)), $(length(direction)), $(length(gradient_trial)), $(length(away_vertex)))"))
        (length(vertex_nzind) == n && length(vertex_nzval) == n) ||
            throw(DimensionMismatch(
                "Cache sparse buffers must have length $n (got vertex_nzind=$(length(vertex_nzind)), vertex_nzval=$(length(vertex_nzval)))"))
        new{T,V}(gradient, vertex, x_trial, direction, vertex_nzind, vertex_nzval,
                 gradient_trial, away_vertex)
    end
end

# Six-buffer form: allocates the trial-gradient and away-vertex buffers like `gradient`.
function Cache{T,V}(gradient::V, vertex::V, x_trial::V, direction::V,
                    vertex_nzind::Vector{Int}, vertex_nzval::Vector{T}) where {T<:Real, V<:AbstractVector{T}}
    return Cache{T,V}(gradient, vertex, x_trial, direction, vertex_nzind, vertex_nzval,
                      fill!(similar(gradient), zero(T)), fill!(similar(gradient), zero(T)))
end

function Cache{T}(n::Int) where {T<:Real}
    n > 0 || throw(ArgumentError("Cache dimension must be positive, got n=$n"))
    vecs = ntuple(_ -> zeros(T, n), Val(4))
    Cache{T, Vector{T}}(vecs..., zeros(Int, n), zeros(T, n), zeros(T, n), zeros(T, n))
end

"""
    Cache(x0::AbstractVector{T})

Construct a `Cache` whose main buffers match the array type of `x0`.
Sparse vertex buffers are always CPU `Vector`.
"""
function Cache(x0::AbstractVector{T}) where {T<:Real}
    n = length(x0)
    n > 0 || throw(ArgumentError("Cache dimension must be positive, got n=$n"))
    _zl(x) = fill!(similar(x), zero(T))
    V = typeof(_zl(x0))
    Cache{T, V}(_zl(x0), _zl(x0), _zl(x0), _zl(x0), zeros(Int, n), zeros(T, n), _zl(x0), _zl(x0))
end

"""
    Cache(n)

Convenience constructor for `Cache{Float64}(n)`.
"""
Cache(n::Int) = Cache{Float64}(n)

"""
    MonotonicStepSize()

The standard Frank-Wolfe step size

```math
\\gamma_t = \\frac{2}{t+2}
```

yielding ``O(1/t)`` convergence. Carderera, Besançon & Pokutta (2024) establish
this rate for generalized self-concordant objectives.
"""
struct MonotonicStepSize end

(::MonotonicStepSize)(t::Int) = 2.0 / (t + 2)

"""
    AdaptiveStepSize(L0::Real=1.0; eta=2.0)

Backtracking line-search step size with Lipschitz estimation.

Starting from estimate ``L``, multiplies ``L`` by ``\\eta`` until sufficient decrease holds,
then sets

```math
\\gamma = \\mathrm{clamp}\\!\\left(\\frac{\\langle \\nabla f,\\, x - v \\rangle}{L \\,\\|d\\|^2},\\; 0,\\; 1\\right)
```
"""
mutable struct AdaptiveStepSize{T<:Real}
    L::T
    η::T
end

function AdaptiveStepSize(L0::Real=1.0; eta::Real=2.0)
    L0f, etaf = promote(float(L0), float(eta))
    return AdaptiveStepSize(L0f, etaf)
end

"""
    ShortStep(L::Real)

Short step with a fixed smoothness constant ``L``. With ``d = v - x``,

```math
\\gamma = \\mathrm{clamp}\\!\\left(\\frac{-\\langle \\nabla f(x),\\, d \\rangle}{L \\,\\|d\\|^2},\\; 0,\\; 1\\right),
```

the minimizer over ``[0, 1]`` of the quadratic upper bound
``f(x) + \\gamma \\langle \\nabla f(x), d \\rangle + \\tfrac{L}{2} \\gamma^2 \\|d\\|^2``.
When ``L`` is a valid Lipschitz constant of ``\\nabla f``, every step decreases ``f``
and the ``O(1/t)`` rate holds. Unlike [`AdaptiveStepSize`](@ref) it never
evaluates ``f`` while choosing ``\\gamma`` and never changes ``L``. On a quadratic
with Hessian ``H``, `ShortStep(L)` with ``L = d^\\top H d / \\|d\\|^2`` lands on
the exact minimizer along ``d``.

CPU only. Not supported by `batch_solve`.
"""
struct ShortStep{T<:Real}
    L::T
    function ShortStep{T}(L::T) where {T<:Real}
        (isfinite(L) && L > zero(T)) ||
            throw(ArgumentError("ShortStep: L must be positive and finite, got L=$L"))
        new{T}(L)
    end
end

ShortStep(L::Real) = (Lf = float(L); ShortStep{typeof(Lf)}(Lf))

"""
    SecantLineSearch(; max_trials=4, γ0=0.5)
    SecantLineSearch(max_trials)

Line search on ``\\varphi(\\gamma) = f(x + \\gamma d)`` along ``d = v - x``, for
convex ``f``, with at most `max_trials` trial points per iteration. Let
``s = -\\langle \\nabla f(x), d \\rangle > 0`` be the initial slope magnitude.

1. Trial 1 at ``\\gamma = \\min(2\\gamma_{\\text{prev}}, 1)``, where ``\\gamma_{\\text{prev}}``
   is the step this rule chose at the previous iteration (`γ0` at the first). Evaluate ``f`` and
   ``\\nabla f`` there and the slope ``\\varphi'(\\gamma) = \\langle \\nabla f(x + \\gamma d), d \\rangle``.
   If ``\\varphi'(\\gamma) \\le 0`` the minimizer along ``d`` lies beyond ``\\gamma``: accept.
2. Otherwise take the secant step ``\\gamma \\leftarrow \\gamma\\, s / (s + \\varphi'(\\gamma))``,
   the zero of the linear interpolant of ``\\varphi'`` on ``[0, \\gamma]``, and accept it
   if ``f`` does not rise.
3. While ``f`` rises, halve ``\\gamma`` (trials 3 to `max_trials`).
4. If every trial fails, fall back to the open-loop step ``2/(t+2)``, which the
   solver's monotone check then accepts or rejects.

On a quadratic, ``\\varphi'`` is affine, so step 2 lands on the exact minimizer
along ``d``. Each trial costs one evaluation of ``f``; trial 1 also costs one
gradient, which the solver reuses as the next gradient when trial 1 is accepted.
The rule is mutable: it remembers ``\\gamma_{\\text{prev}}`` and resets it to `γ0`
at the first iteration of every solve.

CPU only. Not supported by `batch_solve`.
"""
mutable struct SecantLineSearch{T<:Real}
    max_trials::Int
    γ0::T
    γ_prev::T
    function SecantLineSearch{T}(max_trials::Integer, γ0::T) where {T<:Real}
        max_trials ≥ 1 ||
            throw(ArgumentError("SecantLineSearch: max_trials must be ≥ 1, got $max_trials"))
        (zero(T) < γ0 ≤ one(T)) ||
            throw(ArgumentError("SecantLineSearch: γ0 must lie in (0, 1], got $γ0"))
        new{T}(Int(max_trials), γ0, γ0)
    end
end

function SecantLineSearch(max_trials::Integer; γ0::Real=0.5)
    g = float(γ0)
    return SecantLineSearch{typeof(g)}(max_trials, g)
end
SecantLineSearch(; max_trials::Integer=4, γ0::Real=0.5) = SecantLineSearch(max_trials; γ0=γ0)

# ------------------------------------------------------------------
# Wrapper types for solve / bilevel_solve output
# ------------------------------------------------------------------

"""
    SolveResult{T, V<:AbstractVector{T}}

Wrapper for `solve` output. Supports tuple unpacking and pretty-printing.

`x, result = solve(...)` still works via `Base.iterate`.
Provides cleaner REPL display than a raw tuple.

# Fields
- `x::AbstractVector{T}` -- optimal solution ``x^*``
- `result::Result{T}` -- convergence diagnostics
"""
struct SolveResult{T<:Real, V<:AbstractVector{T}}
    x::V
    result::Result{T}
end

Base.iterate(sr::SolveResult) = (sr.x, Val(:result))
Base.iterate(sr::SolveResult, ::Val{:result}) = (sr.result, nothing)
Base.iterate(::SolveResult, ::Nothing) = nothing
Base.length(::SolveResult) = 2
Base.IteratorSize(::Type{<:SolveResult}) = Base.HasLength()

"""
    BilevelResult{T, S}

Wrapper for `bilevel_solve` output. Supports tuple unpacking and pretty-printing.

`x, θ_grad, cg_result = bilevel_solve(...)` still works via `Base.iterate`.

# Fields
- `x::Vector{T}` -- inner problem solution ``x^*(\\theta)``
- `theta_grad::S` -- gradient ``\\nabla_\\theta L(x^*(\\theta))``
- `cg_result::CGResult{T}` -- CG solver diagnostics
"""
struct BilevelResult{T<:Real, S}
    x::Vector{T}
    theta_grad::S
    cg_result::CGResult{T}
end

Base.iterate(br::BilevelResult) = (br.x, Val(:tg))
Base.iterate(br::BilevelResult, ::Val{:tg}) = (br.theta_grad, Val(:cg))
Base.iterate(br::BilevelResult, ::Val{:cg}) = (br.cg_result, nothing)
Base.iterate(::BilevelResult, ::Nothing) = nothing
Base.length(::BilevelResult) = 3
Base.IteratorSize(::Type{<:BilevelResult}) = Base.HasLength()
