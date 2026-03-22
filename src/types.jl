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

"""
    Result{T<:Real}

Immutable record of a Frank-Wolfe solve.

# Fields
- `objective::T` -- final objective value ``f(x^*)``
- `gap::T` -- final Frank-Wolfe duality gap
- `iterations::Int` -- iterations taken
- `converged::Bool` -- whether ``\\mathrm{gap} \\le \\mathrm{tol} \\cdot (1 + |f(x)|)``
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
- `residual_norm::T` -- final residual ``\\|r\\|``
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
Includes sparse vertex buffers used internally by fused LMO+gap computation.

Construct via `Cache{T}(n)` or let `solve` allocate one automatically.
"""
struct Cache{T<:Real}
    gradient::Vector{T}
    vertex::Vector{T}
    x_trial::Vector{T}
    direction::Vector{T}
    vertex_nzind::Vector{Int}   # sparse vertex index buffer (valid for [1:nnz] as returned by _lmo_and_gap!)
    vertex_nzval::Vector{T}     # sparse vertex value buffer (valid for [1:nnz] as returned by _lmo_and_gap!)

    function Cache{T}(gradient::Vector{T}, vertex::Vector{T},
                      x_trial::Vector{T}, direction::Vector{T},
                      vertex_nzind::Vector{Int}, vertex_nzval::Vector{T}) where {T<:Real}
        n = length(gradient)
        (length(vertex) == n && length(x_trial) == n && length(direction) == n) ||
            throw(DimensionMismatch(
                "Cache buffers must all have length $n (got $(length(gradient)), $(length(vertex)), $(length(x_trial)), $(length(direction)))"))
        (length(vertex_nzind) == n && length(vertex_nzval) == n) ||
            throw(DimensionMismatch(
                "Cache sparse buffers must have length $n (got vertex_nzind=$(length(vertex_nzind)), vertex_nzval=$(length(vertex_nzval)))"))
        new{T}(gradient, vertex, x_trial, direction, vertex_nzind, vertex_nzval)
    end
end

function Cache{T}(n::Int) where {T<:Real}
    n > 0 || throw(ArgumentError("Cache dimension must be positive, got n=$n"))
    Cache{T}(zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n),
             zeros(Int, n), zeros(T, n))
end
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

AdaptiveStepSize(L0::Real=1.0; eta=2.0) = AdaptiveStepSize(promote(Float64(L0), Float64(eta))...)

# ------------------------------------------------------------------
# Active set identification
# ------------------------------------------------------------------

"""
    ActiveConstraints{T}

Active constraint identification at a solution ``x^*``.

# Fields
- `bound_indices::Vector{Int}` -- indices pinned to bounds
- `bound_values::Vector{T}` -- their bound values
- `bound_is_lower::BitVector` -- `true` if bound is a lower bound, `false` if upper
- `free_indices::Vector{Int}` -- unconstrained variable indices
- `eq_normals` -- vector-like collection of equality constraint normals (in full space)
- `eq_rhs` -- equality constraint RHS values
"""
struct ActiveConstraints{T<:Real, EN<:AbstractVector, ER<:AbstractVector{T}}
    bound_indices::Vector{Int}
    bound_values::Vector{T}
    bound_is_lower::BitVector
    free_indices::Vector{Int}
    eq_normals::EN
    eq_rhs::ER

    function ActiveConstraints{T}(bound_indices, bound_values, bound_is_lower,
                          free_indices, eq_normals, eq_rhs) where T
        length(bound_indices) == length(bound_values) == length(bound_is_lower) ||
            throw(ArgumentError("ActiveConstraints: bound arrays must have equal length"))
        length(eq_normals) == length(eq_rhs) ||
            throw(ArgumentError("ActiveConstraints: eq_normals and eq_rhs must have equal length"))
        new{T, typeof(eq_normals), typeof(eq_rhs)}(
            bound_indices, bound_values, bound_is_lower, free_indices, eq_normals, eq_rhs)
    end
end

# ------------------------------------------------------------------
# Parametric oracles
# ------------------------------------------------------------------

"""
    ParametricOracle

Abstract type for oracles whose constraint set ``C(\\theta)`` depends on parameters.

Concrete subtypes hold parameter functions (``\\theta \\to`` constraint data).
Use [`materialize`](@ref) to instantiate a concrete oracle for a given ``\\theta``.
"""
abstract type ParametricOracle end

"""
    ParametricBox(lb_fn, ub_fn)

Parametric box

```math
C(\\theta) = \\{x : l(\\theta) \\le x \\le u(\\theta)\\}
```

- `lb_fn(θ) -> Vector`: lower bound function
- `ub_fn(θ) -> Vector`: upper bound function
"""
struct ParametricBox{LB, UB} <: ParametricOracle
    lb_fn::LB
    ub_fn::UB
end

"""
    ParametricSimplex{R, Equality}(r_fn)

Parametric simplex

```math
C(\\theta) = \\{x \\ge 0 : \\sum x_i \\le r(\\theta)\\}
```

(or ``= r(\\theta)`` when `Equality=true`).

- `r_fn(θ) -> scalar`: budget function
"""
struct ParametricSimplex{R, Equality} <: ParametricOracle
    r_fn::R
end

"""
    ParametricProbSimplex(r_fn)

Convenience constructor for `ParametricSimplex{R, true}` -- the parameterized
probability simplex

```math
\\{x \\ge 0 : \\sum x_i = r(\\theta)\\}
```
"""
ParametricProbSimplex(r_fn) = ParametricSimplex{typeof(r_fn), true}(r_fn)

"""
    ParametricSimplex(r_fn)

Convenience constructor for the capped (inequality) variant
`ParametricSimplex{R, false}` -- ``\\{x \\ge 0 : \\sum x_i \\le r(\\theta)\\}``.
"""
ParametricSimplex(r_fn) = ParametricSimplex{typeof(r_fn), false}(r_fn)

"""
    ParametricWeightedSimplex(α_fn, β_fn, lb_fn)

Parametric weighted simplex

```math
C(\\theta) = \\{x \\ge l(\\theta) : \\langle \\alpha(\\theta), x \\rangle \\le \\beta(\\theta)\\}
```

- `α_fn(θ) -> Vector`: cost coefficient function
- `β_fn(θ) -> scalar`: budget function
- `lb_fn(θ) -> Vector`: lower bound function
"""
struct ParametricWeightedSimplex{A, B, LB} <: ParametricOracle
    α_fn::A
    β_fn::B
    lb_fn::LB
end

# ------------------------------------------------------------------
# Wrapper types for solve / bilevel_solve output
# ------------------------------------------------------------------

"""
    SolveResult{T}

Wrapper for `solve` output. Supports tuple unpacking and pretty-printing.

`x, result = solve(...)` still works via `Base.iterate`.
Provides cleaner REPL display than a raw tuple.

# Fields
- `x::Vector{T}` -- optimal solution ``x^*``
- `result::Result{T}` -- convergence diagnostics
"""
struct SolveResult{T<:Real}
    x::Vector{T}
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
