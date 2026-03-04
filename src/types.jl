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
    Result{T<:Real}

Immutable record of a Frank-Wolfe solve.

# Fields
- `objective::T` -- final objective value ``f(x^*)``
- `gap::T` -- final Frank-Wolfe duality gap
- `iterations::Int` -- iterations taken
- `converged::Bool` -- whether ``\\mathrm{gap} \\le \\mathrm{tol} \\cdot |f(x)|``
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

Construct via `Cache{T}(n)` or let `solve` allocate one automatically.
"""
struct Cache{T<:Real}
    gradient::Vector{T}
    vertex::Vector{T}
    x_trial::Vector{T}
    direction::Vector{T}
end

Cache{T}(n::Int) where {T<:Real} = Cache{T}(zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n))
Cache(n::Int) = Cache{Float64}(n)

"""
    MonotonicStepSize()

Step size

```math
\\gamma_t = \\frac{2}{t+2}
```

yielding ``O(1/t)`` convergence on generalized self-concordant objectives
(Carderera, Besançon & Pokutta, 2024).
"""
struct MonotonicStepSize end

(::MonotonicStepSize)(t::Int) = 2.0 / (t + 2)

"""
    AdaptiveStepSize(L0::Real=1.0; η=2.0)

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

AdaptiveStepSize(L0::Real=1.0; η=2.0) = AdaptiveStepSize(promote(Float64(L0), Float64(η))...)

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
- `eq_normals::Vector{Vector{T}}` -- equality constraint normals (in full space)
- `eq_rhs::Vector{T}` -- equality constraint RHS values
"""
struct ActiveConstraints{T}
    bound_indices::Vector{Int}
    bound_values::Vector{T}
    bound_is_lower::BitVector
    free_indices::Vector{Int}
    eq_normals::Vector{Vector{T}}
    eq_rhs::Vector{T}

    function ActiveConstraints{T}(bound_indices, bound_values, bound_is_lower,
                          free_indices, eq_normals, eq_rhs) where T
        length(bound_indices) == length(bound_values) == length(bound_is_lower) ||
            throw(ArgumentError("ActiveConstraints: bound arrays must have equal length"))
        length(eq_normals) == length(eq_rhs) ||
            throw(ArgumentError("ActiveConstraints: eq_normals and eq_rhs must have equal length"))
        new{T}(bound_indices, bound_values, bound_is_lower,
               free_indices, eq_normals, eq_rhs)
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
