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
    LinearOracle

Abstract supertype for Frank-Wolfe linear minimization oracles.

Every concrete oracle `lmo <: LinearOracle` is a callable struct invoked as
`lmo(v, g)`, writing the solution of ``\\min_{v \\in \\mathcal{C}} \\langle g, v \\rangle``
into `v` in-place.

Any plain function `(v, g) -> v` also works as an oracle -- no subtyping required.
"""
abstract type LinearOracle end

# ------------------------------------------------------------------
# 1. Simplex (unified: capped and probability)
# ------------------------------------------------------------------

"""
    Simplex{T, Equality}(r)

Oracle for simplex constraints. The type parameter `Equality` controls whether
the budget constraint is an inequality (``\\leq``) or equality (``=``).

- `Simplex(r)` / `Simplex(; r=1.0)`: capped simplex ``\\{x \\geq 0,\\; \\sum x_i \\leq r\\}``
- `ProbSimplex(r)` / `ProbabilitySimplex(r)`: probability simplex ``\\{x \\geq 0,\\; \\sum x_i = r\\}``

**Capped** (``Equality=false``): Vertices are ``\\{0, r e_1, \\ldots, r e_n\\}``.
Selects ``r e_{i^*}`` when ``g_{i^*} < 0``, otherwise the origin. ``O(n)``.

**Probability** (``Equality=true``): Vertices are ``\\{r e_1, \\ldots, r e_n\\}``.
Always selects ``r e_{i^*}`` where ``i^* = \\arg\\min_i g_i``. ``O(n)``.
"""
struct Simplex{T<:Real, Equality} <: LinearOracle
    r::T
end

Simplex(r::Real) = Simplex{Float64, false}(Float64(r))
Simplex(; r::Real=1.0) = Simplex{Float64, false}(Float64(r))

"""
    ProbSimplex(r=1.0)

Convenience constructor for `Simplex{T, true}(r)` -- the probability simplex
``\\{x \\geq 0,\\; \\sum x_i = r\\}``.
"""
ProbSimplex(r::Real) = Simplex{Float64, true}(Float64(r))
ProbSimplex(; r::Real=1.0) = Simplex{Float64, true}(Float64(r))

"""
    ProbabilitySimplex(r=1.0)

Alias for [`ProbSimplex`](@ref).
"""
ProbabilitySimplex(r::Real) = ProbSimplex(r)
ProbabilitySimplex(; r::Real=1.0) = ProbSimplex(; r=r)

function (lmo::Simplex{<:Real, Equality})(v::AbstractVector, g::AbstractVector) where {Equality}
    fill!(v, zero(eltype(v)))
    i_star = 1
    g_min = g[1]
    @inbounds for i in 2:length(g)
        if g[i] < g_min
            g_min = g[i]
            i_star = i
        end
    end
    if Equality || g_min < zero(g_min)
        @inbounds v[i_star] = lmo.r
    end
    return v
end

# ------------------------------------------------------------------
# 3. Knapsack
# ------------------------------------------------------------------

"""
    Knapsack(budget, m)

Oracle for the knapsack polytope ``\\mathcal{C} = \\{x \\in [0,1]^m :
\\sum_i x_i \\leq q\\}``.

Selects up to `budget` indices with most negative gradient and sets them to 1;
only indices with strictly negative gradient are selected.
``O(m + q \\log q)`` via `partialsortperm!`.
"""
struct Knapsack <: LinearOracle
    perm::Vector{Int}
    k::Int
end

function Knapsack(budget::Int, m::Int)
    budget < 0 && error("budget ($budget) must be ≥ 0")
    return Knapsack(collect(1:m), budget)
end

function (lmo::Knapsack)(v::AbstractVector, g::AbstractVector)
    fill!(v, zero(eltype(v)))
    if lmo.k <= 0
        return v
    end
    k = min(lmo.k, length(g))
    partialsortperm!(lmo.perm, g, 1:k)
    @inbounds for i in 1:k
        g[lmo.perm[i]] >= zero(eltype(g)) && break
        v[lmo.perm[i]] = one(eltype(v))
    end
    return v
end

# ------------------------------------------------------------------
# 3b. MaskedKnapsack
# ------------------------------------------------------------------

"""
    MaskedKnapsack(budget, masked, m)

Oracle for the knapsack polytope with masked indices fixed to 1:
``\\mathcal{C} = \\{x \\in [0,1]^m : \\sum_i x_i \\leq q,\\;
x_e = 1\\;\\forall e \\in \\text{masked}\\}``.

Fixes masked entries to 1, then selects up to ``k = q - |\\text{masked}|``
non-masked indices with most negative gradient; only indices with strictly
negative gradient are selected. ``O(m + k \\log k)`` via `partialsortperm!`.
"""
struct MaskedKnapsack <: LinearOracle
    is_masked::BitVector
    sel::Vector{Int}
    perm::Vector{Int}
    k::Int
end

function MaskedKnapsack(budget::Int, masked::AbstractVector{<:Integer}, m::Int)
    is_masked = falses(m)
    is_masked[masked] .= true
    sel = findall(.!is_masked)
    k = budget - length(masked)
    k < 0 && error("budget ($budget) must be ≥ |masked| ($(length(masked)))")
    return MaskedKnapsack(is_masked, sel, collect(1:length(sel)), k)
end

function (lmo::MaskedKnapsack)(v::AbstractVector, g::AbstractVector)
    fill!(v, zero(eltype(v)))
    @inbounds for i in eachindex(lmo.is_masked)
        if lmo.is_masked[i]
            v[i] = one(eltype(v))
        end
    end
    if lmo.k <= 0
        return v
    end
    k = min(lmo.k, length(lmo.sel))
    g_sel = @view(g[lmo.sel])
    partialsortperm!(lmo.perm, g_sel, 1:k)
    @inbounds for i in 1:k
        g_sel[lmo.perm[i]] >= zero(eltype(g)) && break
        v[lmo.sel[lmo.perm[i]]] = one(eltype(v))
    end
    return v
end

# ------------------------------------------------------------------
# 4. Box
# ------------------------------------------------------------------

"""
    Box(lb, ub)

Oracle for the box ``\\mathcal{C} = \\{x \\in \\mathbb{R}^n : \\ell_i \\leq x_i \\leq u_i\\}``.

Separable LP: ``v_i = \\ell_i`` if ``g_i \\geq 0``, else ``v_i = u_i``. ``O(n)``.
"""
struct Box{T<:Real} <: LinearOracle
    lb::Vector{T}
    ub::Vector{T}
end

Box(lb::AbstractVector, ub::AbstractVector) =
    Box{Float64}(collect(Float64, lb), collect(Float64, ub))

@inline function (lmo::Box)(v::AbstractVector, g::AbstractVector)
    @inbounds @simd for i in eachindex(v, g, lmo.lb, lmo.ub)
        v[i] = g[i] >= zero(g[i]) ? lmo.lb[i] : lmo.ub[i]
    end
    return v
end

# ------------------------------------------------------------------
# 5. WeightedSimplex
# ------------------------------------------------------------------

"""
    WeightedSimplex(α, β, lb)

Oracle for the weighted simplex ``\\mathcal{C} = \\{x \\in \\mathbb{R}^m : x \\geq \\ell,\\;
\\alpha^\\top x \\leq \\beta\\}``.

Shifts ``u = x - \\ell``, adjusted budget ``\\bar\\beta = \\beta - \\alpha^\\top \\ell``.
Then ``u^* = (\\bar\\beta / \\alpha_{i^*}) e_{i^*}`` where
``i^* = \\arg\\min_i \\{g_i / \\alpha_i : g_i < 0\\}``. Returns ``v = u^* + \\ell``.
``O(m)``.
"""
struct WeightedSimplex{T<:Real} <: LinearOracle
    α::Vector{T}
    β::T
    lb::Vector{T}
    β_bar::T  # precomputed: β - α'lb
end

function WeightedSimplex(α::AbstractVector{<:Real}, β::Real, lb::AbstractVector{<:Real})
    α_ = collect(Float64, α)
    lb_ = collect(Float64, lb)
    β_ = Float64(β)
    β_bar = β_ - dot(α_, lb_)
    return WeightedSimplex(α_, β_, lb_, β_bar)
end

function (lmo::WeightedSimplex)(v::AbstractVector, g::AbstractVector)
    copyto!(v, lmo.lb)

    if lmo.β_bar <= zero(lmo.β_bar)
        return v
    end

    best_ratio = typemax(eltype(g))
    best_idx = 0

    @inbounds for i in eachindex(g, lmo.α)
        if g[i] < zero(g[i])
            ratio = g[i] / lmo.α[i]
            if ratio < best_ratio
                best_ratio = ratio
                best_idx = i
            end
        end
    end

    if best_idx > 0
        @inbounds v[best_idx] = lmo.β_bar / lmo.α[best_idx] + lmo.lb[best_idx]
    end

    return v
end
