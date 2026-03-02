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

# Capped simplex: origin is a vertex, so check if g_min < 0
function (lmo::Simplex{T, false})(v::AbstractVector, g::AbstractVector) where T
    fill!(v, zero(eltype(v)))
    i_star = 1
    g_min = g[1]
    @inbounds for i in 2:length(g)
        if g[i] < g_min
            g_min = g[i]
            i_star = i
        end
    end
    if g_min < zero(g_min)
        @inbounds v[i_star] = lmo.r
    end
    return v
end

# Probability simplex: equality constraint, always pick a vertex
function (lmo::Simplex{T, true})(v::AbstractVector, g::AbstractVector) where T
    fill!(v, zero(eltype(v)))
    i_star = 1
    g_min = g[1]
    @inbounds for i in 2:length(g)
        if g[i] < g_min
            g_min = g[i]
            i_star = i
        end
    end
    @inbounds v[i_star] = lmo.r
    return v
end

# ------------------------------------------------------------------
# 3. Knapsack
# ------------------------------------------------------------------

"""
    Knapsack(budget, backbone, m)

Oracle for the knapsack polytope ``\\mathcal{C} = \\{x \\in [0,1]^m :
\\sum_i x_i \\leq q,\\; x_e = 1\\;\\forall e \\in \\mathcal{B}\\}``.

Fixes backbone entries to 1, then selects the ``k = q - |\\mathcal{B}|`` non-backbone
indices with most negative gradient. ``O(m \\log k)`` via `partialsortperm`.
"""
struct Knapsack <: LinearOracle
    budget::Int
    mask::BitVector
    sel::Vector{Int}
    k::Int
end

function Knapsack(budget::Int, backbone::AbstractVector{<:Integer}, m::Int)
    mask = trues(m)
    mask[backbone] .= false
    sel = findall(mask)
    k = budget - length(backbone)
    k < 0 && error("budget ($budget) must be ≥ |backbone| ($(length(backbone)))")
    return Knapsack(budget, mask, sel, k)
end

function (lmo::Knapsack)(v::AbstractVector, g::AbstractVector)
    fill!(v, zero(eltype(v)))
    # Fix backbone to 1
    @inbounds for i in eachindex(lmo.mask)
        if !lmo.mask[i]
            v[i] = one(eltype(v))
        end
    end
    if lmo.k <= 0
        return v
    end
    # Select k non-backbone indices with most negative gradient
    k = min(lmo.k, length(lmo.sel))
    idx = partialsortperm(@view(g[lmo.sel]), 1:k)
    @inbounds for i in 1:k
        v[lmo.sel[idx[i]]] = one(eltype(v))
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

function (lmo::Box)(v::AbstractVector, g::AbstractVector)
    @inbounds for i in eachindex(v, g, lmo.lb, lmo.ub)
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
    # Start at lower bound
    v .= lmo.lb

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
