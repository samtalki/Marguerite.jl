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
    AbstractOracle

Abstract supertype for Frank-Wolfe linear minimization oracles.

Every concrete oracle `lmo <: AbstractOracle` is a callable struct invoked as
`lmo(v, g)`, writing the solution of

```math
\\min_{v \\in C} \\langle g, v \\rangle
```

into `v` in-place.

Any plain function `(v, g) -> v` also works as an oracle -- no subtyping required.
"""
abstract type AbstractOracle end



# ------------------------------------------------------------------
# Simplex (unified: capped and probability)
# ------------------------------------------------------------------

"""
    Simplex{T, Equality}(r)

Oracle for simplex constraints. The type parameter `Equality` controls whether
the budget constraint is ``\\le`` or ``=``.

- `Simplex(r)` / `Simplex(; r=1.0)`: capped simplex ``\\{x \\ge 0,\\; \\sum x_i \\le r\\}``
- `ProbSimplex(r)` / `ProbabilitySimplex(r)`: probability simplex ``\\{x \\ge 0,\\; \\sum x_i = r\\}``

**Capped** (`Equality=false`): Vertices are ``\\{0, r e_1, \\ldots, r e_n\\}``.
Selects ``r e_{i^*}`` where ``i^* = \\arg\\min_i g_i`` when ``g_{i^*} < 0``, otherwise the origin.
**Complexity**: ``O(n)``.

**Probability** (`Equality=true`): Vertices are ``\\{r e_1, \\ldots, r e_n\\}``.
Always selects ``r e_{i^*}`` where

```math
i^* = \\arg\\min_i g_i
```

**Complexity**: ``O(n)``.
"""
struct Simplex{T<:Real, Equality} <: AbstractOracle
    r::T
end

Simplex(r::Real) = Simplex{Float64, false}(Float64(r))
Simplex(; r::Real=1.0) = Simplex{Float64, false}(Float64(r))

"""
    ProbSimplex(r=1.0)

Convenience constructor for `Simplex{T, true}(r)` -- the probability simplex
``\\{x \\ge 0,\\; \\sum x_i = r\\}``.
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
# Shared helper: zero-allocation partial sort by most-negative gradient
# ------------------------------------------------------------------

"""
    _partial_sort_negative!(perm, g, k) -> count

Find up to `k` indices with the most negative values in `g`, stored sorted in
`perm[1:count]`. Zero-allocation: only uses the pre-allocated `perm` buffer.
``O(n \\cdot k)`` — fine since `k` is typically small.
"""
function _partial_sort_negative!(perm::Vector{Int}, g, k::Int)
    n = length(g)
    k = min(k, n)
    k <= 0 && return 0
    count = 0
    nan_seen = false
    @inbounds for i in 1:n
        gi = g[i]
        if gi != gi  # fast NaN check
            nan_seen = true
            continue
        end
        gi < zero(gi) || continue
        if count < k
            count += 1
        elseif gi >= g[perm[count]]
            continue
        end
        # insertion sort: place i into perm[1:count]
        j = count
        while j > 1 && g[perm[j-1]] > gi
            perm[j] = perm[j-1]
            j -= 1
        end
        perm[j] = i
    end
    if nan_seen
        @warn "_partial_sort_negative!: NaN in gradient; affected entries skipped" maxlog=3
    end
    return count
end

# ------------------------------------------------------------------
# Knapsack
# ------------------------------------------------------------------

"""
    Knapsack(budget, m)

Oracle for the knapsack polytope

```math
C = \\{x \\in [0,1]^m : \\sum x_i \\le \\text{budget}\\}
```

Selects up to `budget` indices with most negative gradient and sets them to 1;
only indices with strictly negative gradient are selected.
**Complexity**: ``O(m \\cdot k)`` where ``k = \\text{budget}``, via zero-allocation insertion sort.
For large budgets (``k \\approx m``), consider using a full sort or a [`Box`](@ref) oracle instead.
"""
struct Knapsack <: AbstractOracle
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
    count = _partial_sort_negative!(lmo.perm, g, lmo.k)
    @inbounds for i in 1:count
        v[lmo.perm[i]] = one(eltype(v))
    end
    return v
end

# ------------------------------------------------------------------
# MaskedKnapsack
# ------------------------------------------------------------------

"""
    MaskedKnapsack(budget, masked, m)

Oracle for the knapsack polytope with masked indices fixed to 1:

```math
C = \\{x \\in [0,1]^m : \\sum x_i \\le \\text{budget},\\; x_e = 1 \\;\\forall\\; e \\in \\text{masked}\\}
```

Fixes masked entries to 1, then selects up to ``k = \\text{budget} - |\\text{masked}|``
non-masked indices with most negative gradient; only indices with strictly
negative gradient are selected. **Complexity**: ``O(m \\cdot k)`` via zero-allocation insertion sort.
"""
struct MaskedKnapsack <: AbstractOracle
    is_masked::BitVector
    sel::Vector{Int}
    perm::Vector{Int}
    k::Int
    n_masked::Int
end

function MaskedKnapsack(budget::Int, masked::AbstractVector{<:Integer}, m::Int)
    is_masked = falses(m)
    is_masked[masked] .= true
    sel = findall(.!is_masked)
    k = budget - length(masked)
    k < 0 && error("budget ($budget) must be ≥ |masked| ($(length(masked)))")
    return MaskedKnapsack(is_masked, sel, collect(1:length(sel)), k, length(masked))
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
    g_sel = @view(g[lmo.sel])
    count = _partial_sort_negative!(lmo.perm, g_sel, lmo.k)
    @inbounds for i in 1:count
        v[lmo.sel[lmo.perm[i]]] = one(eltype(v))
    end
    return v
end

# ------------------------------------------------------------------
# Box
# ------------------------------------------------------------------

"""
    Box(lb, ub)

Oracle for the box constraint set

```math
C = \\{x : l_i \\le x_i \\le u_i\\}
```

Each coordinate is solved independently: select ``l_i`` when ``g_i \\ge 0``,
``u_i`` when ``g_i < 0``:

```math
v_i = \\begin{cases} l_i & g_i \\ge 0 \\\\ u_i & g_i < 0 \\end{cases}
```

**Complexity**: ``O(n)``.
"""
struct Box{T<:Real} <: AbstractOracle
    lb::Vector{T}
    ub::Vector{T}
end

function Box(lb::AbstractVector, ub::AbstractVector)
    length(lb) == length(ub) || throw(ArgumentError("Box: lb and ub must have equal length"))
    Box{Float64}(collect(Float64, lb), collect(Float64, ub))
end

@inline function (lmo::Box)(v::AbstractVector, g::AbstractVector)
    @inbounds @simd for i in eachindex(v, g, lmo.lb, lmo.ub)
        v[i] = g[i] >= zero(g[i]) ? lmo.lb[i] : lmo.ub[i]
    end
    return v
end

# ------------------------------------------------------------------
# WeightedSimplex
# ------------------------------------------------------------------

"""
    WeightedSimplex(α, β, lb)

Oracle for the weighted simplex

```math
C = \\{x \\ge l : \\langle \\alpha, x \\rangle \\le \\beta\\}
```

Shifts ``u = x - l``, adjusted budget ``\\bar\\beta = \\beta - \\langle \\alpha, l \\rangle``.
Then

```math
u^* = \\frac{\\bar\\beta}{\\alpha_{i^*}}\\, e_{i^*}, \\quad
i^* = \\arg\\min_i \\left\\{\\frac{g_i}{\\alpha_i} : g_i < 0\\right\\}
```

Returns ``v = u^* + l``.

When all ``g_i \\ge 0``, returns the lower bound ``l``.

**Complexity**: ``O(m)``.
"""
struct WeightedSimplex{T<:Real} <: AbstractOracle
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

# ------------------------------------------------------------------
# Fused LMO + gap computation (sparse vertex protocol)
# ------------------------------------------------------------------

# Returns (fw_gap, nnz) where:
#   nnz = -1 → dense vertex path (c.vertex populated)
#   nnz = 0  → origin vertex
#   nnz > 0  → sparse vertex (c.vertex_nzind[1:nnz], c.vertex_nzval[1:nnz])

"""
    _lmo_and_gap!(lmo, c::Cache, x, n) -> (fw_gap, nnz)

Fused linear minimization oracle + Frank-Wolfe gap computation.

Returns `(fw_gap, nnz)` where `nnz` encodes the vertex representation:
- `nnz = -1`: dense vertex stored in `c.vertex`
- `nnz = 0`: origin vertex (all zeros)
- `nnz > 0`: sparse vertex in `c.vertex_nzind[1:nnz]`, `c.vertex_nzval[1:nnz]`

Specializations exist for `Simplex`, `Knapsack`, and `MaskedKnapsack` to avoid
materializing the full dense vertex vector. The generic fallback calls `lmo(c.vertex, c.gradient)`
and returns `nnz = -1`.
"""
# Dense fallback (any LMO type including user-supplied functions)
function _lmo_and_gap!(lmo, c::Cache{T}, x, n) where T
    lmo(c.vertex, c.gradient)
    fw_gap = zero(T)
    @inbounds @simd for i in 1:n
        fw_gap += c.gradient[i] * (x[i] - c.vertex[i])
    end
    return (fw_gap, -1)
end

# Simplex specialization: single O(n) pass for argmin(g) + dot(g,x)
function _lmo_and_gap!(lmo::Simplex{ST, Equality}, c::Cache{T}, x, n) where {ST, T, Equality}
    g = c.gradient
    dot_gx = zero(T)
    i_star = 1
    g_min = g[1]
    @inbounds for i in 1:n
        gi = g[i]
        dot_gx += gi * x[i]
        if gi < g_min
            g_min = gi
            i_star = i
        end
    end
    if Equality || g_min < zero(g_min)
        c.vertex_nzind[1] = i_star
        c.vertex_nzval[1] = T(lmo.r)
        return (dot_gx - T(lmo.r) * g_min, 1)
    else
        return (dot_gx, 0)
    end
end

# Knapsack specialization: dot(g,x) + partial sort
function _lmo_and_gap!(lmo::Knapsack, c::Cache{T}, x, n) where T
    dot_gx = zero(T)
    @inbounds @simd for i in 1:n
        dot_gx += c.gradient[i] * x[i]
    end
    if lmo.k <= 0
        return (dot_gx, 0)
    end
    count = _partial_sort_negative!(lmo.perm, c.gradient, lmo.k)
    vertex_contrib = zero(T)
    @inbounds for j in 1:count
        idx = lmo.perm[j]
        c.vertex_nzind[j] = idx
        c.vertex_nzval[j] = one(T)
        vertex_contrib += c.gradient[idx]
    end
    return (dot_gx - vertex_contrib, count)
end

# MaskedKnapsack specialization: falls back to dense if nnz > n/2
function _lmo_and_gap!(lmo::MaskedKnapsack, c::Cache{T}, x, n) where T
    # If budget allows many nonzeros, fall back to dense path
    if lmo.k + lmo.n_masked > n ÷ 2
        lmo(c.vertex, c.gradient)
        fw_gap = zero(T)
        @inbounds @simd for i in 1:n
            fw_gap += c.gradient[i] * (x[i] - c.vertex[i])
        end
        return (fw_gap, -1)
    end
    dot_gx = zero(T)
    @inbounds @simd for i in 1:n
        dot_gx += c.gradient[i] * x[i]
    end
    # Masked indices contribute to gap and are nonzeros
    nnz = 0
    vertex_contrib = zero(T)
    @inbounds for i in eachindex(lmo.is_masked)
        if lmo.is_masked[i]
            nnz += 1
            c.vertex_nzind[nnz] = i
            c.vertex_nzval[nnz] = one(T)
            vertex_contrib += c.gradient[i]
        end
    end
    if lmo.k > 0
        g_sel = @view(c.gradient[lmo.sel])
        sel_count = _partial_sort_negative!(lmo.perm, g_sel, lmo.k)
        @inbounds for j in 1:sel_count
            idx = lmo.sel[lmo.perm[j]]
            nnz += 1
            c.vertex_nzind[nnz] = idx
            c.vertex_nzval[nnz] = one(T)
            vertex_contrib += c.gradient[idx]
        end
    end
    return (dot_gx - vertex_contrib, nnz)
end

# ------------------------------------------------------------------
# active_set: identify active constraints at x*
# ------------------------------------------------------------------

"""
    active_set(lmo, x; tol=1e-8) -> ActiveConstraints

Identify active constraints at solution `x` for the given oracle.

Returns an [`ActiveConstraints`](@ref) with bound-pinned indices, free indices,
and equality constraint normals/RHS.
"""
function active_set end

# Default fallback: no active constraints (interior solution)
function active_set(lmo, x::AbstractVector{T}; tol::Real=1e-8) where T
    @warn "no active_set specialization for $(typeof(lmo)); assuming interior solution" maxlog=1
    n = length(x)
    ActiveConstraints{T}(Int[], T[], BitVector(), collect(1:n), Vector{T}[], T[])
end

function active_set(lmo::Box{T}, x::AbstractVector; tol::Real=1e-8) where T
    n = length(x)
    bound_idx = Int[]
    bound_val = T[]
    bound_lower = BitVector()
    free_idx = Int[]
    for i in 1:n
        if abs(x[i] - lmo.lb[i]) ≤ tol
            push!(bound_idx, i)
            push!(bound_val, lmo.lb[i])
            push!(bound_lower, true)
        elseif abs(x[i] - lmo.ub[i]) ≤ tol
            push!(bound_idx, i)
            push!(bound_val, lmo.ub[i])
            push!(bound_lower, false)
        else
            push!(free_idx, i)
        end
    end
    ActiveConstraints{T}(bound_idx, bound_val, bound_lower, free_idx, Vector{T}[], T[])
end

function active_set(lmo::Simplex{T, true}, x::AbstractVector; tol::Real=1e-8) where T
    n = length(x)
    bound_idx = Int[]
    bound_val = T[]
    bound_lower = BitVector()
    free_idx = Int[]
    for i in 1:n
        if abs(x[i]) ≤ tol
            push!(bound_idx, i)
            push!(bound_val, zero(T))
            push!(bound_lower, true)
        else
            push!(free_idx, i)
        end
    end
    # Budget equality ∑x_i = r is always active
    eq_normal = ones(T, n)
    ActiveConstraints{T}(bound_idx, bound_val, bound_lower, free_idx, [eq_normal], [lmo.r])
end

function active_set(lmo::Simplex{T, false}, x::AbstractVector; tol::Real=1e-8) where T
    n = length(x)
    bound_idx = Int[]
    bound_val = T[]
    bound_lower = BitVector()
    free_idx = Int[]
    for i in 1:n
        if abs(x[i]) ≤ tol
            push!(bound_idx, i)
            push!(bound_val, zero(T))
            push!(bound_lower, true)
        else
            push!(free_idx, i)
        end
    end
    # Budget inequality ∑x_i ≤ r: active if ∑x_i ≈ r
    eq_normals = Vector{T}[]
    eq_rhs = T[]
    if abs(sum(x) - lmo.r) ≤ tol
        push!(eq_normals, ones(T, n))
        push!(eq_rhs, lmo.r)
    end
    ActiveConstraints{T}(bound_idx, bound_val, bound_lower, free_idx, eq_normals, eq_rhs)
end

function active_set(lmo::WeightedSimplex{T}, x::AbstractVector; tol::Real=1e-8) where T
    n = length(x)
    bound_idx = Int[]
    bound_val = T[]
    bound_lower = BitVector()
    free_idx = Int[]
    for i in 1:n
        if abs(x[i] - lmo.lb[i]) ≤ tol
            push!(bound_idx, i)
            push!(bound_val, lmo.lb[i])
            push!(bound_lower, true)
        else
            push!(free_idx, i)
        end
    end
    # Budget inequality ⟨α, x⟩ ≤ β: active if ⟨α, x⟩ ≈ β
    eq_normals = Vector{T}[]
    eq_rhs = T[]
    if abs(dot(lmo.α, x) - lmo.β) ≤ tol
        push!(eq_normals, copy(lmo.α))
        push!(eq_rhs, lmo.β)
    end
    ActiveConstraints{T}(bound_idx, bound_val, bound_lower, free_idx, eq_normals, eq_rhs)
end

function active_set(lmo::Knapsack, x::AbstractVector{T}; tol::Real=1e-8) where T
    n = length(x)
    bound_idx = Int[]
    bound_val = T[]
    bound_lower = BitVector()
    free_idx = Int[]
    for i in 1:n
        if abs(x[i]) ≤ tol
            push!(bound_idx, i)
            push!(bound_val, zero(T))
            push!(bound_lower, true)
        elseif abs(x[i] - one(T)) ≤ tol
            push!(bound_idx, i)
            push!(bound_val, one(T))
            push!(bound_lower, false)
        else
            push!(free_idx, i)
        end
    end
    eq_normals = Vector{T}[]
    eq_rhs = T[]
    if abs(sum(x) - lmo.k) ≤ tol
        push!(eq_normals, ones(T, n))
        push!(eq_rhs, T(lmo.k))
    end
    ActiveConstraints{T}(bound_idx, bound_val, bound_lower, free_idx, eq_normals, eq_rhs)
end

function active_set(lmo::MaskedKnapsack, x::AbstractVector{T}; tol::Real=1e-8) where T
    n = length(x)
    bound_idx = Int[]
    bound_val = T[]
    bound_lower = BitVector()
    free_idx = Int[]
    for i in 1:n
        if lmo.is_masked[i]
            # Masked indices always pinned to 1 (upper bound)
            push!(bound_idx, i)
            push!(bound_val, one(T))
            push!(bound_lower, false)
        elseif abs(x[i]) ≤ tol
            push!(bound_idx, i)
            push!(bound_val, zero(T))
            push!(bound_lower, true)
        elseif abs(x[i] - one(T)) ≤ tol
            push!(bound_idx, i)
            push!(bound_val, one(T))
            push!(bound_lower, false)
        else
            push!(free_idx, i)
        end
    end
    # Budget: ∑x_i ≤ budget (with budget = lmo.k + |masked|)
    total_budget = lmo.k + lmo.n_masked
    eq_normals = Vector{T}[]
    eq_rhs = T[]
    if abs(sum(x) - total_budget) ≤ tol
        push!(eq_normals, ones(T, n))
        push!(eq_rhs, T(total_budget))
    end
    ActiveConstraints{T}(bound_idx, bound_val, bound_lower, free_idx, eq_normals, eq_rhs)
end

# ------------------------------------------------------------------
# materialize: instantiate concrete oracle from parameterized oracle
# ------------------------------------------------------------------

"""
    materialize(plmo::ParametricOracle, θ) -> concrete_lmo

Evaluate parameter functions at ``\\theta`` and return a concrete oracle.
"""
function materialize end

function materialize(plmo::ParametricBox, θ)
    Box(plmo.lb_fn(θ), plmo.ub_fn(θ))
end

function materialize(plmo::ParametricSimplex{R, Equality}, θ) where {R, Equality}
    Simplex{Float64, Equality}(Float64(plmo.r_fn(θ)))
end

function materialize(plmo::ParametricWeightedSimplex, θ)
    WeightedSimplex(plmo.α_fn(θ), plmo.β_fn(θ), plmo.lb_fn(θ))
end
