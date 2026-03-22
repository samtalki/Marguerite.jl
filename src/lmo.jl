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
    AbstractOracle

Abstract supertype for Frank-Wolfe linear minimization oracles.

Every concrete oracle `lmo <: AbstractOracle` is a callable struct invoked as
`lmo(v, g)`, writing the solution of

```math
\\min_{v \\in C} \\langle g, v \\rangle
```

into `v` in-place.

Plain functions `(v, g) -> v` are auto-wrapped as [`FunctionOracle`](@ref) by
`solve`. Non-function callable structs should subtype `AbstractOracle` directly
or be wrapped explicitly with `FunctionOracle` for specialized dispatch
(e.g. `active_set`, sparse vertex protocol).
"""
abstract type AbstractOracle end

"""
    FunctionOracle{F} <: AbstractOracle

Wraps a plain function `fn(v, g) -> v` as an [`AbstractOracle`](@ref).

```julia
lmo = FunctionOracle(my_lmo_function)
solve(f, lmo, x0; grad=∇f!)
```
"""
struct FunctionOracle{F} <: AbstractOracle
    fn::F
end
(o::FunctionOracle)(v, g) = o.fn(v, g)
FunctionOracle(o::AbstractOracle) = o

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
    function Simplex{T, Equality}(r::T) where {T<:Real, Equality}
        r >= zero(T) || throw(ArgumentError("Simplex: radius r must be nonnegative, got r=$r"))
        new{T, Equality}(r)
    end
end

Simplex(r::T) where {T<:AbstractFloat} = Simplex{T, false}(r)
Simplex(r::Real) = Simplex{Float64, false}(Float64(r))
Simplex(; r::Real=1.0) = Simplex{Float64, false}(Float64(r))

"""
    ProbSimplex(r=1.0)

Convenience constructor for `Simplex{T, true}(r)` -- the probability simplex
``\\{x \\ge 0,\\; \\sum x_i = r\\}``.
"""
ProbSimplex(r::T) where {T<:AbstractFloat} = Simplex{T, true}(r)
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

    function Box{T}(lb::Vector{T}, ub::Vector{T}) where {T<:Real}
        length(lb) == length(ub) || throw(ArgumentError("Box: lb and ub must have equal length"))
        all(lb .≤ ub) || throw(ArgumentError("Box: requires lb[i] ≤ ub[i] for all i"))
        new{T}(lb, ub)
    end
end

function Box(lb::AbstractVector{T}, ub::AbstractVector{T}) where {T<:AbstractFloat}
    Box{T}(collect(T, lb), collect(T, ub))
end

function Box(lb::AbstractVector, ub::AbstractVector)
    Box{Float64}(collect(Float64, lb), collect(Float64, ub))
end

@inline function (lmo::Box)(v::AbstractVector, g::AbstractVector)
    @inbounds @simd for i in eachindex(v, g, lmo.lb, lmo.ub)
        v[i] = g[i] >= zero(g[i]) ? lmo.lb[i] : lmo.ub[i]
    end
    return v
end

# ------------------------------------------------------------------
# ScalarBox
# ------------------------------------------------------------------

"""
    ScalarBox{T}(lb, ub)

Oracle for a box constraint with uniform scalar bounds:

```math
C = \\{x : l \\le x_i \\le u \\;\\forall i\\}
```

Memory-efficient alternative to [`Box`](@ref) when all bounds are identical.
Convenience constructor: `Box(lb::Real, ub::Real)`.

**Complexity**: ``O(n)``.
"""
struct ScalarBox{T<:Real} <: AbstractOracle
    lb::T
    ub::T
    function ScalarBox{T}(lb::T, ub::T) where {T<:Real}
        lb <= ub || throw(ArgumentError("ScalarBox: lb ($lb) must be ≤ ub ($ub)"))
        new{T}(lb, ub)
    end
end

Box(lb::T, ub::T) where {T<:AbstractFloat} = ScalarBox{T}(lb, ub)
Box(lb::Real, ub::Real) = ScalarBox{Float64}(Float64(lb), Float64(ub))

@inline function (lmo::ScalarBox)(v::AbstractVector, g::AbstractVector)
    @inbounds @simd for i in eachindex(v, g)
        v[i] = g[i] >= zero(g[i]) ? lmo.lb : lmo.ub
    end
    return v
end

# ScalarBox fused LMO + gap (single-pass, identical to Box dense fallback)
function _lmo_and_gap!(lmo::ScalarBox, c::Cache{T}, x, n) where T
    lmo(c.vertex, c.gradient)
    fw_gap = zero(T)
    @inbounds @simd for i in 1:n
        fw_gap += c.gradient[i] * (x[i] - c.vertex[i])
    end
    return (fw_gap, -1)
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

Shifts ``u = x - l``, adjusted budget ``\\beta_{\\mathrm{bar}} = \\beta - \\langle \\alpha, l \\rangle``.
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
    function WeightedSimplex{T}(α::Vector{T}, β::T, lb::Vector{T}, β_bar::T) where {T<:Real}
        length(α) == length(lb) ||
            throw(ArgumentError("WeightedSimplex: α and lb must have equal length"))
        all(>(zero(T)), α) ||
            throw(ArgumentError("WeightedSimplex: all weights α must be positive"))
        β_bar >= zero(T) ||
            throw(ArgumentError("WeightedSimplex: requires β ≥ dot(α, lb) for a nonempty feasible set"))
        new{T}(α, β, lb, β_bar)
    end
end

function WeightedSimplex(α::AbstractVector{T}, β::Real, lb::AbstractVector{<:Real}) where {T<:AbstractFloat}
    length(α) == length(lb) ||
        throw(ArgumentError("WeightedSimplex: α and lb must have equal length"))
    α_ = collect(T, α)
    lb_ = collect(T, lb)
    β_ = T(β)
    β_bar = β_ - dot(α_, lb_)
    return WeightedSimplex{T}(α_, β_, lb_, β_bar)
end

function WeightedSimplex(α::AbstractVector{<:Real}, β::Real, lb::AbstractVector{<:Real})
    length(α) == length(lb) ||
        throw(ArgumentError("WeightedSimplex: α and lb must have equal length"))
    α_ = collect(Float64, α)
    lb_ = collect(Float64, lb)
    β_ = Float64(β)
    β_bar = β_ - dot(α_, lb_)
    return WeightedSimplex{Float64}(α_, β_, lb_, β_bar)
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
# Spectraplex
# ------------------------------------------------------------------

"""
    Spectraplex{T}(n, r)

Oracle for the spectraplex (spectrahedron with trace constraint)

```math
C = \\{X \\in \\mathbb{S}_+^n : \\operatorname{tr}(X) = r\\}
```

The solver operates on `vec(X)` (length ``n^2``). Given gradient ``g`` (as a vector),
reshapes to ``G \\in \\mathbb{R}^{n \\times n}``, symmetrizes, and computes the minimum
eigenvector ``v_{\\min}`` of ``\\frac{1}{2}(G + G^\\top)``. The vertex is the rank-1 matrix
``r \\, v_{\\min} v_{\\min}^\\top``, written column-major into the output buffer.

Convenience: `Spectraplex(n)` gives the unit spectraplex (``r = 1``).

**Complexity**: ``O(n^3)`` (dense eigendecomposition).
"""
struct Spectraplex{T<:Real} <: AbstractOracle
    n::Int
    r::T
end

@inline function _validate_spectraplex_args(n::Int, r::Real)
    n > 0 || throw(ArgumentError("Spectraplex: dimension n must be positive, got n=$n"))
    r >= 0 || throw(ArgumentError("Spectraplex: radius r must be nonnegative, got r=$r"))
    return nothing
end

function Spectraplex(n::Int)
    _validate_spectraplex_args(n, 1.0)
    return Spectraplex{Float64}(n, 1.0)
end

function Spectraplex(n::Int, r::T) where {T<:AbstractFloat}
    _validate_spectraplex_args(n, r)
    return Spectraplex{T}(n, r)
end

function Spectraplex(n::Int, r::Integer)
    r_float = Float64(r)
    _validate_spectraplex_args(n, r_float)
    return Spectraplex{Float64}(n, r_float)
end

# Constraint normal kinds for SpectraplexEqNormal
const _SPECTRAPLEX_ANTISYMMETRIC = 0x01  # X[i,j] - X[j,i] = 0
const _SPECTRAPLEX_TRACE         = 0x02  # tr(X) = r
const _SPECTRAPLEX_MIXED         = 0x03  # u*v' + v*u' (active × null-space)
const _SPECTRAPLEX_NULL          = 0x04  # v_i*v_j' + v_j*v_i' (null × null)

struct SpectraplexEqNormal{T, VT<:AbstractVector{T}, WT<:AbstractVector{T}} <: AbstractVector{T}
    n::Int
    kind::UInt8
    i::Int
    j::Int
    u::VT
    v::WT
end

struct SpectraplexEqNormals{T<:Real, MT1<:AbstractMatrix{T}, MT2<:AbstractMatrix{T}, VT<:AbstractVector{T}} <: AbstractVector{AbstractVector{T}}
    n::Int
    trace_rhs::T
    U::MT1
    V_perp::MT2
    empty::VT
end

Base.IndexStyle(::Type{<:SpectraplexEqNormal}) = IndexLinear()
Base.size(a::SpectraplexEqNormal) = (a.n * a.n,)
Base.IndexStyle(::Type{<:SpectraplexEqNormals}) = IndexLinear()

@inline function _spectraplex_vec_rc(idx::Int, n::Int)
    row = mod1(idx, n)
    col = fld(idx - 1, n) + 1
    return row, col
end

@inline function _spectraplex_sym_count(n::Int)
    return n * (n - 1) ÷ 2
end

@inline function _spectraplex_mixed_count(eq::SpectraplexEqNormals)
    return size(eq.U, 2) * size(eq.V_perp, 2)
end

@inline function _spectraplex_null_count(eq::SpectraplexEqNormals)
    q = size(eq.V_perp, 2)
    return q * (q + 1) ÷ 2
end

function Base.length(eq::SpectraplexEqNormals)
    return _spectraplex_sym_count(eq.n) + 1 + _spectraplex_mixed_count(eq) + _spectraplex_null_count(eq)
end

Base.size(eq::SpectraplexEqNormals) = (length(eq),)

function _spectraplex_sym_pair(idx::Int, n::Int)
    count = idx
    for j in 2:n
        block = j - 1
        if count <= block
            return count, j
        end
        count -= block
    end
    throw(BoundsError())
end

function _spectraplex_upper_tri_pair(idx::Int, q::Int)
    count = idx
    for j in 1:q
        block = q - j + 1
        if count <= block
            return j + count - 1, j
        end
        count -= block
    end
    throw(BoundsError())
end

function Base.getindex(eq::SpectraplexEqNormals{T}, idx::Int) where T
    checkbounds(eq, idx)
    sym_count = _spectraplex_sym_count(eq.n)
    if idx <= sym_count
        i, j = _spectraplex_sym_pair(idx, eq.n)
        return SpectraplexEqNormal(eq.n, _SPECTRAPLEX_ANTISYMMETRIC, i, j, eq.empty, eq.empty)
    end

    idx -= sym_count
    if idx == 1
        return SpectraplexEqNormal(eq.n, _SPECTRAPLEX_TRACE, 0, 0, eq.empty, eq.empty)
    end

    idx -= 1
    q = size(eq.V_perp, 2)
    k = size(eq.U, 2)
    mixed_count = k * q
    if idx <= mixed_count
        u_idx = mod1(idx, k)
        v_idx = fld(idx - 1, k) + 1
        return SpectraplexEqNormal(eq.n, _SPECTRAPLEX_MIXED, u_idx, v_idx,
                                   @view(eq.U[:, u_idx]), @view(eq.V_perp[:, v_idx]))
    end

    idx -= mixed_count
    i, j = _spectraplex_upper_tri_pair(idx, q)
    return SpectraplexEqNormal(eq.n, _SPECTRAPLEX_NULL, i, j,
                               @view(eq.V_perp[:, i]), @view(eq.V_perp[:, j]))
end

function Base.getindex(a::SpectraplexEqNormal{T}, idx::Int) where T
    checkbounds(a, idx)
    row, col = _spectraplex_vec_rc(idx, a.n)
    if a.kind == _SPECTRAPLEX_ANTISYMMETRIC
        return row == a.i && col == a.j ? one(T) :
               row == a.j && col == a.i ? -one(T) : zero(T)
    elseif a.kind == _SPECTRAPLEX_TRACE
        return row == col ? one(T) : zero(T)
    end
    return a.u[row] * a.v[col] + a.v[row] * a.u[col]
end

function dot(a::SpectraplexEqNormal{T}, x::AbstractVector) where T
    if a.kind == _SPECTRAPLEX_ANTISYMMETRIC
        idx1 = (a.j - 1) * a.n + a.i
        idx2 = (a.i - 1) * a.n + a.j
        return T(x[idx1] - x[idx2])
    elseif a.kind == _SPECTRAPLEX_TRACE
        s = zero(T)
        @inbounds for i in 1:a.n
            s += x[(i - 1) * a.n + i]
        end
        return s
    end

    X = reshape(x, a.n, a.n)
    s = zero(T)
    @inbounds for col in 1:a.n
        vc = a.v[col]
        uc = a.u[col]
        for row in 1:a.n
            s += X[row, col] * (a.u[row] * vc + a.v[row] * uc)
        end
    end
    return s
end

"""
    _spectraplex_min_eigen(g, n) -> (λ_min, v_min)

Symmetrize the gradient (reshaped as n×n) and return the minimum eigenvalue and
eigenvector.  Shared by the oracle callable and `_lmo_and_gap!`.
"""
function _spectraplex_min_eigen(g::AbstractVector, n::Int)
    G = reshape(g, n, n)
    buf = Matrix{eltype(g)}(undef, n, n)
    @inbounds for j in 1:n
        for i in 1:n
            buf[i, j] = (G[i, j] + G[j, i]) / 2
        end
    end
    E = eigen!(Symmetric(buf))
    return E.values[1], @view(E.vectors[:, 1])
end

"""
    _spectraplex_write_rank1!(v, v_min, r, n)

Write the rank-1 vertex ``r \\, v_{\\min} v_{\\min}^\\top`` column-major into `v`.
"""
function _spectraplex_write_rank1!(v::AbstractVector, v_min::AbstractVector, r::Real, n::Int)
    @inbounds for j in 1:n
        for i in 1:n
            v[(j-1)*n + i] = r * v_min[i] * v_min[j]
        end
    end
    return v
end

function (lmo::Spectraplex)(v::AbstractVector, g::AbstractVector)
    _, v_min = _spectraplex_min_eigen(g, lmo.n)
    _spectraplex_write_rank1!(v, v_min, lmo.r, lmo.n)
    return v
end

# ------------------------------------------------------------------
# Fused LMO + gap computation (sparse vertex protocol)
# ------------------------------------------------------------------

# Returns (fw_gap, nnz) where:
#   nnz = -1 → dense vertex path (c.vertex populated)
#   nnz = 0  → origin vertex
#   nnz > 0  → sparse vertex (c.vertex_nzind[1:nnz], c.vertex_nzval[1:nnz])

# Dense fallback (Box, WeightedSimplex, FunctionOracle, or any AbstractOracle without specialization)
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

Indices in `c.vertex_nzind[1:nnz]` must be distinct.
"""
function _lmo_and_gap!(lmo, c::Cache{T}, x, n) where T
    lmo(c.vertex, c.gradient)
    fw_gap = zero(T)
    @inbounds @simd for i in 1:n
        fw_gap += c.gradient[i] * (x[i] - c.vertex[i])
    end
    return (fw_gap, -1)
end

# Spectraplex specialization: gap = ⟨g, x⟩ - r·λ_min, dense vertex
function _lmo_and_gap!(lmo::Spectraplex, c::Cache{T}, x, m) where T
    λ_min, v_min = _spectraplex_min_eigen(c.gradient, lmo.n)
    _spectraplex_write_rank1!(c.vertex, v_min, lmo.r, lmo.n)
    fw_gap = dot(c.gradient, x) - T(lmo.r) * T(λ_min)
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

# MaskedKnapsack specialization: falls back to dense if budget > n/2
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

# Shared helpers for active_set methods
@inline function _init_active_arrays(::Type{T}) where T
    (Int[], T[], BitVector(), Int[])
end

@inline function _push_bound!(bound_idx, bound_val, bound_lower, i, val::T, is_lower::Bool) where T
    push!(bound_idx, i)
    push!(bound_val, val)
    push!(bound_lower, is_lower)
end

# Default fallback: no active constraints (interior solution)
function active_set(lmo, x::AbstractVector{T}; tol::Real=1e-8) where T
    @warn "no active_set specialization for $(typeof(lmo)); assuming interior solution" maxlog=1
    n = length(x)
    ActiveConstraints{T}(Int[], T[], BitVector(), collect(1:n), Vector{T}[], T[])
end

function active_set(lmo::Box{T}, x::AbstractVector; tol::Real=1e-8) where T
    n = length(x)
    bound_idx, bound_val, bound_lower, free_idx = _init_active_arrays(T)
    for i in 1:n
        if abs(x[i] - lmo.lb[i]) ≤ tol
            _push_bound!(bound_idx, bound_val, bound_lower, i, lmo.lb[i], true)
        elseif abs(x[i] - lmo.ub[i]) ≤ tol
            _push_bound!(bound_idx, bound_val, bound_lower, i, lmo.ub[i], false)
        else
            push!(free_idx, i)
        end
    end
    ActiveConstraints{T}(bound_idx, bound_val, bound_lower, free_idx, Vector{T}[], T[])
end

function active_set(lmo::ScalarBox{T}, x::AbstractVector; tol::Real=1e-8) where T
    n = length(x)
    bound_idx, bound_val, bound_lower, free_idx = _init_active_arrays(T)
    for i in 1:n
        if abs(x[i] - lmo.lb) ≤ tol
            _push_bound!(bound_idx, bound_val, bound_lower, i, lmo.lb, true)
        elseif abs(x[i] - lmo.ub) ≤ tol
            _push_bound!(bound_idx, bound_val, bound_lower, i, lmo.ub, false)
        else
            push!(free_idx, i)
        end
    end
    ActiveConstraints{T}(bound_idx, bound_val, bound_lower, free_idx, Vector{T}[], T[])
end

function active_set(lmo::Simplex{T, true}, x::AbstractVector; tol::Real=1e-8) where T
    n = length(x)
    bound_idx, bound_val, bound_lower, free_idx = _init_active_arrays(T)
    for i in 1:n
        if abs(x[i]) ≤ tol
            _push_bound!(bound_idx, bound_val, bound_lower, i, zero(T), true)
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
    bound_idx, bound_val, bound_lower, free_idx = _init_active_arrays(T)
    for i in 1:n
        if abs(x[i]) ≤ tol
            _push_bound!(bound_idx, bound_val, bound_lower, i, zero(T), true)
        else
            push!(free_idx, i)
        end
    end
    # Budget inequality ∑x_i ≤ r: active if ∑x_i ≈ r
    eq_normals = Vector{T}[]
    eq_rhs = T[]
    if abs(sum(x) - lmo.r) ≤ tol * (1 + abs(lmo.r))
        push!(eq_normals, ones(T, n))
        push!(eq_rhs, lmo.r)
    end
    ActiveConstraints{T}(bound_idx, bound_val, bound_lower, free_idx, eq_normals, eq_rhs)
end

function active_set(lmo::WeightedSimplex{T}, x::AbstractVector; tol::Real=1e-8) where T
    n = length(x)
    bound_idx, bound_val, bound_lower, free_idx = _init_active_arrays(T)
    for i in 1:n
        if abs(x[i] - lmo.lb[i]) ≤ tol
            _push_bound!(bound_idx, bound_val, bound_lower, i, lmo.lb[i], true)
        else
            push!(free_idx, i)
        end
    end
    # Budget inequality ⟨α, x⟩ ≤ β: active if ⟨α, x⟩ ≈ β
    eq_normals = Vector{T}[]
    eq_rhs = T[]
    if abs(dot(lmo.α, x) - lmo.β) ≤ tol * (1 + abs(lmo.β))
        push!(eq_normals, copy(lmo.α))
        push!(eq_rhs, lmo.β)
    end
    ActiveConstraints{T}(bound_idx, bound_val, bound_lower, free_idx, eq_normals, eq_rhs)
end

function active_set(lmo::Knapsack, x::AbstractVector{T}; tol::Real=1e-8) where T
    n = length(x)
    bound_idx, bound_val, bound_lower, free_idx = _init_active_arrays(T)
    for i in 1:n
        if abs(x[i]) ≤ tol
            _push_bound!(bound_idx, bound_val, bound_lower, i, zero(T), true)
        elseif abs(x[i] - one(T)) ≤ tol
            _push_bound!(bound_idx, bound_val, bound_lower, i, one(T), false)
        else
            push!(free_idx, i)
        end
    end
    eq_normals = Vector{T}[]
    eq_rhs = T[]
    if abs(sum(x) - lmo.k) ≤ tol * (1 + abs(T(lmo.k)))
        push!(eq_normals, ones(T, n))
        push!(eq_rhs, T(lmo.k))
    end
    ActiveConstraints{T}(bound_idx, bound_val, bound_lower, free_idx, eq_normals, eq_rhs)
end

function active_set(lmo::MaskedKnapsack, x::AbstractVector{T}; tol::Real=1e-8) where T
    n = length(x)
    bound_idx, bound_val, bound_lower, free_idx = _init_active_arrays(T)
    for i in 1:n
        if lmo.is_masked[i]
            _push_bound!(bound_idx, bound_val, bound_lower, i, one(T), false)
        elseif abs(x[i]) ≤ tol
            _push_bound!(bound_idx, bound_val, bound_lower, i, zero(T), true)
        elseif abs(x[i] - one(T)) ≤ tol
            _push_bound!(bound_idx, bound_val, bound_lower, i, one(T), false)
        else
            push!(free_idx, i)
        end
    end
    # Budget: ∑x_i ≤ budget (with budget = lmo.k + |masked|)
    total_budget = lmo.k + lmo.n_masked
    eq_normals = Vector{T}[]
    eq_rhs = T[]
    if abs(sum(x) - total_budget) ≤ tol * (1 + abs(T(total_budget)))
        push!(eq_normals, ones(T, n))
        push!(eq_rhs, T(total_budget))
    end
    ActiveConstraints{T}(bound_idx, bound_val, bound_lower, free_idx, eq_normals, eq_rhs)
end

function active_set(lmo::Spectraplex{T}, x::AbstractVector; tol::Real=1e-8) where T
    n = lmo.n
    m = n * n
    TP = promote_type(T, eltype(x))
    X = TP.(reshape(x, n, n))
    X_sym = Symmetric((X .+ X') ./ TP(2))
    E = eigen(X_sym)

    # Rank detection scales with the trace radius, plus a floor from the
    # eigendecomposition backward error O(n · eps · ‖X‖) to avoid
    # misclassifying numerical noise as real eigenvalues for large n or
    # tight tol.
    max_abs_λ = maximum(abs, E.values)
    rank_tol = if iszero(lmo.r)
        zero(TP)
    else
        max(TP(tol) * abs(TP(lmo.r)), TP(n) * eps(TP) * max_abs_λ)
    end
    k = count(λ -> λ > rank_tol, E.values)
    n_zero = n - k
    V_perp = Matrix{TP}(E.vectors[:, 1:n_zero])
    U = Matrix{TP}(E.vectors[:, (n_zero + 1):n])
    eq_normals = SpectraplexEqNormals(n, TP(lmo.r), U, V_perp, TP[])
    eq_rhs = zeros(TP, length(eq_normals))
    eq_rhs[_spectraplex_sym_count(n) + 1] = TP(lmo.r)

    ActiveConstraints{TP}(Int[], TP[], BitVector(), collect(1:m), eq_normals, eq_rhs)
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
    r = plmo.r_fn(θ)
    T = r isa AbstractFloat ? typeof(r) : Float64
    Simplex{T, Equality}(T(r))
end

function materialize(plmo::ParametricWeightedSimplex, θ)
    WeightedSimplex(plmo.α_fn(θ), plmo.β_fn(θ), plmo.lb_fn(θ))
end
