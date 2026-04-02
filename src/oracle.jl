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
Simplex(r::Real) = (rf = float(r); Simplex{typeof(rf), false}(rf))
Simplex(; r::Real=1.0) = (rf = float(r); Simplex{typeof(rf), false}(rf))

"""
    ProbSimplex(r=1.0)

Convenience constructor for `Simplex{T, true}(r)` -- the probability simplex
``\\{x \\ge 0,\\; \\sum x_i = r\\}``.
"""
ProbSimplex(r::T) where {T<:AbstractFloat} = Simplex{T, true}(r)
ProbSimplex(r::Real) = (rf = float(r); Simplex{typeof(rf), true}(rf))
ProbSimplex(; r::Real=1.0) = (rf = float(r); Simplex{typeof(rf), true}(rf))

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
    if g_min != g_min  # NaN check
        @warn "Simplex oracle: NaN in gradient" maxlog=3
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
    budget < 0 && throw(ArgumentError("Knapsack: budget must be ≥ 0, got $budget"))
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
    k < 0 && throw(ArgumentError("MaskedKnapsack: budget ($budget) must be ≥ |masked| ($(length(masked)))"))
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
``r \\, v_{\\min} v_{\\min}^\\top``, written column major into the output buffer.

Convenience: `Spectraplex(n)` gives the unit spectraplex (``r = 1``).

**Complexity**: ``O(n^3)`` (dense eigendecomposition).
"""
struct Spectraplex{T<:Real} <: AbstractOracle
    n::Int
    r::T
    function Spectraplex{T}(n::Int, r::T) where {T<:Real}
        _validate_spectraplex_args(n, r)
        new{T}(n, r)
    end
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

"""
    SpectraplexEqNormals{T}

Lightweight representation of equality constraints for the spectraplex active set.
Stores the active eigenvectors `U` (rank columns) and null space eigenvectors
`V_perp` (nullity columns). The constraint count encodes antisymmetry, trace, mixed,
and null-null constraints without materializing them as dense vectors — the
differentiation pipeline dispatches on `ActiveConstraints{AT, <:SpectraplexEqNormals}`
and works with `U`/`V_perp` directly via tangent space compress/expand operations.
"""
struct SpectraplexEqNormals{T<:Real, MT1<:AbstractMatrix{T}, MT2<:AbstractMatrix{T}}
    n::Int
    trace_rhs::T
    U::MT1
    V_perp::MT2
end

"""
    _spectraplex_sym_count(n) -> Int

Number of antisymmetry constraints for an ``n \\times n`` matrix: ``n(n-1)/2``.
"""
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

"""
    _spectraplex_min_eigen(g, n[, buf]) -> (λ_min, v_min)

Symmetrize the gradient (reshaped as n×n) and return the minimum eigenvalue and
eigenvector.  Shared by the oracle callable and `_lmo_and_gap!`.

The `buf` argument is an n×n pre-allocated matrix used for symmetrization and
destroyed by `eigen!`. When omitted, a fresh buffer is allocated.
"""
function _spectraplex_min_eigen(g::AbstractVector, n::Int, buf::AbstractMatrix)
    G = reshape(g, n, n)
    nan_seen = false
    @inbounds for j in 1:n
        for i in 1:n
            val = (G[i, j] + G[j, i]) / 2
            nan_seen |= (val != val)
            buf[i, j] = val
        end
    end
    if nan_seen
        @warn "Spectraplex oracle: NaN in symmetrized gradient; eigendecomposition may be unreliable" maxlog=3
    end
    E = eigen!(Symmetric(buf))
    return E.values[1], @view(E.vectors[:, 1])
end

function _spectraplex_min_eigen(g::AbstractVector, n::Int)
    return _spectraplex_min_eigen(g, n, Matrix{eltype(g)}(undef, n, n))
end

"""
    _spectraplex_write_rank1!(v, v_min, r, n)

Write the rank-1 vertex ``r \\, v_{\\min} v_{\\min}^\\top`` column major into `v`.
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
# Reuses c.direction (length n²) as the n×n symmetrization buffer — safe because
# c.direction is used only as scratch here; it is overwritten before its next
# use in the FW iteration (by _compute_step for AdaptiveStepSize, or unused
# for MonotonicStepSize).
function _lmo_and_gap!(lmo::Spectraplex, c::Cache{T}, x, m) where T
    buf = reshape(c.direction, lmo.n, lmo.n)
    λ_min, v_min = _spectraplex_min_eigen(c.gradient, lmo.n, buf)
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
    if g_min != g_min  # NaN check
        @warn "Simplex oracle: NaN in gradient" maxlog=3
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
# Product oracle (Cartesian product of constraint sets)
# ------------------------------------------------------------------

"""
    ProductOracle(block₁ => lmo₁, block₂ => lmo₂, ...)

Oracle for the Cartesian product ``C_1 \\times C_2 \\times \\cdots \\times C_k``
where each block ``x_j \\in C_j`` has its own independent constraint set.

Each block is specified as a `UnitRange{Int} => AbstractOracle` pair mapping
variable indices to their oracle. Blocks must be contiguous, non-overlapping,
and cover all variables.

# Example
```julia
# Variables 1:3 on probability simplex, variables 4:5 in [0,1] box
lmo = ProductOracle(1:3 => ProbSimplex(), 4:5 => Box(0.0, 1.0))
```
"""
struct ProductOracle{LT<:Tuple} <: AbstractOracle
    lmos::LT
    block_ranges::Vector{UnitRange{Int}}
    n::Int

    function ProductOracle(pairs::Pair{UnitRange{Int}, <:AbstractOracle}...)
        length(pairs) >= 2 || throw(ArgumentError("ProductOracle requires at least 2 blocks"))
        ranges = [p.first for p in pairs]
        lmos = Tuple(p.second for p in pairs)
        # Validate contiguous, non-overlapping coverage
        sorted = sort(ranges; by=first)
        for i in 2:length(sorted)
            last(sorted[i-1]) + 1 == first(sorted[i]) || throw(ArgumentError(
                "ProductOracle: blocks must be contiguous (gap between $(sorted[i-1]) and $(sorted[i]))"))
        end
        first(sorted[1]) == 1 || throw(ArgumentError(
            "ProductOracle: first block must start at index 1 (got $(first(sorted[1])))"))
        n = last(sorted[end])
        new{typeof(lmos)}(lmos, collect(ranges), n)
    end
end

function (lmo::ProductOracle)(v::AbstractVector, g::AbstractVector)
    for (rng, oracle) in zip(lmo.block_ranges, lmo.lmos)
        oracle(@view(v[rng]), @view(g[rng]))
    end
    return v
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
