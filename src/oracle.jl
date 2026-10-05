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

# Largest k handled by the insertion path; larger k use quickselect.
const _INSERTION_SELECT_MAX_K = 64

"""
    _partial_sort_negative!(perm, g, k) -> count

Find up to `k` indices with the most negative values in `g` and store them in
`perm[1:count]`. Only strictly negative entries are selected; NaN entries are
skipped with a warning. Ties are broken by the smaller index, so the selected
set is the `k` smallest entries under the order ``(g_i, i)``, whichever path
runs. Zero-allocation: only uses the pre-allocated `perm` buffer.

- `k ≤ 64`, or `perm` shorter than `g`: insertion sort, ``O(n \\cdot k)``, with
  `perm[1:count]` sorted by increasing `g`.
- `k > 64` and `length(perm) ≥ length(g)`: the negative indices are gathered in
  `perm` and a quickselect moves the `k` smallest to the front, expected
  ``O(n)``. The order within `perm[1:count]` is then unspecified.
"""
function _partial_sort_negative!(perm::Vector{Int}, g, k::Int)
    n = length(g)
    k = min(k, n)
    k <= 0 && return 0
    if k > _INSERTION_SELECT_MAX_K && length(perm) >= n
        return _quickselect_negative!(perm, g, k)
    end
    return _insertion_select_negative!(perm, g, k)
end

# Insertion path of _partial_sort_negative!: O(n·k), perm[1:count] sorted.
function _insertion_select_negative!(perm::Vector{Int}, g, k::Int)
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

# Quickselect path of _partial_sort_negative!: gather the strictly negative
# indices into perm, then move the k smallest under (g_i, i) to perm[1:k].
# Requires length(perm) ≥ length(g). Expected O(n), no allocation.
function _quickselect_negative!(perm::Vector{Int}, g, k::Int)
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
        if gi < zero(gi)
            count += 1
            perm[count] = i
        end
    end
    if nan_seen
        @warn "_partial_sort_negative!: NaN in gradient; affected entries skipped" maxlog=3
    end
    if count > k
        _quickselect_by_value!(perm, count, k, g)
        count = k
    end
    return count
end

# Strict total order on indices: by value, then by index. Distinct indices are
# never equal, so the k smallest form a unique set even with tied values.
@inline function _value_index_lt(g, i::Int, j::Int)
    @inbounds gi = g[i]
    @inbounds gj = g[j]
    return gi < gj || (gi == gj && i < j)
end

# Rearrange perm[1:hi] so that perm[1:k] hold its k smallest entries under
# _value_index_lt (Hoare's FIND with a median-of-three pivot).
function _quickselect_by_value!(perm::Vector{Int}, hi::Int, k::Int, g)
    lo = 1
    @inbounds while lo < hi
        mid = (lo + hi) >>> 1
        a = perm[lo]; b = perm[mid]; c = perm[hi]
        _value_index_lt(g, b, a) && ((a, b) = (b, a))
        if _value_index_lt(g, c, b)
            b = c
            _value_index_lt(g, b, a) && (b = a)
        end
        pivot = b
        i = lo
        j = hi
        while i <= j
            while _value_index_lt(g, perm[i], pivot)
                i += 1
            end
            while _value_index_lt(g, pivot, perm[j])
                j -= 1
            end
            if i <= j
                perm[i], perm[j] = perm[j], perm[i]
                i += 1
                j -= 1
            end
        end
        j < k && (lo = i)
        k < i && (hi = j)
    end
    return perm
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
only indices with strictly negative gradient are selected, and ties go to the
smaller index.
**Complexity**: zero-allocation. ``O(m \\cdot k)`` insertion sort for
``k = \\text{budget} \\le 64``; expected ``O(m)`` quickselect for larger budgets.
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
negative gradient are selected, and ties go to the smaller index.
**Complexity**: zero-allocation. ``O(m \\cdot k)`` insertion sort for ``k \\le 64``;
expected ``O(m)`` quickselect for larger ``k``.
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
# Pairwise oracle interface: away vertex and largest feasible step
# ------------------------------------------------------------------

"""
    away_vertex!(lmo, c::Cache, x, g; atol) -> c.away_vertex

Away-vertex oracle for `solve(...; variant=:pairwise)`. Writes into
`c.away_vertex` a vertex ``v^-`` of the smallest face of ``C`` containing `x`
that maximizes ``\\langle g, v \\rangle`` over that face, and returns it. Moving
from `x` away from ``v^-`` stays feasible for a positive step, and
``\\langle g, v^- \\rangle \\ge \\langle g, x \\rangle``, so the pairwise
direction ``v^+ - v^-`` is a descent direction whenever the Frank-Wolfe gap is
positive. No active set is stored.

`atol` is the tolerance below which a distance to a bound counts as zero, so
that rounding residues (a coordinate left at ``10^{-17}`` instead of 0) do not
define the face; `solve` passes its `pairwise_atol`. Methods that have no use
for it may omit the keyword.

Implemented for [`MaskedKnapsack`](@ref). To use the pairwise variant with
another oracle, define this method and [`pairwise_max_step`](@ref) for it.
"""
function away_vertex! end

"""
    pairwise_max_step(lmo, x, d) -> γ_max

Largest ``\\gamma \\ge 0`` with ``x + \\gamma d \\in C``, for a pairwise
direction `d`. Returns `0` when `d` is zero. Implemented for
[`MaskedKnapsack`](@ref); see [`away_vertex!`](@ref).
"""
function pairwise_max_step end

# Negated view used to select the largest entries with the smallest-first quickselect.
struct _NegatedVector{T, V<:AbstractVector{T}} <: AbstractVector{T}
    parent::V
end
Base.size(a::_NegatedVector) = size(a.parent)
Base.@propagate_inbounds Base.getindex(a::_NegatedVector, i::Int) = -a.parent[i]

"""
    away_vertex!(lmo::MaskedKnapsack, c::Cache, x, g; atol=sqrt(eps(T)))

For the masked knapsack the smallest face containing `x` fixes the masked
coordinates and the optional coordinates with ``x_e = 1`` to 1, fixes those
with ``x_e = 0`` to 0, and, when ``\\sum_e x_e`` equals the budget, keeps the
budget tight. The away vertex therefore sets those fixed coordinates and fills
the remaining slots with the fractional coordinates of largest gradient:

- budget tight: all ``k - |\\{e : x_e = 1\\}|`` slots, whatever the sign of the
  gradient (fewer if fewer fractional coordinates exist);
- budget slack: only fractional coordinates with positive gradient, up to
  the same number of slots.

The tests use the tolerance `atol` (default ``\\sqrt{\\varepsilon}``, about
``1.5 \\times 10^{-8}`` in `Float64`; coordinates live in ``[0, 1]``, so no
further scaling is applied): ``x_e \\le \\text{atol}`` counts as 0,
``x_e \\ge 1 - \\text{atol}`` counts as 1, and the budget counts as tight when
``\\text{budget} - \\sum_e x_e \\le \\max(m\\varepsilon, \\text{atol}) \\max(1, \\text{budget})``.
Without these tolerances a coordinate left at ``10^{-17}`` by rounding would
stay in the away support and cap every later step at ``10^{-17}``. With these
tests every coordinate that the pairwise direction moves is more than `atol`
from the bound it moves toward, and a slack budget leaves more than `atol` of
room per unit of added mass, so [`pairwise_max_step`](@ref) returns more than
`atol`. The remaining degenerate cases are left to `solve`, which then takes a
Frank-Wolfe step.

On the face ``\\sum_e x_e = \\text{budget}`` this is the masked set plus the
`k` optional coordinates of largest gradient among those with ``x_e > \\text{atol}``,
with every coordinate at (or within `atol` of) 1 included. Ties go to the
smaller index. Zero-allocation (the oracle's `perm` buffer is reused); ``O(m)`` expected.
"""
function away_vertex!(lmo::MaskedKnapsack, c::Cache{T}, x::AbstractVector,
                      g::AbstractVector; atol::Real=sqrt(eps(T))) where {T}
    v = c.away_vertex
    m = length(x)
    length(lmo.is_masked) == m ||
        throw(DimensionMismatch("away_vertex!: oracle dimension $(length(lmo.is_masked)) ≠ length(x) = $m"))
    tol = T(atol)
    budget = T(lmo.k + lmo.n_masked)
    sum_x = zero(T)
    @inbounds @simd for e in 1:m
        sum_x += x[e]
    end
    tight = sum_x ≥ budget - max(m * eps(T), tol) * max(one(T), budget)

    # Fixed coordinates, and the fractional candidates gathered in lmo.perm
    n_ones = 0
    n_cand = 0
    perm = lmo.perm
    @inbounds for e in 1:m
        if lmo.is_masked[e]
            v[e] = one(T)
        elseif x[e] ≥ one(T) - tol
            v[e] = one(T)
            n_ones += 1
        else
            v[e] = zero(T)
            if x[e] > tol && (tight || g[e] > zero(g[e]))
                n_cand += 1
                perm[n_cand] = e
            end
        end
    end

    slots = max(lmo.k - n_ones, 0)
    if n_cand > slots
        slots > 0 && _quickselect_by_value!(perm, n_cand, slots, _NegatedVector(g))
        n_cand = slots
    end
    @inbounds for j in 1:n_cand
        v[perm[j]] = one(T)
    end
    return v
end

"""
    pairwise_max_step(lmo::MaskedKnapsack, x, d)

``\\min\\big(\\min_{d_e < 0} x_e / (-d_e),\\; \\min_{d_e > 0} (1 - x_e) / d_e\\big)``,
further capped by the slack ``(\\text{budget} - \\sum_e x_e) / \\sum_e d_e`` when
``\\sum_e d_e > 0``.
"""
function pairwise_max_step(lmo::MaskedKnapsack, x::AbstractVector{T}, d::AbstractVector) where {T}
    γ_max = T(Inf)
    sum_x = zero(T)
    sum_d = zero(T)
    @inbounds for e in eachindex(x, d)
        de = d[e]
        xe = x[e]
        sum_x += xe
        sum_d += de
        if de < zero(de)
            γ_max = min(γ_max, max(xe, zero(T)) / -de)
        elseif de > zero(de)
            γ_max = min(γ_max, max(one(T) - xe, zero(T)) / de)
        end
    end
    isfinite(γ_max) || return zero(T)   # d == 0
    if sum_d > zero(T)
        slack = max(T(lmo.k + lmo.n_masked) - sum_x, zero(T))
        γ_max = min(γ_max, slack / sum_d)
    end
    return γ_max
end

# Snap a pairwise trial point: coordinates that the step decreased (d_e < 0)
# to below `atol` are set to exactly 0, so rounding residues and tiny negative
# values do not survive. Oracles without a method do nothing.
_pairwise_snap!(lmo, buffer, d, atol) = buffer
function _pairwise_snap!(::MaskedKnapsack, buffer::AbstractVector{T}, d::AbstractVector, atol) where {T}
    tol = T(atol)
    @inbounds for e in eachindex(buffer, d)
        if d[e] < zero(eltype(d)) && buffer[e] < tol
            buffer[e] = zero(T)
        end
    end
    return buffer
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
    nan_seen = false
    @inbounds @simd for i in eachindex(v, g, lmo.lb, lmo.ub)
        gi = g[i]
        nan_seen |= isnan(gi)
        v[i] = gi >= zero(gi) ? lmo.lb[i] : lmo.ub[i]
    end
    nan_seen && @warn "Box oracle: NaN in gradient" maxlog=3
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
    nan_seen = false
    @inbounds @simd for i in eachindex(v, g)
        gi = g[i]
        nan_seen |= isnan(gi)
        v[i] = gi >= zero(gi) ? lmo.lb : lmo.ub
    end
    nan_seen && @warn "ScalarBox oracle: NaN in gradient" maxlog=3
    return v
end

# ScalarBox fused LMO + gap
_lmo_and_gap!(::KernelAbstractions.CPU, lmo::ScalarBox, c::Cache, x, n) = _dense_lmo_and_gap!(lmo, c, x, n)

function _lmo_and_gap!(::KernelAbstractions.Backend, lmo::ScalarBox, c::Cache{T}, x, n) where T
    c.vertex .= ifelse.(c.gradient .>= zero(T), lmo.lb, lmo.ub)
    fw_gap = dot(c.gradient, x) - dot(c.gradient, c.vertex)
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

function WeightedSimplex(α::AbstractVector{T}, β::Real, lb::AbstractVector{T}) where {T<:AbstractFloat}
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
    nan_seen = false

    @inbounds for i in eachindex(g, lmo.α)
        gi = g[i]
        if isnan(gi)
            nan_seen = true
            continue
        end
        if gi < zero(gi)
            ratio = gi / lmo.α[i]
            if ratio < best_ratio
                best_ratio = ratio
                best_idx = i
            end
        end
    end

    if best_idx > 0
        @inbounds v[best_idx] = lmo.β_bar / lmo.α[best_idx] + lmo.lb[best_idx]
    end

    nan_seen && @warn "WeightedSimplex oracle: NaN in gradient" maxlog=3
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

Dispatches on `KernelAbstractions.get_backend(x)` to select CPU or GPU code paths.
"""
_lmo_and_gap!(lmo, c::Cache, x, n) = _lmo_and_gap!(KernelAbstractions.get_backend(x), lmo, c, x, n)

"""
    _gpu_unsupported(oracle_name)

Throw an informative error for oracles not yet supported on GPU arrays.
"""
function _gpu_unsupported(oracle_name::String)
    throw(ArgumentError(
        "$oracle_name oracle is not yet supported with GPU arrays. " *
        "Use Box, ScalarBox, or ProbSimplex instead, or copy data to CPU."))
end

# Shared dense gap computation: call oracle, accumulate ⟨g, x-v⟩
@inline function _dense_lmo_and_gap!(lmo, c::Cache{T}, x, n) where T
    lmo(c.vertex, c.gradient)
    fw_gap = zero(T)
    @inbounds @simd for i in 1:n
        fw_gap += c.gradient[i] * (x[i] - c.vertex[i])
    end
    return (fw_gap, -1)
end

# ------------------------------------------------------------------
# Generic fallback (FunctionOracle or any unspecialized AbstractOracle)
# ------------------------------------------------------------------

function _lmo_and_gap!(::KernelAbstractions.CPU, lmo, c::Cache{T}, x, n) where T
    _dense_lmo_and_gap!(lmo, c, x, n)
end

function _lmo_and_gap!(::KernelAbstractions.Backend, lmo, c::Cache{T}, x, n) where T
    lmo(c.vertex, c.gradient)
    fw_gap = dot(c.gradient, x) - dot(c.gradient, c.vertex)
    return (fw_gap, -1)
end

# ------------------------------------------------------------------
# Simplex
# ------------------------------------------------------------------

function _lmo_and_gap!(::KernelAbstractions.CPU, lmo::Simplex{ST, Equality}, c::Cache{T}, x, n) where {ST, T, Equality}
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

function _lmo_and_gap!(::KernelAbstractions.Backend, lmo::Simplex{ST, Equality}, c::Cache{T}, x, n) where {ST, T, Equality}
    g = c.gradient
    g_min = minimum(g)
    i_star = argmin(g)
    dot_gx = dot(g, x)
    if Equality || g_min < zero(g_min)
        c.vertex .= ifelse.(eachindex(c.vertex) .== i_star, T(lmo.r), zero(T))
        return (dot_gx - T(lmo.r) * T(g_min), -1)
    else
        fill!(c.vertex, zero(T))
        return (dot_gx, -1)
    end
end

# ------------------------------------------------------------------
# Box (vector bounds) — CPU only, GPU unsupported
# ------------------------------------------------------------------

_lmo_and_gap!(::KernelAbstractions.CPU, lmo::Box, c::Cache{T}, x, n) where T = _dense_lmo_and_gap!(lmo, c, x, n)

_lmo_and_gap!(::KernelAbstractions.Backend, lmo::Box, c::Cache{T}, x, n) where T = _gpu_unsupported("Box")

# ------------------------------------------------------------------
# WeightedSimplex — CPU only, GPU unsupported
# ------------------------------------------------------------------

_lmo_and_gap!(::KernelAbstractions.CPU, lmo::WeightedSimplex, c::Cache{T}, x, n) where T = _dense_lmo_and_gap!(lmo, c, x, n)

_lmo_and_gap!(::KernelAbstractions.Backend, lmo::WeightedSimplex, c::Cache{T}, x, n) where T = _gpu_unsupported("WeightedSimplex")

# ------------------------------------------------------------------
# Knapsack — CPU only, GPU unsupported
# ------------------------------------------------------------------

function _lmo_and_gap!(::KernelAbstractions.CPU, lmo::Knapsack, c::Cache{T}, x, n) where T
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

_lmo_and_gap!(::KernelAbstractions.Backend, lmo::Knapsack, c::Cache{T}, x, n) where T = _gpu_unsupported("Knapsack")

# ------------------------------------------------------------------
# MaskedKnapsack — CPU only, GPU unsupported
# ------------------------------------------------------------------

function _lmo_and_gap!(::KernelAbstractions.CPU, lmo::MaskedKnapsack, c::Cache{T}, x, n) where T
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

_lmo_and_gap!(::KernelAbstractions.Backend, lmo::MaskedKnapsack, c::Cache{T}, x, n) where T = _gpu_unsupported("MaskedKnapsack")

# ------------------------------------------------------------------
# Spectraplex — CPU only, GPU unsupported
# ------------------------------------------------------------------

function _lmo_and_gap!(::KernelAbstractions.CPU, lmo::Spectraplex, c::Cache{T}, x, m) where T
    buf = reshape(c.direction, lmo.n, lmo.n)
    λ_min, v_min = _spectraplex_min_eigen(c.gradient, lmo.n, buf)
    _spectraplex_write_rank1!(c.vertex, v_min, lmo.r, lmo.n)
    fw_gap = dot(c.gradient, x) - T(lmo.r) * T(λ_min)
    return (fw_gap, -1)
end

_lmo_and_gap!(::KernelAbstractions.Backend, lmo::Spectraplex, c::Cache{T}, x, n) where T = _gpu_unsupported("Spectraplex")

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
