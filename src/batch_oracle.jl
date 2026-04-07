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
# Batched LMO + gap computation
# ------------------------------------------------------------------
# All methods write into c.vertex[:, b] and c.gap[b] for all b in 1:B.
# Always dense (no sparse vertex protocol in batch mode).
#
# GPU paths use broadcast operations and reductions to avoid scalar
# indexing. CPU paths retain @inbounds @simd loops for performance.

"""
    _batch_lmo_and_gap!(lmo, c::BatchCache, X)

Batched linear minimization oracle + Frank-Wolfe gap computation.
Column-wise: for each problem `b`, compute the LMO vertex and FW gap.

Dispatches on [`_array_style`](@ref) for GPU-compatible paths where available.
"""
function _batch_lmo_and_gap! end

# ------------------------------------------------------------------
# ScalarBox
# ------------------------------------------------------------------

function _batch_lmo_and_gap!(lmo::ScalarBox{ST}, c::BatchCache{T}, X) where {ST, T}
    _batch_lmo_and_gap!(_array_style(X), lmo, c, X)
end

function _batch_lmo_and_gap!(::_CPUStyle, lmo::ScalarBox{ST}, c::BatchCache{T}, X) where {ST, T}
    G = c.gradient
    V = c.vertex
    n, B = size(X)
    @. V = ifelse(G >= zero(T), T(lmo.lb), T(lmo.ub))
    @inbounds for b in 1:B
        gap = zero(T)
        @simd for i in 1:n
            gap += G[i, b] * (X[i, b] - V[i, b])
        end
        c.gap[b] = gap
    end
end

function _batch_lmo_and_gap!(::_GPUStyle, lmo::ScalarBox{ST}, c::BatchCache{T}, X) where {ST, T}
    @. c.vertex = ifelse(c.gradient >= zero(T), T(lmo.lb), T(lmo.ub))
    gap_mat = c.gradient .* (X .- c.vertex)
    _sum_columns_to_gap!(c.gap, gap_mat)
end

# ------------------------------------------------------------------
# Box (vector bounds)
# ------------------------------------------------------------------

function _batch_lmo_and_gap!(lmo::Box{BT}, c::BatchCache{T}, X) where {BT, T}
    _batch_lmo_and_gap!(_array_style(X), lmo, c, X)
end

function _batch_lmo_and_gap!(::_CPUStyle, lmo::Box{BT}, c::BatchCache{T}, X) where {BT, T}
    G = c.gradient
    V = c.vertex
    n, B = size(X)
    @inbounds for b in 1:B
        gap = zero(T)
        @simd for i in 1:n
            vi = G[i, b] >= zero(T) ? T(lmo.lb[i]) : T(lmo.ub[i])
            V[i, b] = vi
            gap += G[i, b] * (X[i, b] - vi)
        end
        c.gap[b] = gap
    end
end

function _batch_lmo_and_gap!(::_GPUStyle, lmo::Box{BT}, c::BatchCache{T}, X) where {BT, T}
    lb = reshape(T.(lmo.lb), :, 1)
    ub = reshape(T.(lmo.ub), :, 1)
    @. c.vertex = ifelse(c.gradient >= zero(T), lb, ub)
    gap_mat = c.gradient .* (X .- c.vertex)
    _sum_columns_to_gap!(c.gap, gap_mat)
end

# ------------------------------------------------------------------
# ProbSimplex (Equality=true)
# ------------------------------------------------------------------

function _batch_lmo_and_gap!(lmo::Simplex{ST, true}, c::BatchCache{T}, X) where {ST, T}
    _batch_lmo_and_gap!(_array_style(X), lmo, c, X)
end

function _batch_lmo_and_gap!(::_CPUStyle, lmo::Simplex{ST, true}, c::BatchCache{T}, X) where {ST, T}
    G = c.gradient
    V = c.vertex
    n, B = size(X)
    fill!(V, zero(T))
    @inbounds for b in 1:B
        i_star = 1
        g_min = G[1, b]
        dot_gx = zero(T)
        for i in 1:n
            gi = G[i, b]
            dot_gx += gi * X[i, b]
            if gi < g_min
                g_min = gi
                i_star = i
            end
        end
        V[i_star, b] = T(lmo.r)
        c.gap[b] = dot_gx - T(lmo.r) * g_min
    end
end

function _batch_lmo_and_gap!(::_GPUStyle, lmo::Simplex{ST, true}, c::BatchCache{T}, X) where {ST, T}
    G = c.gradient
    n, B = size(X)
    g_mins = minimum(G, dims=1)        # (1, B) — on device
    g_mins_cpu = g_mins isa Array ? g_mins : Array(g_mins)  # single D2H transfer
    i_stars = _argmin_cols(G)          # (1, B) indices
    row_idx = reshape(1:n, n, 1)
    @. c.vertex = ifelse(row_idx == i_stars, T(lmo.r), zero(T))
    gap_mat = G .* X
    _sum_columns_to_gap!(c.gap, gap_mat)
    @inbounds for b in 1:B
        c.gap[b] -= T(lmo.r) * g_mins_cpu[1, b]
    end
end

# ------------------------------------------------------------------
# Capped Simplex (Equality=false)
# ------------------------------------------------------------------

function _batch_lmo_and_gap!(lmo::Simplex{ST, false}, c::BatchCache{T}, X) where {ST, T}
    _batch_lmo_and_gap!(_array_style(X), lmo, c, X)
end

function _batch_lmo_and_gap!(::_CPUStyle, lmo::Simplex{ST, false}, c::BatchCache{T}, X) where {ST, T}
    G = c.gradient
    V = c.vertex
    n, B = size(X)
    fill!(V, zero(T))
    @inbounds for b in 1:B
        i_star = 1
        g_min = G[1, b]
        dot_gx = zero(T)
        for i in 1:n
            gi = G[i, b]
            dot_gx += gi * X[i, b]
            if gi < g_min
                g_min = gi
                i_star = i
            end
        end
        if g_min < zero(g_min)
            V[i_star, b] = T(lmo.r)
            c.gap[b] = dot_gx - T(lmo.r) * g_min
        else
            c.gap[b] = dot_gx
        end
    end
end

function _batch_lmo_and_gap!(::_GPUStyle, lmo::Simplex{ST, false}, c::BatchCache{T}, X) where {ST, T}
    G = c.gradient
    n, B = size(X)
    g_mins = minimum(G, dims=1)        # (1, B) — on device
    g_mins_cpu = g_mins isa Array ? g_mins : Array(g_mins)  # single D2H transfer
    i_stars = _argmin_cols(G)          # (1, B) indices
    row_idx = reshape(1:n, n, 1)
    neg_mask = g_mins .< zero(T)       # (1, B) Bool — on device
    @. c.vertex = ifelse((row_idx == i_stars) & neg_mask, T(lmo.r), zero(T))
    gap_mat = G .* X
    _sum_columns_to_gap!(c.gap, gap_mat)
    @inbounds for b in 1:B
        if g_mins_cpu[1, b] < zero(T)
            c.gap[b] -= T(lmo.r) * g_mins_cpu[1, b]
        end
    end
end

# ------------------------------------------------------------------
# Knapsack (CPU only)
# ------------------------------------------------------------------

function _batch_lmo_and_gap!(lmo::Knapsack, c::BatchCache{T}, X) where T
    _array_style(X) isa _GPUStyle && throw(ArgumentError(
        "Knapsack oracle does not support GPU arrays in batch_solve. " *
        "Use ScalarBox, Box, or Simplex for GPU-accelerated batch solving."))
    G = c.gradient
    V = c.vertex
    n, B = size(X)
    fill!(V, zero(T))
    @inbounds for b in 1:B
        dot_gx = zero(T)
        @simd for i in 1:n
            dot_gx += G[i, b] * X[i, b]
        end
        if lmo.k <= 0
            c.gap[b] = dot_gx
            continue
        end
        g_col = @view(G[:, b])
        count = _partial_sort_negative!(lmo.perm, g_col, lmo.k)
        vertex_contrib = zero(T)
        for j in 1:count
            idx = lmo.perm[j]
            V[idx, b] = one(T)
            vertex_contrib += G[idx, b]
        end
        c.gap[b] = dot_gx - vertex_contrib
    end
end

# ------------------------------------------------------------------
# Generic fallback (CPU only)
# ------------------------------------------------------------------

function _batch_lmo_and_gap!(lmo::AbstractOracle, c::BatchCache{T}, X) where T
    _array_style(X) isa _GPUStyle && throw(ArgumentError(
        "Generic oracle fallback does not support GPU arrays in batch_solve. " *
        "Provide a specialized _batch_lmo_and_gap! method for $(typeof(lmo)), " *
        "or use a built-in oracle (ScalarBox, Box, Simplex)."))
    n, B = size(X)
    v_buf = Vector{T}(undef, n)
    g_buf = Vector{T}(undef, n)
    @inbounds for b in 1:B
        copyto!(g_buf, @view(c.gradient[:, b]))
        fill!(v_buf, zero(T))
        lmo(v_buf, g_buf)
        gap = zero(T)
        @simd for i in 1:n
            c.vertex[i, b] = v_buf[i]
            gap += g_buf[i] * (X[i, b] - v_buf[i])
        end
        c.gap[b] = gap
    end
end

# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

"""
    _sum_columns_to_gap!(gap::Vector, M::AbstractMatrix)

Sum each column of `M` into `gap`. Single GPU-to-CPU transfer for the
`(1, B)` reduction result via `Array(sums)`.
"""
function _sum_columns_to_gap!(gap::Vector{T}, M::AbstractMatrix{T}) where T
    sums = sum(M, dims=1)  # (1, B) — computed on device if M is GPU
    sums_cpu = sums isa Array ? sums : Array(sums)  # single D2H transfer
    @inbounds for b in eachindex(gap)
        gap[b] = sums_cpu[1, b]
    end
end

"""
    _argmin_cols(G::AbstractMatrix)

Column-wise argmin. Returns a `(1, B)` matrix of row indices.
GPU-safe: uses `findmin` reduction along dim=1.
"""
function _argmin_cols(G::AbstractMatrix)
    _, idx = findmin(G, dims=1)   # (1, B) CartesianIndex
    return map(ci -> ci[1], idx)  # (1, B) Int — extract row index
end
