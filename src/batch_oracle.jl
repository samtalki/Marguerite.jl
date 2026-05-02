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

# Batched LMO + gap. Writes c.vertex[:, b] and c.gap[b] for all b. GPU paths
# use broadcast + reductions; CPU paths use @inbounds @simd loops. Always dense.

"""
    _batch_lmo_and_gap!(lmo, c::BatchCache, X)

Batched linear minimization oracle + Frank-Wolfe gap computation.
Column-wise: for each problem `b`, compute the LMO vertex and FW gap.

Dispatches on [`_array_style`](@ref) to choose the device path when one
exists for `lmo`.
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
    @. c.x_trial = c.gradient * (X - c.vertex)
    _sum_columns_to_gap!(c.gap, c.x_trial)
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
    lb = reshape(eltype(lmo.lb) === T ? lmo.lb : convert(Vector{T}, lmo.lb), :, 1)
    ub = reshape(eltype(lmo.ub) === T ? lmo.ub : convert(Vector{T}, lmo.ub), :, 1)
    @. c.vertex = ifelse(c.gradient >= zero(T), lb, ub)
    @. c.x_trial = c.gradient * (X - c.vertex)
    _sum_columns_to_gap!(c.gap, c.x_trial)
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
    g_mins, idx = findmin(G, dims=1)
    i_stars = map(ci -> ci[1], idx)
    row_idx = reshape(1:n, n, 1)
    @. c.vertex = ifelse(row_idx == i_stars, T(lmo.r), zero(T))
    @. c.x_trial = G * X
    # gap_b = ⟨g_b, x_b⟩ − r·g_min_b. Fuse on device, single transfer to c.gap.
    gap_dev = sum(c.x_trial, dims=1) .- T(lmo.r) .* g_mins
    _copy_row_to_gap!(c.gap, gap_dev)
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
    g_mins, idx = findmin(G, dims=1)
    i_stars = map(ci -> ci[1], idx)
    row_idx = reshape(1:n, n, 1)
    neg_mask = g_mins .< zero(T)
    @. c.vertex = ifelse((row_idx == i_stars) & neg_mask, T(lmo.r), zero(T))
    @. c.x_trial = G * X
    # gap_b = ⟨g_b, x_b⟩ − (r·g_min_b if g_min_b < 0 else 0). Fuse on device.
    correction = ifelse.(neg_mask, T(lmo.r) .* g_mins, zero(T))
    gap_dev = sum(c.x_trial, dims=1) .- correction
    _copy_row_to_gap!(c.gap, gap_dev)
end

# ------------------------------------------------------------------
# Knapsack (CPU only)
# ------------------------------------------------------------------

function _batch_lmo_and_gap!(lmo::Knapsack, c::BatchCache{T}, X) where T
    _array_style(X) isa _GPUStyle && throw(ArgumentError(
        "Knapsack oracle does not support device arrays in batch_solve. " *
        "Use ScalarBox, Box, or Simplex for the device path."))
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

Sum each column of `M` into `gap`. Single device-to-host transfer for the
`(1, B)` reduction result via `Array(sums)`.
"""
function _sum_columns_to_gap!(gap::Vector{T}, M::AbstractMatrix{T}) where T
    sums = sum(M, dims=1)
    _copy_row_to_gap!(gap, sums)
end

"""
    _copy_row_to_gap!(gap::Vector, row::AbstractMatrix)

Copy a `(1, B)` row matrix into `gap`. One device-to-host transfer when
`row` lives on the device. Lets oracle paths fuse the column reduction and
any per-column correction into a single transfer.
"""
function _copy_row_to_gap!(gap::Vector{T}, row) where T
    row_cpu = row isa Array ? row : Array(row)
    @inbounds for b in eachindex(gap)
        gap[b] = row_cpu[1, b]
    end
end
