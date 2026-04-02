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

"""
    _batch_lmo_and_gap!(lmo, c::BatchCache, X)

Batched linear minimization oracle + Frank-Wolfe gap computation.
Column-wise: for each problem `b`, compute the LMO vertex and FW gap.
"""
function _batch_lmo_and_gap! end

# ScalarBox: v[i,b] = lb if g[i,b] >= 0, ub otherwise
function _batch_lmo_and_gap!(lmo::ScalarBox{ST}, c::BatchCache{T}, X) where {ST, T}
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

# Box: v[i,b] = lb[i] if g[i,b] >= 0, ub[i] otherwise
function _batch_lmo_and_gap!(lmo::Box{BT}, c::BatchCache{T}, X) where {BT, T}
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

# ProbSimplex (Equality=true): column-wise argmin + gap
function _batch_lmo_and_gap!(lmo::Simplex{ST, true}, c::BatchCache{T}, X) where {ST, T}
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

# Capped Simplex (Equality=false): column-wise argmin + gap, with origin vertex
function _batch_lmo_and_gap!(lmo::Simplex{ST, false}, c::BatchCache{T}, X) where {ST, T}
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

# Knapsack: column-wise partial sort (CPU only)
function _batch_lmo_and_gap!(lmo::Knapsack, c::BatchCache{T}, X) where T
    G = c.gradient
    V = c.vertex
    n, B = size(X)
    perm = Vector{Int}(undef, n)
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
        count = _partial_sort_negative!(perm, g_col, lmo.k)
        vertex_contrib = zero(T)
        for j in 1:count
            idx = perm[j]
            V[idx, b] = one(T)
            vertex_contrib += G[idx, b]
        end
        c.gap[b] = dot_gx - vertex_contrib
    end
end

# Fallback: column-wise single-problem oracle call
function _batch_lmo_and_gap!(lmo::AbstractOracle, c::BatchCache{T}, X) where T
    n, B = size(X)
    v_buf = zeros(T, n)
    g_buf = zeros(T, n)
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
