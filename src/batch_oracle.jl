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
    _batch_lmo_and_gap!(lmo, c::BatchCache, X)

Batched linear minimization oracle and Frank-Wolfe gap. For each active
problem `b`, computes the LMO vertex into `c.vertex[:, b]` and the FW gap
into `c.gap[b]`. Dispatches on `KernelAbstractions.get_backend(X)`.
"""
_batch_lmo_and_gap!(lmo, c::BatchCache, X) =
    _batch_lmo_and_gap!(KernelAbstractions.get_backend(X), lmo, c, X)

# ------------------------------------------------------------------
# ScalarBox
# ------------------------------------------------------------------

function _batch_lmo_and_gap!(::KernelAbstractions.CPU, lmo::ScalarBox{ST}, c::BatchCache{T}, X) where {ST, T}
    G = c.gradient; V = c.vertex
    n, B = size(X)
    @inbounds for b in 1:B
        c.active[b] || continue
        gap = zero(T)
        @simd for i in 1:n
            v = G[i, b] >= zero(T) ? T(lmo.lb) : T(lmo.ub)
            V[i, b] = v
            gap += G[i, b] * (X[i, b] - v)
        end
        c.gap[b] = gap
    end
end

@kernel function _scalarbox_kernel!(V, gap, @Const(G), @Const(X), lb, ub, @Const(active_idx))
    b = @index(Global)
    b_root = @inbounds active_idx[b]
    n = size(G, 1)
    g_local = zero(eltype(gap))
    @inbounds for i in 1:n
        v = G[i, b_root] >= zero(eltype(G)) ? lb : ub
        V[i, b_root] = v
        g_local += G[i, b_root] * (X[i, b_root] - v)
    end
    @inbounds gap[b_root] = g_local
end

function _batch_lmo_and_gap!(backend::KernelAbstractions.Backend, lmo::ScalarBox{ST}, c::BatchCache{T}, X) where {ST, T}
    nactive = c.nactive[]
    nactive == 0 && return
    _scalarbox_kernel!(backend)(c.vertex, c.gap, c.gradient, X,
                                T(lmo.lb), T(lmo.ub), c.active_indices_dev;
                                ndrange=nactive)
    KernelAbstractions.synchronize(backend)
end

# ------------------------------------------------------------------
# Box (vector bounds)
# ------------------------------------------------------------------

function _batch_lmo_and_gap!(::KernelAbstractions.CPU, lmo::Box{BT}, c::BatchCache{T}, X) where {BT, T}
    G = c.gradient; V = c.vertex
    n, B = size(X)
    @inbounds for b in 1:B
        c.active[b] || continue
        gap = zero(T)
        @simd for i in 1:n
            vi = G[i, b] >= zero(T) ? T(lmo.lb[i]) : T(lmo.ub[i])
            V[i, b] = vi
            gap += G[i, b] * (X[i, b] - vi)
        end
        c.gap[b] = gap
    end
end

@kernel function _box_kernel!(V, gap, @Const(G), @Const(X), @Const(lb), @Const(ub), @Const(active_idx))
    b = @index(Global)
    b_root = @inbounds active_idx[b]
    n = size(G, 1)
    g_local = zero(eltype(gap))
    @inbounds for i in 1:n
        v = G[i, b_root] >= zero(eltype(G)) ? lb[i] : ub[i]
        V[i, b_root] = v
        g_local += G[i, b_root] * (X[i, b_root] - v)
    end
    @inbounds gap[b_root] = g_local
end

function _batch_lmo_and_gap!(backend::KernelAbstractions.Backend, lmo::Box{BT}, c::BatchCache{T}, X) where {BT, T}
    nactive = c.nactive[]
    nactive == 0 && return
    lb, ub = _cached_box_bounds(c, backend, lmo, T)
    _box_kernel!(backend)(c.vertex, c.gap, c.gradient, X, lb, ub, c.active_indices_dev;
                          ndrange=nactive)
    KernelAbstractions.synchronize(backend)
end

# Cache device-resident lb/ub on the BatchCache so we don't re-upload every
# FW iter. Recomputed when the lmo identity, backend, or eltype changes.
function _cached_box_bounds(c::BatchCache, backend, lmo::Box, ::Type{T}) where T
    cached = c.oracle_dev_cache[]
    if cached isa Tuple{Box, Any, Any, Any, DataType} && cached[1] === lmo &&
       cached[2] === backend && cached[5] === T
        return cached[3], cached[4]
    end
    lb = adapt(backend, eltype(lmo.lb) === T ? lmo.lb : convert(Vector{T}, lmo.lb))
    ub = adapt(backend, eltype(lmo.ub) === T ? lmo.ub : convert(Vector{T}, lmo.ub))
    c.oracle_dev_cache[] = (lmo, backend, lb, ub, T)
    return lb, ub
end

# ------------------------------------------------------------------
# Probability simplex (Equality=true)
# ------------------------------------------------------------------

function _batch_lmo_and_gap!(::KernelAbstractions.CPU, lmo::Simplex{ST, true}, c::BatchCache{T}, X) where {ST, T}
    G = c.gradient; V = c.vertex
    n, B = size(X)
    fill!(V, zero(T))
    @inbounds for b in 1:B
        c.active[b] || continue
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

@kernel function _simplex_eq_kernel!(V, gap, @Const(G), @Const(X), r, @Const(active_idx))
    b = @index(Global)
    b_root = @inbounds active_idx[b]
    n = size(G, 1)
    @inbounds begin
        i_star = 1
        g_min = G[1, b_root]
        dot_gx = G[1, b_root] * X[1, b_root]
        for i in 2:n
            gi = G[i, b_root]
            dot_gx += gi * X[i, b_root]
            if gi < g_min
                g_min = gi
                i_star = i
            end
        end
        for i in 1:n
            V[i, b_root] = (i == i_star) ? r : zero(r)
        end
        gap[b_root] = dot_gx - r * g_min
    end
end

function _batch_lmo_and_gap!(backend::KernelAbstractions.Backend, lmo::Simplex{ST, true}, c::BatchCache{T}, X) where {ST, T}
    nactive = c.nactive[]
    nactive == 0 && return
    _simplex_eq_kernel!(backend)(c.vertex, c.gap, c.gradient, X,
                                  T(lmo.r), c.active_indices_dev; ndrange=nactive)
    KernelAbstractions.synchronize(backend)
end

# ------------------------------------------------------------------
# Capped simplex (Equality=false)
# ------------------------------------------------------------------

function _batch_lmo_and_gap!(::KernelAbstractions.CPU, lmo::Simplex{ST, false}, c::BatchCache{T}, X) where {ST, T}
    G = c.gradient; V = c.vertex
    n, B = size(X)
    @inbounds for b in 1:B
        c.active[b] || continue
        @simd for i in 1:n
            V[i, b] = zero(T)
        end
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

@kernel function _simplex_capped_kernel!(V, gap, @Const(G), @Const(X), r, @Const(active_idx))
    b = @index(Global)
    b_root = @inbounds active_idx[b]
    n = size(G, 1)
    @inbounds begin
        i_star = 1
        g_min = G[1, b_root]
        dot_gx = G[1, b_root] * X[1, b_root]
        for i in 2:n
            gi = G[i, b_root]
            dot_gx += gi * X[i, b_root]
            if gi < g_min
                g_min = gi
                i_star = i
            end
        end
        if g_min < zero(g_min)
            for i in 1:n
                V[i, b_root] = (i == i_star) ? r : zero(r)
            end
            gap[b_root] = dot_gx - r * g_min
        else
            for i in 1:n
                V[i, b_root] = zero(r)
            end
            gap[b_root] = dot_gx
        end
    end
end

function _batch_lmo_and_gap!(backend::KernelAbstractions.Backend, lmo::Simplex{ST, false}, c::BatchCache{T}, X) where {ST, T}
    nactive = c.nactive[]
    nactive == 0 && return
    _simplex_capped_kernel!(backend)(c.vertex, c.gap, c.gradient, X,
                                      T(lmo.r), c.active_indices_dev; ndrange=nactive)
    KernelAbstractions.synchronize(backend)
end

# ------------------------------------------------------------------
# Knapsack (CPU only)
# ------------------------------------------------------------------

function _batch_lmo_and_gap!(::KernelAbstractions.CPU, lmo::Knapsack, c::BatchCache{T}, X) where T
    G = c.gradient; V = c.vertex
    n, B = size(X)
    @inbounds for b in 1:B
        c.active[b] || continue
        @simd for i in 1:n
            V[i, b] = zero(T)
        end
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

_batch_lmo_and_gap!(::KernelAbstractions.Backend, ::Knapsack, c::BatchCache{T}, X) where T =
    throw(ArgumentError(
        "Knapsack oracle does not support device arrays in batch_solve. " *
        "Use ScalarBox, Box, or Simplex for the device path."))

# ------------------------------------------------------------------
# Generic AbstractOracle fallback (CPU only)
# ------------------------------------------------------------------

function _batch_lmo_and_gap!(::KernelAbstractions.CPU, lmo::AbstractOracle, c::BatchCache{T}, X) where T
    n, B = size(X)
    v_buf = Vector{T}(undef, n)
    g_buf = Vector{T}(undef, n)
    @inbounds for b in 1:B
        c.active[b] || continue
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

_batch_lmo_and_gap!(::KernelAbstractions.Backend, lmo::AbstractOracle, c::BatchCache{T}, X) where T =
    throw(ArgumentError(
        "Generic oracle fallback does not support GPU arrays in batch_solve. " *
        "Provide a specialized _batch_lmo_and_gap! method for $(typeof(lmo)), " *
        "or use a built-in oracle (ScalarBox, Box, Simplex)."))
