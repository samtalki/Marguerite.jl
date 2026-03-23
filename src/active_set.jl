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
# Active set identification
# ------------------------------------------------------------------

"""
    ActiveConstraints{T}

Active constraint identification at a solution ``x^*``.

# Fields
- `bound_indices::Vector{Int}` -- indices pinned to bounds
- `bound_values::Vector{T}` -- their bound values
- `bound_is_lower::BitVector` -- `true` if bound is a lower bound, `false` if upper
- `free_indices::Vector{Int}` -- unconstrained variable indices
- `eq_normals` -- vector-like collection of equality constraint normals (in full space)
- `eq_rhs` -- equality constraint RHS values
"""
struct ActiveConstraints{T<:Real, EN, ER<:AbstractVector{T}}
    bound_indices::Vector{Int}
    bound_values::Vector{T}
    bound_is_lower::BitVector
    free_indices::Vector{Int}
    eq_normals::EN
    eq_rhs::ER

    function ActiveConstraints{T}(bound_indices, bound_values, bound_is_lower,
                          free_indices, eq_normals, eq_rhs) where T
        length(bound_indices) == length(bound_values) == length(bound_is_lower) ||
            throw(ArgumentError("ActiveConstraints: bound arrays must have equal length"))
        length(eq_normals) == length(eq_rhs) ||
            throw(ArgumentError("ActiveConstraints: eq_normals and eq_rhs must have equal length"))
        new{T, typeof(eq_normals), typeof(eq_rhs)}(
            bound_indices, bound_values, bound_is_lower, free_indices, eq_normals, eq_rhs)
    end
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
    eq_normals = SpectraplexEqNormals(n, TP(lmo.r), U, V_perp)
    eq_rhs = zeros(TP, length(eq_normals))
    eq_rhs[_spectraplex_sym_count(n) + 1] = TP(lmo.r)

    ActiveConstraints{TP}(Int[], TP[], BitVector(), collect(1:m), eq_normals, eq_rhs)
end
