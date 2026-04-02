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
# Array style trait for CPU/GPU dispatch
# ------------------------------------------------------------------

"""
    _ArrayStyle

Abstract supertype for array backend dispatch. GPU package extensions
override `_array_style` for their array types.
"""
abstract type _ArrayStyle end
struct _CPUStyle <: _ArrayStyle end
struct _GPUStyle <: _ArrayStyle end

"""
    _array_style(x) -> _CPUStyle() or _GPUStyle()

Trait function for CPU/GPU dispatch. Default is CPU. GPU package
extensions (Metal.jl, CUDA.jl) should define methods returning
`_GPUStyle()` for their array types.
"""
_array_style(::AbstractArray) = _CPUStyle()

"""
    Result{T<:Real}

Immutable record of a Frank-Wolfe solve.

# Fields
- `objective::T` -- final objective value ``f(x^*)``
- `gap::T` -- final Frank-Wolfe duality gap
- `iterations::Int` -- iterations taken
- `converged::Bool` -- whether ``\\mathrm{gap} \\le \\mathrm{tol} \\cdot (1 + |f(x)|)``
- `discards::Int` -- rejected non-improving updates (monotonic mode)
"""
struct Result{T<:Real}
    objective::T
    gap::T
    iterations::Int
    converged::Bool
    discards::Int
end

"""
    CGResult{T<:Real}

Diagnostics from the linear solve in implicit differentiation.

The `rrule` pullback uses a cached direct factorization (Cholesky/LU) and returns
nominal values `(0, 0.0, true)`. The CG fields are meaningful only for
`bilevel_solve`, which uses iterative CG.

# Fields
- `iterations::Int` -- CG iterations taken (0 for direct-solve path)
- `residual_norm::T` -- final residual ``\\|r\\|`` (0 for direct-solve path)
- `converged::Bool` -- whether solve succeeded
"""
struct CGResult{T<:Real}
    iterations::Int
    residual_norm::T
    converged::Bool
end

"""
    Cache{T<:Real}

Pre-allocated working buffers for the Frank-Wolfe inner loop.
Includes sparse vertex buffers used internally by fused LMO+gap computation.

Construct via `Cache{T}(n)` or let `solve` allocate one automatically.
"""
struct Cache{T<:Real, V<:AbstractVector{T}}
    gradient::V
    vertex::V
    x_trial::V
    direction::V
    vertex_nzind::Vector{Int}   # always CPU — scalar indexing in sparse vertex protocol
    vertex_nzval::Vector{T}     # always CPU — scalar indexing in sparse vertex protocol

    function Cache{T,V}(gradient::V, vertex::V,
                        x_trial::V, direction::V,
                        vertex_nzind::Vector{Int}, vertex_nzval::Vector{T}) where {T<:Real, V<:AbstractVector{T}}
        n = length(gradient)
        (length(vertex) == n && length(x_trial) == n && length(direction) == n) ||
            throw(DimensionMismatch(
                "Cache buffers must all have length $n (got $(length(gradient)), $(length(vertex)), $(length(x_trial)), $(length(direction)))"))
        (length(vertex_nzind) == n && length(vertex_nzval) == n) ||
            throw(DimensionMismatch(
                "Cache sparse buffers must have length $n (got vertex_nzind=$(length(vertex_nzind)), vertex_nzval=$(length(vertex_nzval)))"))
        new{T,V}(gradient, vertex, x_trial, direction, vertex_nzind, vertex_nzval)
    end
end

function Cache{T}(n::Int) where {T<:Real}
    n > 0 || throw(ArgumentError("Cache dimension must be positive, got n=$n"))
    vecs = ntuple(_ -> zeros(T, n), Val(4))
    Cache{T, Vector{T}}(vecs..., zeros(Int, n), zeros(T, n))
end

"""
    Cache(x0::AbstractVector{T})

Construct a `Cache` whose main buffers match the array type of `x0`.
Sparse vertex buffers are always CPU `Vector`.
"""
function Cache(x0::AbstractVector{T}) where {T<:Real}
    n = length(x0)
    n > 0 || throw(ArgumentError("Cache dimension must be positive, got n=$n"))
    _zl(x) = fill!(similar(x), zero(T))
    V = typeof(_zl(x0))
    Cache{T, V}(_zl(x0), _zl(x0), _zl(x0), _zl(x0), zeros(Int, n), zeros(T, n))
end

"""
    Cache(n)

Convenience constructor for `Cache{Float64}(n)`.
"""
Cache(n::Int) = Cache{Float64}(n)

"""
    MonotonicStepSize()

The standard Frank-Wolfe step size

```math
\\gamma_t = \\frac{2}{t+2}
```

yielding ``O(1/t)`` convergence. Carderera, Besançon & Pokutta (2024) establish
this rate for generalized self-concordant objectives.
"""
struct MonotonicStepSize end

(::MonotonicStepSize)(t::Int) = 2.0 / (t + 2)

"""
    AdaptiveStepSize(L0::Real=1.0; eta=2.0)

Backtracking line-search step size with Lipschitz estimation.

Starting from estimate ``L``, multiplies ``L`` by ``\\eta`` until sufficient decrease holds,
then sets

```math
\\gamma = \\mathrm{clamp}\\!\\left(\\frac{\\langle \\nabla f,\\, x - v \\rangle}{L \\,\\|d\\|^2},\\; 0,\\; 1\\right)
```
"""
mutable struct AdaptiveStepSize{T<:Real}
    L::T
    η::T
end

function AdaptiveStepSize(L0::Real=1.0; eta::Real=2.0)
    L0f, etaf = promote(float(L0), float(eta))
    return AdaptiveStepSize(L0f, etaf)
end

# ------------------------------------------------------------------
# Wrapper types for solve / bilevel_solve output
# ------------------------------------------------------------------

"""
    SolveResult{T, V<:AbstractVector{T}}

Wrapper for `solve` output. Supports tuple unpacking and pretty-printing.

`x, result = solve(...)` still works via `Base.iterate`.
Provides cleaner REPL display than a raw tuple.

# Fields
- `x::AbstractVector{T}` -- optimal solution ``x^*``
- `result::Result{T}` -- convergence diagnostics
"""
struct SolveResult{T<:Real, V<:AbstractVector{T}}
    x::V
    result::Result{T}
end

Base.iterate(sr::SolveResult) = (sr.x, Val(:result))
Base.iterate(sr::SolveResult, ::Val{:result}) = (sr.result, nothing)
Base.iterate(::SolveResult, ::Nothing) = nothing
Base.length(::SolveResult) = 2
Base.IteratorSize(::Type{<:SolveResult}) = Base.HasLength()

"""
    BilevelResult{T, S}

Wrapper for `bilevel_solve` output. Supports tuple unpacking and pretty-printing.

`x, θ_grad, cg_result = bilevel_solve(...)` still works via `Base.iterate`.

# Fields
- `x::Vector{T}` -- inner problem solution ``x^*(\\theta)``
- `theta_grad::S` -- gradient ``\\nabla_\\theta L(x^*(\\theta))``
- `cg_result::CGResult{T}` -- CG solver diagnostics
"""
struct BilevelResult{T<:Real, S}
    x::Vector{T}
    theta_grad::S
    cg_result::CGResult{T}
end

Base.iterate(br::BilevelResult) = (br.x, Val(:tg))
Base.iterate(br::BilevelResult, ::Val{:tg}) = (br.theta_grad, Val(:cg))
Base.iterate(br::BilevelResult, ::Val{:cg}) = (br.cg_result, nothing)
Base.iterate(::BilevelResult, ::Nothing) = nothing
Base.length(::BilevelResult) = 3
Base.IteratorSize(::Type{<:BilevelResult}) = Base.HasLength()
