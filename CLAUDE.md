# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Marguerite.jl** is a minimal, performant, and differentiable Frank-Wolfe (conditional gradient) solver for constrained convex optimization. Named in honor of Marguerite Frank (1927–2024), co-inventor of the Frank-Wolfe algorithm (1956). Requires Julia 1.12+.

## Build and Development Commands

```bash
# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run the default representative test suite
julia --project=. -e 'using Pkg; Pkg.test()'

# Run the exhaustive suite
MARGUERITE_TEST_GROUP=all julia --project=. -e 'using Pkg; Pkg.test()'

# Run a single test file
julia --project=. test/test_solver.jl

# Interactive development
julia --project=.
julia> using Revise, Marguerite

# Build docs
julia --project=docs docs/make.jl
```

Test files: `test_lmo.jl`, `test_active_set.jl`, `test_solver.jl`, fast representative files for differentiation/bilevel/verification, and exhaustive `test_differentiation.jl`, `test_bilevel.jl`, `test_verification.jl`. Orchestrated by `test/runtests.jl` via `MARGUERITE_TEST_GROUP=fast|all`.

## Architecture

### Module Organization

```
src/
  Marguerite.jl     # Module file: includes, exports, precompile workload
  types.jl          # Result, Cache, ActiveConstraints, ParametricOracle types, step sizes
  lmo.jl            # AbstractOracle abstract type + concrete oracles + active_set + materialize
  solver.jl         # solve() -- core Frank-Wolfe loop (2 public methods + _solve_core)
  diff_rules.jl     # ChainRulesCore rrule, CG solver, KKT adjoint, cross-derivatives
  bilevel.jl        # bilevel_solve / bilevel_gradient for bilevel optimization
  show.jl           # Pretty printing for all public types
```

### Key Design Decisions

- **Single entry point**: Everything goes through `solve()`. Two public methods handle ±parameters; `grad=` keyword controls manual vs auto gradient; `ParametricOracle` is handled via `isa` checks inside `solve`.
- **Oracle interface**: Any callable `(v, g) -> v` works as an oracle — plain functions are auto-wrapped as `FunctionOracle` by `solve`. Subtype `AbstractOracle` for specialized dispatch (e.g. `active_set`, sparse vertex protocol). Differentiation (`rrule`/`bilevel_solve`) requires an `AbstractOracle` with `active_set` support, or `assume_interior=true`.
- **Zero-allocation inner loop**: `Cache` holds 6 pre-allocated buffers (gradient, vertex, x_trial, direction, vertex_nzind, vertex_nzval); hot loops use `@inbounds @simd`.
- **DifferentiationInterface + ForwardDiff default**: All AD goes through DI with `DEFAULT_BACKEND = DI.AutoForwardDiff()`. Users can override with any DI backend.
- **Implicit differentiation**: `rrule` on the θ-accepting `solve` method. Cached Hessian factorization for the pullback (direct solve, not CG), KKT adjoint system on the active face. `bilevel_solve` uses iterative CG.

### Solve Dispatch Chain

`solve(f, lmo, x0; ...)` and `solve(f, lmo, x0, θ; ...)` are the two public methods. Both:
1. Normalize `lmo` to an `AbstractOracle` (auto-wrap functions, materialize `ParametricOracle`)
2. Handle `grad=nothing` (auto-gradient via DI) vs manual gradient
3. Delegate to `_solve_core(f, ∇f!, lmo, x0; ...)` which runs the FW loop

The θ-variant closes `f(x, θ)` → `fθ(x)` and `grad(g, x, θ)` → `∇fθ!(g, x)`, then calls the non-θ `solve`.

### Sparse Vertex Protocol

`_lmo_and_gap!(lmo, cache, x, n) -> (fw_gap, nnz)` is the fused LMO+gap computation. The `nnz` return encodes the vertex representation:
- `nnz = -1`: dense vertex in `cache.vertex` (Box, WeightedSimplex, FunctionOracle, Spectraplex)
- `nnz = 0`: origin vertex (capped Simplex when all gradients ≥ 0)
- `nnz > 0`: sparse vertex in `cache.vertex_nzind[1:nnz]`, `cache.vertex_nzval[1:nnz]` (Simplex, Knapsack, MaskedKnapsack)

`_trial_update!` and `_ensure_vertex!` dispatch on `nnz` to avoid materializing dense vertex vectors for sparse oracles.

### Differentiation Pipeline (diff_rules.jl)

The `rrule` for `solve(f, lmo, x0, θ; ...)`:
1. Solves the forward problem
2. Builds `_PullbackState` (or `_SpectraplexPullbackState`): active set identification, HVP preparation, constraint orthogonalization — all cached for reuse across pullback calls
3. Returns a `solve_pullback(dy)` closure that:
   - Runs `_kkt_adjoint_solve_cached` (reduced Hessian CG on the null space of active constraints)
   - Computes cross-derivative dθ via manual gradient (`_cross_derivative_manual`) or joint HVP (`_cross_derivative_hvp`)
   - For `ParametricOracle`: adds constraint sensitivity via `_constraint_pullback` → `_constraint_scalar`

`bilevel_solve` follows the same pipeline but without caching (single-use).

### Result Wrapper Types

`SolveResult` and `BilevelResult` wrap solver output with `Base.iterate` for tuple unpacking (`x, result = solve(...)` and `x, dθ, cg = bilevel_solve(...)`) plus pretty REPL display.

### API

```julia
# Auto gradient:
x, result = solve(f, lmo, x0; kwargs...)

# Manual gradient:
x, result = solve(f, lmo, x0; grad=∇f!, kwargs...)

# With parameters θ (differentiable):
x, result = solve(f, lmo, x0, θ; kwargs...)
x, result = solve(f, lmo, x0, θ; grad=∇f!, kwargs...)

# With ParametricOracle (differentiable constraint set):
x, result = solve(f, plmo, x0, θ; kwargs...)

# Bilevel:
x, dθ, cg_result = bilevel_solve(outer, inner, lmo, x0, θ; kwargs...)
dθ = bilevel_gradient(outer, inner, lmo, x0, θ; kwargs...)
```

## Coding Conventions

- Four-space indentation
- `snake_case` functions, `CamelCase` types
- Unicode math symbols in internal code (∇, γ, α, β, ε, θ)
- Docstrings with LaTeX math for public API
- `@inbounds` in hot loops, `@simd` where applicable
- Concise, lowercase commit messages
- Apache 2.0 license header on all source files

## Dependencies

**Runtime**: LinearAlgebra (stdlib), Printf (stdlib), DifferentiationInterface, ADTypes, ChainRulesCore, ForwardDiff, PrecompileTools
**Test-only**: Test, Random, LinearAlgebra, BenchmarkTools, JuMP, Clarabel
**Docs-only**: Documenter
