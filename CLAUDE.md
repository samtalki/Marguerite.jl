# CLAUDE.md

## Project Overview

**Marguerite.jl** is a minimal, performant, and differentiable Frank-Wolfe (conditional gradient) solver for constrained convex optimization. Named in honor of Marguerite Frank (1927–2024), co-inventor of the Frank-Wolfe algorithm (1956).

## Build and Development Commands

```bash
# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Interactive development
julia --project=.
julia> using Revise, Marguerite

# Build docs
julia --project=docs docs/make.jl
```

## Architecture

### Module Organization

```
src/
  Marguerite.jl     # Module file: includes, exports
  types.jl          # Result, Cache, ActiveConstraints, ParametricOracle types, step sizes
  lmo.jl            # AbstractOracle abstract type + 5 concrete oracles + active_set
  solver.jl         # solve() -- core Frank-Wolfe loop (2 public methods + _solve_core)
  diff_rules.jl     # ChainRulesCore rrule for implicit differentiation
  bilevel.jl        # bilevel_solve / bilevel_gradient for bilevel optimization
```

### Key Design Decisions

- **Single entry point**: Everything goes through `solve()`. Two public methods handle ±parameters; `grad=` keyword controls manual vs auto gradient; `ParametricOracle` is handled via `isa` checks.
- **Oracle interface**: Any callable `(v, g) -> v` works as an oracle — plain functions are auto-wrapped by `solve`. Subtype `AbstractOracle` for specialized dispatch (e.g. `active_set`, sparse vertex protocol).
- **Zero-allocation inner loop**: `Cache` holds pre-allocated buffers; hot loops use `@inbounds`.
- **DifferentiationInterface + ForwardDiff default**: All AD goes through DI with `DEFAULT_BACKEND = DI.AutoForwardDiff()`. Users can override with any DI backend.
- **Implicit differentiation**: `rrule` on θ-accepting `solve` methods. CG with HVPs for Hessian solve.

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
x, result = solve(f, plmo, x0, θ; grad=∇f!, kwargs...)
```

## Coding Conventions

- Four-space indentation
- `snake_case` functions, `CamelCase` types
- Unicode math symbols in internal code (∇, γ, α, β, ε, θ)
- Docstrings with LaTeX math for public API
- `@inbounds` in hot loops
- Concise, lowercase commit messages

## Dependencies

**Runtime**: LinearAlgebra (stdlib), DifferentiationInterface, ADTypes, ChainRulesCore, ForwardDiff, PrecompileTools
**Test-only**: Test, Random, JuMP, Clarabel
**Docs-only**: Documenter
