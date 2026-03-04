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
  solver.jl         # solve() -- core Frank-Wolfe loop (6 method signatures)
  diff_rules.jl     # ChainRulesCore rrule for implicit differentiation
  bilevel.jl        # bilevel_solve / bilevel_gradient for bilevel optimization
```

### Key Design Decisions

- **Single entry point**: Everything goes through `solve()`. Six method signatures handle ±gradient, ±parameters, ±ParametricOracle.
- **Oracle interface**: Callable structs `lmo(v, g) <: AbstractOracle`, in-place. Plain functions can be wrapped with `FunctionOracle(fn)`.
- **Zero-allocation inner loop**: `Cache` holds pre-allocated buffers; hot loops use `@inbounds`.
- **DifferentiationInterface + ForwardDiff default**: All AD goes through DI with `DEFAULT_BACKEND = DI.AutoForwardDiff()`. Users can override with any DI backend.
- **Implicit differentiation**: `rrule` on θ-accepting `solve` methods. CG with HVPs for Hessian solve.

### API

```julia
# Manual gradient:
x, result = solve(f, ∇f!, lmo, x0; kwargs...)

# Auto gradient (ForwardDiff default):
x, result = solve(f, lmo, x0; kwargs...)

# With parameters θ (differentiable):
x, result = solve(f, ∇f!, lmo, x0, θ; kwargs...)

# Auto gradient + parameters:
x, result = solve(f, lmo, x0, θ; kwargs...)

# With ParametricOracle (differentiable constraint set):
x, result = solve(f, ∇f!, plmo, x0, θ; kwargs...)

# Auto gradient + ParametricOracle:
x, result = solve(f, plmo, x0, θ; kwargs...)
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
