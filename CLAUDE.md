# CLAUDE.md

## Project Overview

**Marguerite.jl** is a minimal, performant, and differentiable Frank-Wolfe (conditional gradient) solver for constrained convex optimization. Named after Marguerite Frank (1927--), co-inventor of the Frank-Wolfe algorithm (1956).

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
  types.jl          # Result, Cache, MonotonicStepSize, AdaptiveStepSize
  lmo.jl            # LinearOracle abstract type + 5 concrete oracles
  solver.jl         # solve() -- core Frank-Wolfe loop (4 method signatures)
  diff_rules.jl     # ChainRulesCore rrule for implicit differentiation
```

### Key Design Decisions

- **Single entry point**: Everything goes through `solve()`. Four method signatures handle ±gradient, ±parameters.
- **Oracle interface**: Callable structs `lmo(v, g)`, in-place. Any function `(v, g) -> v` works without subtyping.
- **Zero-allocation inner loop**: `Cache` holds pre-allocated buffers; hot loops use `@inbounds`.
- **DifferentiationInterface + Mooncake default**: Mooncake is a runtime dependency. All AD goes through DI with `DEFAULT_BACKEND = DI.AutoMooncake(; config=nothing)`. Users can override with any DI backend.
- **Implicit differentiation**: `rrule` on θ-accepting `solve` methods. CG with HVPs for Hessian solve.

### API

```julia
# Manual gradient:
x, result = solve(f, ∇f!, lmo, x0; kwargs...)

# Auto gradient (Mooncake default):
x, result = solve(f, lmo, x0; kwargs...)

# With parameters θ (differentiable):
x, result = solve(f, ∇f!, lmo, x0, θ; kwargs...)

# Auto gradient + parameters:
x, result = solve(f, lmo, x0, θ; kwargs...)
```

## Coding Conventions

- Four-space indentation
- `snake_case` functions, `CamelCase` types
- Unicode math symbols in internal code (∇, γ, α, β, ε, θ)
- Docstrings with LaTeX math for public API
- `@inbounds` in hot loops
- Concise, lowercase commit messages

## Dependencies

**Runtime**: LinearAlgebra (stdlib), DifferentiationInterface, ADTypes, ChainRulesCore, Mooncake
**Test-only**: Test, ForwardDiff, Random
**Docs-only**: Documenter
