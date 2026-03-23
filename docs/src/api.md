# API Reference

```@meta
CurrentModule = Marguerite
```

## Module

```@docs
Marguerite
```

## Solver

```@docs
solve
```

## Bilevel

```@docs
bilevel_solve
bilevel_gradient
```

## Jacobian

```@docs
solution_jacobian
solution_jacobian!
```

## Types

```@docs
Result
CGResult
SolveResult
BilevelResult
Cache
```

## Step Size Rules

```@docs
MonotonicStepSize
AdaptiveStepSize
```

## AD Backends

```@docs
Marguerite.DEFAULT_BACKEND
Marguerite.SECOND_ORDER_BACKEND
```

## Oracle Types

See [Oracles](@ref) for full documentation and constraint set diagrams.

## Parametric Oracle Types

See [Parametric Oracles](@ref) for full documentation.

## Internals

These are not part of the public API and may change without notice.

```@docs
Marguerite._cg_solve
Marguerite._hessian_cg_solve
Marguerite._kkt_adjoint_solve
Marguerite._constraint_pullback
Marguerite._null_project!
Marguerite._constraint_scalar
Marguerite._correct_bound_multipliers!
Marguerite._make_∇ₓf_of_θ
Marguerite._cross_derivative_manual
Marguerite._cross_derivative_hvp
Marguerite._TangentMap
Marguerite._PolyhedralTangentMap
Marguerite._SpectralTangentMap
Marguerite._project_tangent!
Marguerite._expand_tangent!
Marguerite._tangent_correction!
Marguerite._reduced_dim
Marguerite._build_tangent_map
Marguerite._solve_core
Marguerite._to_oracle
Marguerite._partial_sort_negative!
Marguerite._lmo_and_gap!
Marguerite._ensure_vertex!
Marguerite._trial_update!
```

## Index

```@index
```
