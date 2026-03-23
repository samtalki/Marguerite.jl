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

### Solver

```@docs
Marguerite._solve_core
Marguerite._to_oracle
Marguerite._partial_sort_negative!
Marguerite._lmo_and_gap!
Marguerite._ensure_vertex!
Marguerite._trial_update!
```

### Differentiation

#### Pullback State

```@docs
Marguerite._PullbackState
Marguerite._build_pullback_state
Marguerite._kkt_adjoint_solve_cached
Marguerite._build_cross_matrix!
Marguerite._has_active_set
```

#### CG and Hessian Solves

```@docs
Marguerite._cg_solve
Marguerite._hessian_cg_solve
Marguerite._kkt_adjoint_solve
Marguerite._factor_reduced_hessian
```

#### Null-Space Projection

```@docs
Marguerite._orthogonalize!
Marguerite._null_project!
Marguerite._recover_μ_eq
Marguerite._recover_face_multipliers
Marguerite._primal_face_multipliers
Marguerite._correct_bound_multipliers!
Marguerite._objective_gradient
```

#### Cross-Derivatives

```@docs
Marguerite._make_∇ₓf_of_θ
Marguerite._cross_derivative_manual
Marguerite._cross_derivative_hvp
```

#### Constraint Pullback

```@docs
Marguerite._constraint_scalar
Marguerite._constraint_pullback
```

### Tangent Maps

```@docs
Marguerite._TangentMap
Marguerite._PolyhedralTangentMap
Marguerite._SpectralTangentMap
Marguerite._project_tangent!
Marguerite._expand_tangent!
Marguerite._tangent_correction!
Marguerite._reduced_dim
Marguerite._build_tangent_map
```

### Spectraplex

#### Oracle

```@docs
Marguerite.SpectraplexEqNormals
Marguerite._spectraplex_min_eigen
Marguerite._spectraplex_write_rank1!
```

#### Tangent-Space Coordinates

```@docs
Marguerite._spectraplex_trace_zero_dim
Marguerite._spectraplex_tangent_dim
Marguerite._spectraplex_pack_trace_zero!
Marguerite._spectraplex_unpack_trace_zero!
Marguerite._spectraplex_pack_mixed!
Marguerite._spectraplex_unpack_mixed!
Marguerite._spectraplex_compress!
Marguerite._spectraplex_expand!
Marguerite._spectraplex_add_mixed_curvature!
```

## Index

```@index
```
