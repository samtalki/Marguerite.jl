# API Reference

```@meta
CurrentModule = Marguerite
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

## Types

```@docs
Result
CGResult
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
Marguerite._kkt_implicit_pullback
Marguerite._kkt_implicit_pullback_hvp
Marguerite._constraint_pullback
Marguerite._null_project!
Marguerite._constraint_scalar
Marguerite._correct_bound_multipliers!
Marguerite._make_∇_x_f_of_θ
Marguerite._cross_derivative_manual
Marguerite._cross_derivative_hvp
Marguerite._partial_sort_negative!
ChainRulesCore.rrule(::typeof(solve), ::Any, ::Any, ::ParametricOracle, ::Any, ::Any)
Marguerite._lmo_and_gap!
Marguerite._ensure_vertex!
Marguerite._trial_update!
```

## Index

```@index
```
