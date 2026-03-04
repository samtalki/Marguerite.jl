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
```

## Step Size Rules

```@docs
MonotonicStepSize
AdaptiveStepSize
```

## Internals

These are not part of the public API and may change without notice.

```@docs
Marguerite.Cache
Marguerite._cg_solve
Marguerite._hessian_cg_solve
Marguerite._kkt_adjoint_solve
Marguerite._kkt_implicit_pullback
Marguerite._kkt_implicit_pullback_hvp
Marguerite._constraint_pullback
```

## Index

```@index
```
