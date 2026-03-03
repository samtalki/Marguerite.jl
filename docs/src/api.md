<!-- Copyright 2026 Samuel Talkington and contributors
   SPDX-License-Identifier: Apache-2.0 -->

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
Marguerite._implicit_pullback
Marguerite._implicit_pullback_hvp
```

## Index

```@index
```
