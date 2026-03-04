<!-- Copyright 2026 Samuel Talkington and contributors
   SPDX-License-Identifier: Apache-2.0 -->

# Linear Oracles

All oracles solve the linear minimization problem

```math
v^* = \arg\min_{v \in \mathcal{C}} \langle g, v \rangle
```

in-place via `lmo(v, g)`. Any callable `(v, g) -> v` works as an oracle.

```@docs
AbstractOracle
LinearOracle
```

## Simplex

```@docs
Simplex
ProbSimplex
ProbabilitySimplex
```

## Knapsack

```@docs
Knapsack
```

## MaskedKnapsack

```@docs
MaskedKnapsack
```

## Box

```@docs
Box
```

## WeightedSimplex

```@docs
WeightedSimplex
```

## Active Set Identification

At a solution ``x^*``, Marguerite identifies which constraints are active
(binding) to support KKT adjoint differentiation. Each oracle type has a
specialized [`active_set`](@ref) method.

```@docs
ActiveConstraints
active_set
```

# Parametric Oracles

Parametric oracles represent constraint sets ``C(\theta)`` that depend on
parameters. They are used with the ``\theta``-accepting `solve` variants
to enable differentiation through both the objective and constraint set.

Use [`materialize`](@ref) to instantiate a concrete oracle at a given ``\theta``.

```@docs
ParametricOracle
materialize
```

## ParametricBox

```@docs
ParametricBox
```

## ParametricSimplex

```@docs
ParametricSimplex
ParametricProbSimplex
```

## ParametricWeightedSimplex

```@docs
ParametricWeightedSimplex
```
