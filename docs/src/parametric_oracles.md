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
