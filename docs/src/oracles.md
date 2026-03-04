# Oracles

All oracles solve the linear minimization problem

```math
v^* = \arg\min_{v \in \mathcal{C}} \langle g, v \rangle
```

in-place via `lmo(v, g)`. Any callable `(v, g) -> v` works as an oracle.

```@docs
AbstractOracle
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
