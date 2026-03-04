<!-- Copyright 2026 Samuel Talkington and contributors
   SPDX-License-Identifier: Apache-2.0 -->


# Oracles

All oracles solve the linear minimization problem

```math
v^* = \arg\min_{v \in \mathcal{C}} \langle g, v \rangle
```

in-place via `lmo(v, g)`. Any callable `(v, g) -> v` works as an oracle.
See the [Tutorial](@ref) for an example of writing a custom oracle.

```@docs
AbstractOracle
```

## Simplex

![Capped simplex constraint set](assets/simplex_capped.svg)

![Probability simplex constraint set](assets/simplex_prob.svg)

```@docs
Simplex
ProbSimplex
ProbabilitySimplex
```

## Knapsack

![Knapsack constraint set](assets/knapsack.svg)

```@docs
Knapsack
```

## MaskedKnapsack

![Masked knapsack constraint set](assets/knapsack_masked.svg)

```@docs
MaskedKnapsack
```

## Box

![Box constraint set](assets/box.svg)

```@docs
Box
```

## WeightedSimplex

![Weighted simplex constraint set](assets/weighted_simplex.svg)

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
