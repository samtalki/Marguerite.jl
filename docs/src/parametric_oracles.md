# Parametric Oracles

Parametric oracles represent constraint sets ``C(\theta)`` that depend on
parameters. They are used with the ``\theta``-accepting `solve` variants
to enable differentiation through both the objective and constraint set.

Use [`materialize`](@ref) to instantiate a concrete oracle at a given ``\theta``.

## Example: Learning Box Bounds

Use `ParametricBox` to optimize over a box constraint set with learnable bounds:

```julia
using Marguerite, LinearAlgebra

f(x, θ) = 0.5 * dot(x, x) - dot(θ[1:2], x)
∇f!(g, x, θ) = (g .= x .- θ[1:2])

# Box [lb(θ), ub(θ)] with θ-dependent bounds
plmo = ParametricBox(θ -> fill(θ[3], 2), θ -> fill(θ[4], 2))
θ = [0.8, 0.2, 0.0, 1.0]

x, result = solve(f, plmo, [0.5, 0.5], θ; grad=∇f!)
```

The `rrule` for this signature computes ``d\theta`` through both the
objective and constraint parameters via KKT adjoint differentiation.

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
