# Linear Oracles

All oracles solve the linear minimization problem

```math
v^* = \arg\min_{v \in \mathcal{C}} \langle g, v \rangle
```

in-place via `lmo(v, g)`. Any callable `(v, g) -> v` works as an oracle.

```@docs
LinearOracle
```

## Simplex

```@docs
Simplex
```

**Capped** (default, `Simplex(r)`): ``\mathcal{C} = \{x \in \mathbb{R}^n : x \geq 0,\; \sum_i x_i \leq r\}``

Vertices are ``\{0, r e_1, \ldots, r e_n\}``. The LP
``\min \langle g, v \rangle`` selects ``r e_{i^*}`` when ``r g_{i^*} < 0``
(where ``i^* = \arg\min_i g_i``), otherwise the origin. ``O(n)`` via a single pass.

**Probability** (`ProbSimplex(r)` or `ProbabilitySimplex(r)`): ``\mathcal{C} = \{x \in \mathbb{R}^n : x \geq 0,\; \sum_i x_i = r\}``

Vertices are ``\{r e_1, \ldots, r e_n\}``. The equality constraint
eliminates the origin, so the LP always picks ``r e_{i^*}`` where
``i^* = \arg\min_i g_i``. ``O(n)`` via a single pass.

```@docs
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

**Set**: ``\mathcal{C} = \{x \in [0,1]^m : \sum_i x_i \leq q,\; x_e = 1\;\forall e \in \mathcal{B}\}``

**Derivation**: The LP decomposes: backbone indices are pinned at 1, remaining
budget ``k = q - |\mathcal{B}|`` goes to coordinates where ``g_i`` is most
negative (box ``[0,1]`` makes each at upper bound). ``O(m \log k)`` via `partialsortperm`.

## Box

```@docs
Box
```

**Set**: ``\mathcal{C} = \{x \in \mathbb{R}^n : \ell_i \leq x_i \leq u_i\}``

**Derivation**: Separable LP: each coordinate independently selects the bound
that minimizes ``g_i v_i``. Positive gradient ``\to`` lower bound; negative
``\to`` upper bound. ``O(n)``.

## WeightedSimplex

```@docs
WeightedSimplex
```

**Set**: ``\mathcal{C} = \{x \in \mathbb{R}^m : x \geq \ell,\; \alpha^\top x \leq \beta\}``

**Derivation**: Shift ``u = x - \ell``, adjusted budget ``\bar\beta = \beta - \alpha^\top \ell``.
The shifted feasible set is a weighted simplex ``\{u \geq 0, \alpha^\top u \leq \bar\beta\}``
with vertices ``\{0, (\bar\beta/\alpha_1) e_1, \ldots\}``. The LP picks the vertex with
best gradient-per-unit-weight ratio: ``i^* = \arg\min_i \{g_i / \alpha_i : g_i < 0\}``.
``O(m)``.
