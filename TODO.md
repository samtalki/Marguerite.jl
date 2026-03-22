# TODO

Future work and feature ideas for Marguerite.jl.

## Solver features

- **Warm starting**: Accept a previous solution/cache to resume optimization, useful for parameter sweeps and online settings.
- **Callback / logging**: User-provided callback at each iteration for custom logging, early stopping, or solution tracking.

## Oracles

- **Product oracle**: Oracle for Cartesian products `C = C_1 x C_2 x ...` where each
  block is solved independently. Design: `ProductOracle((lmo_1, lmo_2, ...), [1:n_1, n_1+1:n_1+n_2, ...])`
  with composed `active_set` (shift indices per block, zero-pad equality normals).
  Does NOT handle intersection `C_1 ∩ C_2` — intersection LMOs require LP solves
  or alternating projection, which is a separate feature.
## Performance

- **ARPACK / iterative eigensolver for Spectraplex**: Replace dense `eigen` with iterative minimum eigenpair computation for large-scale SDPs.
