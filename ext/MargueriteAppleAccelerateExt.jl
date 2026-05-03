# Copyright 2026 Samuel Talkington
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    MargueriteAppleAccelerateExt

Loading `AppleAccelerate` redirects LinearAlgebra (BLAS and LAPACK) through
Apple's Accelerate framework via libblastrampoline, including the AMX matrix
coprocessor on Apple Silicon. For Marguerite this primarily accelerates the
Spectraplex eigendecomposition (`eigen!(Symmetric(...))`) and the dense
`mul!` paths in the batch solver.

Importing AppleAccelerate is sufficient — no Marguerite-side configuration.
On Apple Silicon, prefer it for the Spectraplex and batched matmul paths.

To verify forwarding after loading:

```julia
using LinearAlgebra, AppleAccelerate
LinearAlgebra.BLAS.get_config()
```

The `Accelerate` provider should appear for both LP64 and ILP64 lines.

# Name collision

`AppleAccelerate.solve` (a sparse factorization solve) shadows
`Marguerite.solve` when both are imported with `using`. Either qualify
explicitly (`Marguerite.solve(...)`) or import selectively
(`using Marguerite: solve`). Same applies to a few other names — qualify
or import selectively.
"""
module MargueriteAppleAccelerateExt

# Loading AppleAccelerate (via the Project.toml weakdep) triggers its
# __init__ which calls BLAS.lbt_forward(...). No Marguerite hooks needed.
using AppleAccelerate: AppleAccelerate

end # module
