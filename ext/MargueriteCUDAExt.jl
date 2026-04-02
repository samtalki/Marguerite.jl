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
    MargueriteCUDAExt

Package extension for CUDA.jl support. Registers CUDA GPU arrays
with the Marguerite array style trait so that GPU-friendly oracle paths
are dispatched automatically.

When this extension is loaded, `solve(f, lmo, CuVector(x0); grad=...)` and
`batch_solve(f, lmo, CuMatrix(X0); grad_batch=...)` use broadcast-based
oracle and update paths that avoid scalar indexing on GPU.
"""
module MargueriteCUDAExt

using Marguerite: Marguerite, _GPUStyle
using CUDA: CuArray

# Register all CUDA arrays as GPU arrays
Marguerite._array_style(::CuArray) = _GPUStyle()

end # module
