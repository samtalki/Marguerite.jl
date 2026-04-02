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
    MargueriteAMDGPUExt

Package extension for AMDGPU.jl support. Registers ROCm GPU arrays
with the Marguerite array style trait so that GPU-friendly oracle paths
are dispatched automatically.

When this extension is loaded, `solve(f, lmo, ROCVector(x0); grad=...)` and
`batch_solve(f, lmo, ROCMatrix(X0); grad_batch=...)` use broadcast-based
oracle and update paths that avoid scalar indexing on GPU.
"""
module MargueriteAMDGPUExt

using Marguerite: Marguerite, _GPUStyle
using AMDGPU: ROCArray

# Register all ROCm arrays as GPU arrays
Marguerite._array_style(::ROCArray) = _GPUStyle()

end # module
