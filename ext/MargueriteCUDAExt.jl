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

Registers `CuArray` with Marguerite's array-style trait so the GPU
broadcast paths dispatch automatically when CUDA.jl is loaded.
"""
module MargueriteCUDAExt

using Marguerite: Marguerite, _GPUStyle
using CUDA: CuArray

Marguerite._array_style(::CuArray) = _GPUStyle()

end # module
