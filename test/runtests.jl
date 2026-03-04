# Copyright 2026 Samuel Talkington and contributors
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

using Marguerite
using Test

@testset "Marguerite.jl" begin
    for (label, file) in [
        ("LMO", "test_lmo.jl"),
        ("Active Set", "test_active_set.jl"),
        ("Solver", "test_solver.jl"),
        ("Differentiation", "test_differentiation.jl"),
        ("Bilevel", "test_bilevel.jl"),
        ("Verification", "test_verification.jl"),
    ]
        t0 = time()
        include(file)
        dt = round(time() - t0; digits=1)
        @info "[$label] completed in $(dt)s"
    end
end
