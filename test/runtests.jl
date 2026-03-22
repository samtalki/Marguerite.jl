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

using Marguerite
using Test

const TEST_GROUP = let group = lowercase(get(ENV, "MARGUERITE_TEST_GROUP", "fast"))
    if group in ("fast", "all")
        group
    else
        error("Unsupported MARGUERITE_TEST_GROUP=$(repr(group)); expected \"fast\" or \"all\".")
    end
end

function include_timed(label, file)
    t0 = time()
    include(file)
    dt = round(time() - t0; digits=1)
    @info "[$label] completed in $(dt)s"
end

core_files = [
    ("LMO", "test_lmo.jl"),
    ("Active Set", "test_active_set.jl"),
    ("Solver", "test_solver.jl"),
]

heavy_files = TEST_GROUP == "fast" ? [
    ("Differentiation (fast)", "test_differentiation_fast.jl"),
    ("Bilevel (fast)", "test_bilevel_fast.jl"),
    ("Verification (fast)", "test_verification_fast.jl"),
] : [
    ("Differentiation", "test_differentiation.jl"),
    ("Bilevel", "test_bilevel.jl"),
    ("Verification", "test_verification.jl"),
]

@testset "Marguerite.jl [$TEST_GROUP]" begin
    @info "Running test group: $TEST_GROUP"
    for (label, file) in vcat(core_files, heavy_files)
        include_timed(label, file)
    end
end
