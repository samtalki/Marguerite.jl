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
    if group in ("fast", "all", "gpu")
        group
    else
        error("Unsupported MARGUERITE_TEST_GROUP=$(repr(group)); expected \"fast\", \"all\", or \"gpu\".")
    end
end

# AppleAccelerate extension smoke test. The extension itself is just a
# `using AppleAccelerate` shim that triggers BLAS forwarding through Apple's
# Accelerate framework. We can't load AppleAccelerate as a hard test
# dependency (it only has Apple Silicon binaries), so this stays opt-in.
@static if Sys.isapple()
    try
        @eval using AppleAccelerate
        using LinearAlgebra: BLAS
        @info "[AppleAccelerate] BLAS config: $(BLAS.get_config())"
    catch e
        @info "AppleAccelerate not available; skipping forwarding smoke test" exception=e
    end
end

function include_timed(label, file)
    t0 = time()
    include(file)
    dt = round(time() - t0; digits=1)
    @info "[$label] completed in $(dt)s"
end

core_files = [
    ("Oracle", "test_oracle.jl"),
    ("Active Set", "test_active_set.jl"),
    ("Solver", "test_solver.jl"),
    ("Show", "test_show.jl"),
    ("GPU Compat", "test_gpu_compat.jl"),
    ("Batch Solver", "test_batch_solver.jl"),
]

heavy_files = if TEST_GROUP == "fast"
    [
        ("Differentiation (fast)", "test_differentiation_fast.jl"),
        ("Bilevel (fast)", "test_bilevel_fast.jl"),
        ("Verification (fast)", "test_verification_fast.jl"),
        ("Batch Bilevel (fast)", "test_batch_bilevel_fast.jl"),
        ("Batch Diff (fast)", "test_batch_diff_fast.jl"),
    ]
elseif TEST_GROUP == "all"
    [
        ("Differentiation", "test_differentiation.jl"),
        ("Bilevel", "test_bilevel.jl"),
        ("Verification", "test_verification.jl"),
        ("Batch Bilevel", "test_batch_bilevel.jl"),
        ("Batch Diff", "test_batch_diff.jl"),
    ]
else
    # gpu group runs only core_files plus gpu_files
    Tuple{String,String}[]
end

# gpu group runs the same correctness suite against a real device backend
# (Metal, CUDA, AMDGPU). Skips cleanly if no device package is loadable.
gpu_files = TEST_GROUP == "gpu" ? [
    ("GPU Backend", "test_gpu_backend.jl"),
] : []

@testset "Marguerite.jl [$TEST_GROUP]" begin
    @info "Running test group: $TEST_GROUP"
    for (label, file) in vcat(core_files, heavy_files, gpu_files)
        include_timed(label, file)
    end
end
