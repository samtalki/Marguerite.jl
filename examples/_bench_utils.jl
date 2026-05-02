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

# Shared helpers for examples/bench_*.jl.  Loaded via `include("_bench_utils.jl")`.

using LinearAlgebra: BLAS
using Statistics: median
using Dates

# ── GPU backend auto-detect ──────────────────────────────────────────
#
# `using Metal` (etc.) must run at top-level for the module binding to be
# visible without world-age dance. We try each backend's import at load
# time of this file (silent on failure), then `detect_gpu_backend()` reads
# whichever is loaded.

const BENCH_NO_GPU = get(ENV, "BENCH_NO_GPU", "0") == "1"

if !BENCH_NO_GPU
    Sys.isapple() && try
        @eval Main using Metal
    catch
    end
    try
        @eval Main using CUDA
    catch
    end
    try
        @eval Main using AMDGPU
    catch
    end
end

function _backend_from_module(name::String, mod::Module, arr_sym::Symbol, sync_sym::Symbol)
    try
        if mod.functional()
            return (name=name, arr=getfield(mod, arr_sym), sync=getfield(mod, sync_sym))
        end
    catch
    end
    return nothing
end

"""
    detect_gpu_backend() -> NamedTuple or nothing

Detect a functional GPU backend (Metal on Apple Silicon, then CUDA, then
AMDGPU). Returns `(name, arr, sync)` or `nothing`. Honors
`ENV["BENCH_NO_GPU"]=="1"`.
"""
function detect_gpu_backend()
    BENCH_NO_GPU && return nothing
    if Sys.isapple() && isdefined(Main, :Metal)
        b = _backend_from_module("Metal", Main.Metal, :MtlArray, :synchronize)
        b !== nothing && return b
    end
    if isdefined(Main, :CUDA)
        b = _backend_from_module("CUDA", Main.CUDA, :CuArray, :synchronize)
        b !== nothing && return b
    end
    if isdefined(Main, :AMDGPU)
        b = _backend_from_module("AMDGPU", Main.AMDGPU, :ROCArray, :synchronize)
        b !== nothing && return b
    end
    return nothing
end

# ── Hardware fingerprint ─────────────────────────────────────────────

"""
    hardware_fingerprint(; gpu_backend, use_accelerate, grid, max_iters, tol) -> Dict

One-line-per-key fingerprint for JSONL headers. CPU model, BLAS config,
GPU device descriptor, run config.
"""
function hardware_fingerprint(; gpu_backend=nothing, use_accelerate::Bool=false,
                                 grid::AbstractString="?", max_iters::Integer=0,
                                 tol::Real=0.0)
    info = Sys.cpu_info()
    gpu_str = if gpu_backend === nothing
        "none"
    else
        try
            if gpu_backend.name == "Metal"
                string(gpu_backend.name, ":", string(Main.Metal.device().name))
            else
                gpu_backend.name
            end
        catch
            gpu_backend.name
        end
    end
    return Dict(
        "cpu_model"      => length(info) > 0 ? info[1].model : "unknown",
        "cpu_threads"    => length(info),
        "machine"        => Sys.MACHINE,
        "julia_version"  => string(VERSION),
        "blas_config"    => string(BLAS.get_config()),
        "use_accelerate" => use_accelerate,
        "gpu_backend"    => gpu_str,
        "grid"           => grid,
        "tol"            => tol,
        "max_iters"      => max_iters,
        "timestamp"      => string(now()),
    )
end

# ── JSONL writer ─────────────────────────────────────────────────────

function jsonl_value(v)
    if v isa AbstractString
        '"' * replace(string(v), '\\' => "\\\\", '"' => "\\\"", '\n' => "\\n") * '"'
    elseif v isa Bool
        v ? "true" : "false"
    elseif v isa Real
        isfinite(v) ? string(v) : "null"
    elseif v === nothing
        "null"
    else
        '"' * replace(string(v), '\\' => "\\\\", '"' => "\\\"", '\n' => "\\n") * '"'
    end
end

jsonl_record(d::AbstractDict) =
    "{" * join(["$(jsonl_value(string(k))): $(jsonl_value(v))" for (k, v) in d], ", ") * "}"

# ── Timing ───────────────────────────────────────────────────────────

"""
    timed_runs(f, args...; samples=3) -> NamedTuple

Run `f(args...)` once for warmup, then `samples` timed runs. Returns
median, min, max wall time and the result of the last run.
"""
function timed_runs(f, args...; samples::Int=3)
    f(args...)  # warmup
    times = Float64[]
    local last
    for _ in 1:samples
        GC.gc(false)
        t0 = time_ns()
        last = f(args...)
        push!(times, (time_ns() - t0) * 1e-9)
    end
    return (
        wall_time_s = median(times),
        wall_min_s  = minimum(times),
        wall_max_s  = maximum(times),
        samples     = samples,
        result      = last,
    )
end
