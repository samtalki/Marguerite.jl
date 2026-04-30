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

# Experiment 2: batched broadcast oracles benchmark on Apple Silicon
# ===================================================================
#
# Sweeps (n, B, T, oracle) and times three conditions per cell:
#   serial_cpu     — solve() called B times sequentially
#   batched_cpu    — batch_solve() on Matrix{T}
#   batched_gpu    — batch_solve() on the device array of the loaded backend
#                    (Metal/CUDA/AMDGPU). Skipped if no GPU is available.
#
# AppleAccelerate is loaded only when BENCH_USE_ACCELERATE=1 in the env, since
# `using AppleAccelerate` redirects BLAS/LAPACK globally for the session.
# Run the script twice for the OpenBLAS-vs-Accelerate comparison.
#
# Usage:
#   julia --project=examples examples/bench_batched_oracles.jl                  # default grid=small, OpenBLAS
#   BENCH_USE_ACCELERATE=1 julia --project=examples examples/bench_batched_oracles.jl  # AppleAccelerate
#   julia --project=examples examples/bench_batched_oracles.jl --grid=smoke     # quick sanity (~30 s)
#   julia --project=examples examples/bench_batched_oracles.jl --grid=full      # full sweep (~45 min)
#
# Output: examples/results/bench_batched_oracles_<config>_<timestamp>.jsonl
# (one JSON record per (n, B, T, oracle, condition); plus a summary table on stdout)

using Marguerite
# AppleAccelerate also exports `solve` (sparse-factorization). Import the names
# we need explicitly so it doesn't shadow Marguerite's solve when loaded.
using Marguerite: solve, batch_solve, Box, ProbSimplex, Simplex, Spectraplex, ScalarBox
using LinearAlgebra
using BenchmarkTools  # imported for ergonomics; we use a custom timer (see timed_runs)
using Random: Xoshiro
using Statistics: median
using Printf
using Pkg
using Dates

# ── Configuration ────────────────────────────────────────────────────

const USE_ACCEL = get(ENV, "BENCH_USE_ACCELERATE", "0") == "1"
if USE_ACCEL
    @eval using AppleAccelerate
end

# GPU detection (priority: Metal → CUDA → AMDGPU)
const GPU_BACKEND = let
    backend = nothing
    if get(ENV, "BENCH_NO_GPU", "0") != "1"
        if Sys.isapple()
            try
                @eval using Metal
                if Metal.functional()
                    backend = (name="Metal", arr=Metal.MtlArray, sync=Metal.synchronize)
                end
            catch
            end
        end
        if backend === nothing
            try
                @eval using CUDA
                if CUDA.functional()
                    backend = (name="CUDA", arr=CUDA.CuArray, sync=CUDA.synchronize)
                end
            catch
            end
        end
        if backend === nothing
            try
                @eval using AMDGPU
                if AMDGPU.functional()
                    backend = (name="AMDGPU", arr=AMDGPU.ROCArray, sync=AMDGPU.synchronize)
                end
            catch
            end
        end
    end
    backend
end

const GRID = let
    g = "small"
    for arg in ARGS
        if startswith(arg, "--grid=")
            g = arg[8:end]
        end
    end
    g
end

# Fixed-iteration regime: tol=0 forces every condition to do the same number
# of FW iterations (max_iters), so wall-time differences reflect only the
# per-iteration cost — not convergence-rate noise.
const TOL = 0.0
const MAX_ITERS = GRID == "smoke" ? 200 :
                  GRID == "small" ? 500 :
                                    1000

# Sweep grids.
const GRID_CELLS = if GRID == "smoke"
    [(100, 64)]
elseif GRID == "small"
    [(100, 64), (100, 1024), (1000, 256), (10000, 64)]
elseif GRID == "full"
    [(n, B) for n in (100, 1000, 10000) for B in (64, 256, 1024)]
else
    error("unknown --grid=$(GRID); choose smoke|small|full")
end

const PRECISIONS = (Float32, Float64)  # both axes always; Metal F64 is skipped at run-time
const ORACLES = GRID == "smoke" ?
    (("Box", T -> Box(zero(T), one(T))), ) :
    (("Box", T -> Box(zero(T), one(T))),
     ("ProbSimplex", T -> ProbSimplex(one(T))))  # parametric on T to match X eltype

# ── Hardware / environment fingerprint ───────────────────────────────

function hardware_fingerprint()
    info = Sys.cpu_info()
    blas_cfg = string(BLAS.get_config())
    gpu_str = GPU_BACKEND === nothing ? "none" : begin
        try
            string(GPU_BACKEND.name, ":", GPU_BACKEND.arr === Metal.MtlArray ? string(Metal.current_device()) : "device")
        catch
            GPU_BACKEND.name
        end
    end
    return Dict(
        "cpu_model"      => length(info) > 0 ? info[1].model : "unknown",
        "cpu_threads"    => length(info),
        "machine"        => Sys.MACHINE,
        "julia_version"  => string(VERSION),
        "blas_config"    => blas_cfg,
        "use_accelerate" => USE_ACCEL,
        "gpu_backend"    => gpu_str,
        "grid"           => GRID,
        "tol"            => TOL,
        "max_iters"      => MAX_ITERS,
        "timestamp"      => string(now()),
    )
end

# ── Problem generator ────────────────────────────────────────────────

function make_qp(::Type{T}, n::Int, B::Int; seed::Int=42) where {T}
    rng = Xoshiro(seed)
    A = randn(rng, T, n, n)
    # Normalize Q so eigenvalues are O(1) — keeps the FW gap well-scaled
    # across all (n, B) cells. Without this, larger n produces O(n) gaps.
    Q = (A'A) ./ T(n) + T(0.1) * I  # `+ I` adds UniformScaling; `.+ I` would error
    C = randn(rng, T, n, B) ./ T(sqrt(n))
    X0 = ones(T, n, B) ./ T(n)  # interior; ProbSimplex-feasible by construction
    return (Q=Matrix{T}(Q), C=C, X0=X0)
end

function batch_objgrad_factory(Q::AbstractMatrix{T}, C::AbstractMatrix{T}) where {T}
    buf = similar(C)  # (n, B); shares array type with Q/C (CPU or GPU)
    f_batch = function (X)
        # Vectorized + dims=1 reductions — GPU-safe.
        # f_b = 0.5 * x_b' Q x_b + c_b' x_b
        mul!(buf, Q, X)                               # buf = Q X     (n, B)
        obj_dev = T(0.5) .* sum(X .* buf; dims=1) .+ # 0.5 x'Qx       (1, B)
                          sum(C .* X;   dims=1)      # + c'x          (1, B)
        return collect(vec(obj_dev))                  # always a CPU Vector{T}
    end
    grad_batch! = function (G, X)
        mul!(G, Q, X)
        G .+= C
    end
    return f_batch, grad_batch!
end

function scalar_objgrad_factory(Q::AbstractMatrix{T}, c::AbstractVector{T}) where {T}
    f = x -> T(0.5) * dot(x, Q * x) + dot(c, x)
    grad! = (g, x) -> (mul!(g, Q, x); g .+= c)
    return f, grad!
end

# ── Conditions ───────────────────────────────────────────────────────

function run_serial_cpu(prob, lmo)
    Q, C, X0 = prob.Q, prob.C, prob.X0
    n, B = size(X0)
    T = eltype(X0)
    X_out = similar(X0)
    iters = 0
    gap_max = zero(T)
    for b in 1:B
        c_b = view(C, :, b)
        f, grad! = scalar_objgrad_factory(Q, collect(c_b))
        x_b, res = solve(f, lmo, copy(view(X0, :, b)); grad=grad!,
                         tol=T(TOL), max_iters=MAX_ITERS)
        X_out[:, b] = x_b
        iters = max(iters, res.iterations)
        gap_max = max(gap_max, T(res.gap))
    end
    return X_out, iters, gap_max
end

function run_batched(prob, lmo)
    f_batch, grad_batch! = batch_objgrad_factory(prob.Q, prob.C)
    X, res = batch_solve(f_batch, lmo, copy(prob.X0);
                         grad_batch=grad_batch!,
                         tol=eltype(prob.X0)(TOL), max_iters=MAX_ITERS)
    return X, res.iterations, maximum(res.gaps)
end

function run_batched_gpu(prob, lmo)
    Arr = GPU_BACKEND.arr
    Q_d = Arr(prob.Q); C_d = Arr(prob.C); X0_d = Arr(prob.X0)
    f_batch, grad_batch! = batch_objgrad_factory(Q_d, C_d)
    X, res = batch_solve(f_batch, lmo, X0_d;
                         grad_batch=grad_batch!,
                         tol=eltype(prob.X0)(TOL), max_iters=MAX_ITERS)
    GPU_BACKEND.sync()  # ensure all kernels complete before timer stops
    return Array(X), res.iterations, maximum(res.gaps)
end

# ── Custom timer (3 timed runs, median; warmup once) ─────────────────

function timed_runs(label::String, f, args...; samples::Int=3)
    # Warmup
    f(args...)
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

# ── JSONL writer (no JSON3 dep — we control the schema) ──────────────

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

function jsonl_record(d::AbstractDict)
    parts = ["$(jsonl_value(string(k))): $(jsonl_value(v))" for (k, v) in d]
    "{" * join(parts, ", ") * "}"
end

# ── Sweep ────────────────────────────────────────────────────────────

function run_sweep(out_io)
    fp = hardware_fingerprint()
    println(out_io, jsonl_record(merge(Dict("kind" => "fingerprint"), fp)))
    flush(out_io)

    println("\n", "="^88)
    println("  Marguerite.jl — Experiment 2 (batched broadcast oracles), grid=$GRID")
    println("  BLAS: $(USE_ACCEL ? "AppleAccelerate" : "default (OpenBLAS / Accelerate-LP64-mixed)")")
    println("  GPU:  $(GPU_BACKEND === nothing ? "none" : GPU_BACKEND.name)")
    println("="^88)

    @printf("\n%-22s %-7s %-5s %-7s | %12s %12s %12s | %s\n",
            "case", "oracle", "T", "cond", "wall(s)", "max_gap", "iters", "verify")
    println("-"^120)

    for (n, B) in GRID_CELLS, T in PRECISIONS, (oracle_name, oracle_factory) in ORACLES
        prob = make_qp(T, n, B)
        lmo = oracle_factory(T)

        case = "n=$n B=$B"
        all_X = Dict{String, Matrix{T}}()

        # Serial CPU
        let r = timed_runs("$case-$oracle_name-$T-serial", run_serial_cpu, prob, lmo)
            X, iters, gap = r.result
            all_X["serial_cpu"] = X
            row = Dict(
                "kind" => "measurement",
                "n" => n, "B" => B, "T" => string(T), "oracle" => oracle_name,
                "condition" => "serial_cpu",
                "wall_time_s" => r.wall_time_s, "wall_min_s" => r.wall_min_s, "wall_max_s" => r.wall_max_s,
                "samples" => r.samples,
                "max_gap" => Float64(gap), "iters" => iters,
            )
            println(out_io, jsonl_record(row)); flush(out_io)
            @printf("%-22s %-7s %-5s %-7s | %12.4f %12.2e %12d | (ref)\n",
                    case, oracle_name, string(T), "serial", r.wall_time_s, gap, iters)
        end

        # Batched CPU
        let r = timed_runs("$case-$oracle_name-$T-batched_cpu", run_batched, prob, lmo)
            X, iters, gap = r.result
            all_X["batched_cpu"] = X
            err = norm(all_X["serial_cpu"] - X) / max(1, norm(all_X["serial_cpu"]))
            verify = err < 1e-2 ? "ok" : @sprintf("rel=%.1e", err)
            row = Dict(
                "kind" => "measurement",
                "n" => n, "B" => B, "T" => string(T), "oracle" => oracle_name,
                "condition" => "batched_cpu",
                "wall_time_s" => r.wall_time_s, "wall_min_s" => r.wall_min_s, "wall_max_s" => r.wall_max_s,
                "samples" => r.samples,
                "max_gap" => Float64(gap), "iters" => iters,
                "rel_err_vs_serial" => err,
            )
            println(out_io, jsonl_record(row)); flush(out_io)
            @printf("%-22s %-7s %-5s %-7s | %12.4f %12.2e %12d | %s\n",
                    case, oracle_name, string(T), "batched", r.wall_time_s, gap, iters, verify)
        end

        # Batched GPU — Metal does NOT support Float64 (HW limitation),
        # so we skip that combination explicitly with a recorded reason.
        if GPU_BACKEND !== nothing
            metal_no_f64 = (GPU_BACKEND.name == "Metal" && T === Float64)
            if metal_no_f64
                row = Dict(
                    "kind" => "measurement", "n" => n, "B" => B, "T" => string(T),
                    "oracle" => oracle_name,
                    "condition" => "batched_$(lowercase(GPU_BACKEND.name))",
                    "skipped" => true,
                    "skip_reason" => "Metal does not support Float64",
                )
                println(out_io, jsonl_record(row)); flush(out_io)
                @printf("%-22s %-7s %-5s %-7s | %12s %12s %12s | skip (Metal F64)\n",
                        case, oracle_name, string(T),
                        lowercase(GPU_BACKEND.name)[1:min(6, end)],
                        "—", "—", "—")
            else
                r = try
                    timed_runs("$case-$oracle_name-$T-batched_gpu", run_batched_gpu, prob, lmo)
                catch e
                    @warn "GPU run failed for $case $oracle_name $T" exception=(e, catch_backtrace())
                    nothing
                end
                if r !== nothing
                    X, iters, gap = r.result
                    # Loose verification at fixed-iteration regime: paths can diverge
                    # numerically without converging. Tag mismatches but don't fail.
                    err = norm(all_X["serial_cpu"] - X) / max(1, norm(all_X["serial_cpu"]))
                    verify = err < 1e-2 ? "ok" : @sprintf("rel=%.1e", err)
                    row = Dict(
                        "kind" => "measurement",
                        "n" => n, "B" => B, "T" => string(T), "oracle" => oracle_name,
                        "condition" => "batched_$(lowercase(GPU_BACKEND.name))",
                        "wall_time_s" => r.wall_time_s, "wall_min_s" => r.wall_min_s, "wall_max_s" => r.wall_max_s,
                        "samples" => r.samples,
                        "max_gap" => Float64(gap), "iters" => iters,
                        "rel_err_vs_serial" => err,
                    )
                    println(out_io, jsonl_record(row)); flush(out_io)
                    @printf("%-22s %-7s %-5s %-7s | %12.4f %12.2e %12d | %s\n",
                            case, oracle_name, string(T),
                            lowercase(GPU_BACKEND.name)[1:min(6, end)],
                            r.wall_time_s, gap, iters, verify)
                end
            end
        end
    end

    println("\n", "="^88)
    println("  Done. JSONL: $(out_io === stdout ? "stdout" : out_io)")
    println("="^88)
end

# ── Main ─────────────────────────────────────────────────────────────

function main()
    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)
    config_tag = USE_ACCEL ? "accelerate" : "openblas"
    gpu_tag = GPU_BACKEND === nothing ? "nogpu" : lowercase(GPU_BACKEND.name)
    ts = Dates.format(now(), "yyyymmdd-HHMMSS")
    out_path = joinpath(results_dir, "bench_batched_oracles_$(GRID)_$(config_tag)_$(gpu_tag)_$(ts).jsonl")
    open(out_path, "w") do io
        run_sweep(io)
    end
    println("\nWrote: $out_path")
end

main()
