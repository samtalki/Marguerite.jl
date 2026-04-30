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

# Experiment 3: Spectraplex FW iteration with AppleAccelerate
# ===========================================================
#
# Trace regression on the PSD cone:
#   f(X) = ½ ‖A X − B‖²_F
#   X ∈ S₊ⁿ with tr(X) = r   (Spectraplex)
#   vectorized as an n²-vector for `solve`
#
# Per FW iter the solver does:
#   ∇f(x) = vec(Aᵀ(A X − B))     ← two matmuls per gradient   (matmul-heavy)
#   _lmo_and_gap! on Spectraplex ← symmetrize + eigen           (LAPACK heavy)
#   trial update (rank-1)                                        (cheap)
#
# Pre-flight finding (recorded so the experiment is interpreted correctly):
# `eigen(Symmetric{Float64}, n=100..1000)` shows essentially no speedup from
# AppleAccelerate vs OpenBLAS on M5 Pro (0.92x..1.18x). The matmul path
# (`mul!`) does benefit from AMX via Accelerate. So in this experiment, any
# AppleAccelerate uplift comes from the gradient evaluations, not the eigen.
#
# Conditions: default (OpenBLAS LAPACK + LP64-Accelerate BLAS, Julia 1.12
# default on Apple Silicon) vs `BENCH_USE_ACCELERATE=1` (full Accelerate
# forwarding via AppleAccelerate.jl).
#
# Usage:
#   julia --project=examples examples/bench_spectraplex_apple.jl --grid=small
#   BENCH_USE_ACCELERATE=1 julia --project=examples examples/bench_spectraplex_apple.jl --grid=small
#
# Output: examples/results/bench_spectraplex_apple_<grid>_<config>_<timestamp>.jsonl

using Marguerite
using Marguerite: solve, Spectraplex
using LinearAlgebra
using Random: Xoshiro
using Statistics: median
using Printf, Pkg, Dates

const USE_ACCEL = get(ENV, "BENCH_USE_ACCELERATE", "0") == "1"
if USE_ACCEL
    @eval using AppleAccelerate
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

const N_VALUES = if GRID == "smoke"
    (50,)
elseif GRID == "small"
    (50, 100, 200)
elseif GRID == "full"
    (50, 100, 200, 500)
else
    error("unknown --grid=$(GRID); choose smoke|small|full")
end

const PRECISIONS = (Float32, Float64)
const MAX_ITERS = GRID == "smoke" ? 50 : 200
const TOL = 0.0  # fixed iteration regime — measure per-iter cost only

# ── Hardware fingerprint ─────────────────────────────────────────────

function hardware_fingerprint()
    info = Sys.cpu_info()
    return Dict(
        "cpu_model"      => length(info) > 0 ? info[1].model : "unknown",
        "cpu_threads"    => length(info),
        "machine"        => Sys.MACHINE,
        "julia_version"  => string(VERSION),
        "blas_config"    => string(BLAS.get_config()),
        "use_accelerate" => USE_ACCEL,
        "grid"           => GRID,
        "max_iters"      => MAX_ITERS,
        "tol"            => TOL,
        "timestamp"      => string(now()),
    )
end

# ── Trace regression problem generator ───────────────────────────────

function make_trace_regression(::Type{T}, n; seed=42, rank=3, noise=T(1e-3)) where T
    rng = Xoshiro(seed)
    # Random A scaled so ||A||₂ ≈ 1 (well-conditioned)
    A = randn(rng, T, n, n) ./ T(sqrt(n))
    # Ground-truth low-rank PSD with unit trace
    U = randn(rng, T, n, rank) ./ T(sqrt(n))
    X_true = U * U'
    X_true ./= tr(X_true)
    B = A * X_true + noise * randn(rng, T, n, n)
    return A, B
end

function make_objective_grad(A::Matrix{T}, B::Matrix{T}, n) where T
    AX_buf = similar(A)  # (n, n)
    f = function (x)
        X = reshape(x, n, n)
        mul!(AX_buf, A, X)
        AX_buf .-= B
        return T(0.5) * sum(abs2, AX_buf)
    end
    grad! = function (g, x)
        X = reshape(x, n, n)
        mul!(AX_buf, A, X)        # AX_buf = A X
        AX_buf .-= B               # AX - B
        G = reshape(g, n, n)
        mul!(G, A', AX_buf)        # g = A' (AX - B)
        return g
    end
    return f, grad!
end

# ── Custom timer (3 timed runs, median) ──────────────────────────────

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

# ── JSONL helpers (same as Experiment 2) ─────────────────────────────

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

# ── Sub-experiment timers (per phase) ────────────────────────────────
# Time gradient + eigen separately so we can see which dominates and which
# benefits (or doesn't) from AppleAccelerate.

function time_grad(grad!, x, samples=10)
    g = similar(x)
    grad!(g, x)  # warmup
    times = Float64[]
    for _ in 1:samples
        t0 = time_ns()
        grad!(g, x)
        push!(times, (time_ns() - t0) * 1e-9)
    end
    return median(times)
end

function time_eigen(M::Matrix, samples=10)
    eigen(Symmetric(M))  # warmup
    times = Float64[]
    for _ in 1:samples
        t0 = time_ns()
        eigen(Symmetric(M))
        push!(times, (time_ns() - t0) * 1e-9)
    end
    return median(times)
end

# ── Sweep ────────────────────────────────────────────────────────────

function run_sweep(out_io)
    fp = hardware_fingerprint()
    println(out_io, jsonl_record(merge(Dict("kind" => "fingerprint"), fp)))
    flush(out_io)

    println("\n", "="^104)
    println("  Marguerite.jl — Experiment 3 (Spectraplex FW + AppleAccelerate), grid=$GRID")
    println("  BLAS: $(USE_ACCEL ? "AppleAccelerate (full Accelerate forwarding)" : "default (Julia 1.12 LBT)")")
    println("="^104)

    @printf("\n%-8s %-5s | %12s %12s %12s | %12s %12s %12s\n",
            "n", "T", "solve(s)", "min(s)", "max(s)",
            "grad(ms)", "eigen(ms)", "iter(ms)")
    println("-"^104)

    for n in N_VALUES, T in PRECISIONS
        A, B = make_trace_regression(T, n)
        f, grad! = make_objective_grad(A, B, n)
        lmo = Spectraplex(n, T(1.0))
        x0 = vec(Matrix{T}(I, n, n) ./ T(n))

        # Time the full FW solve
        run_solve = () -> solve(f, lmo, copy(x0); grad=grad!,
                                max_iters=MAX_ITERS, tol=T(TOL))
        r = timed_runs(run_solve)
        x_star, res = r.result

        # Time individual phases (gradient and eigen) at a representative iterate
        g = similar(x0)
        grad!(g, x_star)  # use the converged point's gradient
        t_grad = time_grad(grad!, x_star)

        # Build the symmetric gradient matrix that goes into eigen (per Spectraplex LMO)
        G_mat = reshape(g, n, n)
        G_sym = (G_mat .+ G_mat') ./ T(2)
        t_eigen = time_eigen(Matrix(G_sym))

        per_iter_ms = r.wall_time_s / MAX_ITERS * 1000

        row = Dict(
            "kind" => "measurement",
            "n" => n, "T" => string(T),
            "wall_time_s" => r.wall_time_s,
            "wall_min_s"  => r.wall_min_s,
            "wall_max_s"  => r.wall_max_s,
            "samples"     => r.samples,
            "iters"       => res.iterations,
            "final_obj"   => Float64(res.objective),
            "final_gap"   => Float64(res.gap),
            "grad_ms"     => t_grad * 1000,
            "eigen_ms"    => t_eigen * 1000,
            "per_iter_ms" => per_iter_ms,
        )
        println(out_io, jsonl_record(row)); flush(out_io)
        @printf("%-8d %-5s | %12.4f %12.4f %12.4f | %12.4f %12.4f %12.4f\n",
                n, string(T), r.wall_time_s, r.wall_min_s, r.wall_max_s,
                t_grad * 1000, t_eigen * 1000, per_iter_ms)
    end

    println("\n", "="^104)
end

# ── Main ─────────────────────────────────────────────────────────────

function main()
    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)
    config_tag = USE_ACCEL ? "accelerate" : "openblas"
    ts = Dates.format(now(), "yyyymmdd-HHMMSS")
    out_path = joinpath(results_dir, "bench_spectraplex_apple_$(GRID)_$(config_tag)_$(ts).jsonl")
    open(out_path, "w") do io
        run_sweep(io)
    end
    println("\nWrote: $out_path")
end

main()
