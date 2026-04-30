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

# Experiment 1: batched bilevel on Apple Silicon
# ===============================================
#
# Tikhonov-regularized QP on the simplex with closed-form bilevel gradient
# for ground-truth verification:
#
#   inner_b(x, θ) = ½ ‖x‖² − (θ + c_b)' x       on  ProbSimplex
#   outer_b(x)    = ½ ‖x − x_target_b‖²
#
# Inner has H = I (well-conditioned). Interior optimum:
#   x*_b(θ) = (θ + c_b) − mean(θ + c_b) + 1/n
#
# Closed-form bilevel gradient (interior):
#   dθ_j = Σ_b ((x*_b,j − x_target_b,j) − (1/n) Σ_i (x*_b,i − x_target_b,i))
#
# We pick θ and c_b small enough that x*_b stays interior, then verify the
# numerical hypergradient against the closed form.
#
# Conditions per cell:
#   serial_cpu     — call `bilevel_solve` B times sequentially
#   batched_cpu    — `batch_bilevel_solve` on Matrix{T}
#   batched_metal  — `batch_bilevel_solve` on MtlMatrix{T}; F32 only
#                    (Metal does not support F64)
#
# Times forward (inner solve) and pullback (gradient) separately so the
# serial-Julia pullback bottleneck (see issues/future_gpu_optimizations.md)
# is visible.
#
# Usage:
#   julia --project=examples examples/bench_batched_bilevel.jl --grid=small
#   BENCH_USE_ACCELERATE=1 julia --project=examples examples/bench_batched_bilevel.jl --grid=small

using Marguerite
using Marguerite: solve, bilevel_solve, batch_solve, batch_bilevel_solve, ProbSimplex
using LinearAlgebra
using Random: Xoshiro
using Statistics: median
using Printf, Pkg, Dates

const USE_ACCEL = get(ENV, "BENCH_USE_ACCELERATE", "0") == "1"
if USE_ACCEL
    @eval using AppleAccelerate
end

const GPU_BACKEND = let
    # GPU bilevel is currently blocked by scalar-indexing errors inside
    # `batch_bilevel_solve`'s per-problem KKT adjoint / cross-derivative path
    # when X is a real device array (Metal/CUDA/AMDGPU). Tracked in
    # issues/future_gpu_optimizations.md → "gpu compatibility for batch_bilevel".
    # Until that lands, this experiment compares only serial CPU vs batched CPU.
    if get(ENV, "BENCH_FORCE_GPU", "0") == "1" && Sys.isapple()
        try
            @eval using Metal
            Metal.functional() ?
                (name="Metal", arr=Metal.MtlArray, sync=Metal.synchronize) : nothing
        catch
            nothing
        end
    else
        nothing
    end
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

const GRID_CELLS = if GRID == "smoke"
    [(100, 8)]
elseif GRID == "small"
    [(100, 16), (100, 64), (1000, 16), (1000, 64)]
elseif GRID == "full"
    [(n, B) for n in (100, 1000) for B in (16, 64, 256, 1024)]
else
    error("unknown --grid=$(GRID); choose smoke|small|full")
end

const PRECISIONS = (Float32, Float64)
const INNER_MAX_ITERS = GRID == "smoke" ? 200 : 500
const TOL = 0.0   # fixed-iteration regime: every condition does INNER_MAX_ITERS iters
                  # so wall-time differences reflect per-iter cost, not convergence-rate noise

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
        "gpu_backend"    => GPU_BACKEND === nothing ? "none" : GPU_BACKEND.name,
        "grid"           => GRID,
        "tol"            => TOL,
        "max_iters"      => INNER_MAX_ITERS,
        "timestamp"      => string(now()),
    )
end

# ── Tikhonov simplex bilevel problem ─────────────────────────────────

struct TikhonovBilevel{T, V<:AbstractVector{T}, M<:AbstractMatrix{T}}
    θ::V                          # (n,)    shared parameter
    C::M                          # (n, B)  per-problem bias c_b
    X_target::M                   # (n, B)  per-problem target
end

function make_problem(::Type{T}, n::Int, B::Int; seed::Int=42, scale::T=T(0.05)) where {T}
    rng = Xoshiro(seed)
    θ = scale * randn(rng, T, n)
    C = scale * randn(rng, T, n, B)
    X_target = max.(zero(T), scale * randn(rng, T, n, B) .+ T(1.0/n))
    X_target ./= sum(X_target, dims=1)  # project to simplex
    return TikhonovBilevel(θ, C, X_target)
end

# Closed-form interior optimum for inner_b(x; θ + c_b) on ProbSimplex with H=I
function inner_optimum_closed(prob::TikhonovBilevel{T}, n, B) where T
    X = similar(prob.C)
    @inbounds for b in 1:B
        cb = @view prob.C[:, b]
        v = prob.θ .+ cb
        m = sum(v) / T(n)
        X[:, b] .= v .- m .+ T(1.0/n)
    end
    return X
end

# Closed-form bilevel gradient (interior). Returns dθ ∈ R^n.
function bilevel_grad_closed(prob::TikhonovBilevel{T}, n, B) where T
    Xstar = inner_optimum_closed(prob, n, B)
    diff = Xstar .- prob.X_target  # (n, B)
    # dθ_j = sum_b (diff[j,b] - (1/n) sum_i diff[i,b])
    col_means = sum(diff; dims=1) ./ T(n)  # (1, B)
    centered = diff .- col_means
    return vec(sum(centered; dims=2))  # (n,)
end

# ── Marguerite-format closures (manual gradient and outer) ───────────

function batched_inner(prob::TikhonovBilevel{T}, B::Int) where T
    Q = T(0.5)
    function f_batch(X::AbstractMatrix, θ::AbstractVector)
        # 0.5 * ‖x_b‖² − (θ + c_b)'x_b   per problem b
        # Vectorized for GPU friendliness
        biases = @. prob.C + θ                # (n, B); θ broadcasts on rows
        out = T(0.5) .* sum(X .* X; dims=1) .- sum(biases .* X; dims=1)  # (1, B)
        return collect(vec(out))               # always CPU Vector{T}
    end
    return f_batch
end

function batched_inner_grad!(prob::TikhonovBilevel{T}) where T
    function grad!(G::AbstractMatrix, X::AbstractMatrix, θ::AbstractVector)
        # ∇_x = X − (θ + C)
        @. G = X - prob.C - θ   # θ broadcasts along rows
    end
    return grad!
end

function batched_outer(prob::TikhonovBilevel{T}, B::Int) where T
    function outer_batch(X::AbstractMatrix)
        diff = X .- prob.X_target
        return collect(vec(T(0.5) .* sum(diff .* diff; dims=1)))
    end
    return outer_batch
end

# ── Conditions ───────────────────────────────────────────────────────

function run_serial_cpu(prob::TikhonovBilevel{T}, n, B) where T
    dθ_total = zeros(T, n)
    Xstar = zeros(T, n, B)
    forward_t = 0.0
    pullback_t = 0.0
    for b in 1:B
        cb = collect(@view prob.C[:, b])
        x_target_b = collect(@view prob.X_target[:, b])
        # closures over scalars
        inner_b = (x, θ) -> T(0.5) * dot(x, x) - dot(prob.θ .+ cb, x) - dot(θ .- prob.θ, x)
        # actually we want θ to be the differentiable parameter. Cleaner:
        inner_b_clean = (x, θ) -> T(0.5) * dot(x, x) - dot(θ .+ cb, x)
        grad_b! = (g, x, θ) -> (g .= x .- θ .- cb)
        outer_b = x -> T(0.5) * sum((x .- x_target_b).^2)
        x0 = fill(T(1.0/n), n)
        # Forward only
        forward_t += @elapsed solve(inner_b_clean, ProbSimplex(one(T)), copy(x0), prob.θ;
                                    grad=grad_b!, max_iters=INNER_MAX_ITERS, tol=T(TOL))
        # Full bilevel (forward + pullback)
        local x_b, dθ_b
        pullback_t += @elapsed begin
            x_b, dθ_b, _ = bilevel_solve(outer_b, inner_b_clean, ProbSimplex(one(T)),
                                          copy(x0), prob.θ;
                                          grad=grad_b!, max_iters=INNER_MAX_ITERS, tol=T(TOL))
        end
        Xstar[:, b] .= x_b
        dθ_total .+= dθ_b
    end
    # Adjust pullback_t to subtract forward (since pullback_t includes both)
    pullback_only = pullback_t - forward_t
    return Xstar, dθ_total, forward_t, pullback_only
end

function run_batched_cpu(prob::TikhonovBilevel{T}, n, B) where T
    inner = batched_inner(prob, B)
    grad! = batched_inner_grad!(prob)
    outer = batched_outer(prob, B)
    X0 = fill(T(1.0/n), n, B)

    forward_t = @elapsed batch_solve(inner, ProbSimplex(one(T)), X0, prob.θ;
                                      grad_batch=grad!, max_iters=INNER_MAX_ITERS, tol=T(TOL))
    local Xstar, dθ
    full_t = @elapsed begin
        Xstar, dθ, _ = batch_bilevel_solve(outer, inner, ProbSimplex(one(T)), X0, prob.θ;
                                            grad_batch=grad!, max_iters=INNER_MAX_ITERS, tol=T(TOL))
    end
    return Xstar, dθ, forward_t, full_t - forward_t
end

function run_batched_gpu(prob::TikhonovBilevel{T}, n, B) where T
    Arr = GPU_BACKEND.arr
    # Move all problem data — θ, C, X_target — to the device so user closures
    # that broadcast over them produce all-device kernel calls.
    θ_d = Arr(prob.θ)
    C_d = Arr(prob.C)
    X_target_d = Arr(prob.X_target)
    prob_d = TikhonovBilevel(θ_d, C_d, X_target_d)  # generic struct preserves device array types
    inner = batched_inner(prob_d, B)
    grad! = batched_inner_grad!(prob_d)
    outer = batched_outer(prob_d, B)
    X0_d = Arr(fill(T(1.0/n), n, B))

    forward_t = @elapsed begin
        batch_solve(inner, ProbSimplex(one(T)), X0_d, θ_d;
                    grad_batch=grad!, max_iters=INNER_MAX_ITERS, tol=T(TOL))
        GPU_BACKEND.sync()
    end
    local Xstar, dθ
    full_t = @elapsed begin
        Xstar, dθ, _ = batch_bilevel_solve(outer, inner, ProbSimplex(one(T)), X0_d, θ_d;
                                            grad_batch=grad!, max_iters=INNER_MAX_ITERS, tol=T(TOL))
        GPU_BACKEND.sync()
    end
    return Array(Xstar), Array(dθ), forward_t, full_t - forward_t
end

# ── JSONL helpers ────────────────────────────────────────────────────

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

# ── Sweep ────────────────────────────────────────────────────────────

function timed_with_warmup(f::Function; samples::Int=2)
    f()  # warmup
    times = Float64[]
    local last
    for _ in 1:samples
        GC.gc(false)
        t0 = time_ns()
        last = f()
        push!(times, (time_ns() - t0) * 1e-9)
    end
    return (median=median(times), result=last)
end

function run_sweep(out_io)
    fp = hardware_fingerprint()
    println(out_io, jsonl_record(merge(Dict("kind" => "fingerprint"), fp)))
    flush(out_io)

    println("\n", "="^120)
    println("  Marguerite.jl — Experiment 1 (batched bilevel), grid=$GRID")
    println("  BLAS: $(USE_ACCEL ? "AppleAccelerate" : "default")")
    println("  GPU:  $(GPU_BACKEND === nothing ? "none" : GPU_BACKEND.name)")
    println("="^120)

    @printf("\n%-15s %-5s %-10s | %12s %12s | %14s %12s\n",
            "case", "T", "cond", "forward(s)", "pullback(s)", "dθ_err", "Xstar_err")
    println("-"^120)

    for (n, B) in GRID_CELLS, T in PRECISIONS
        prob = make_problem(T, n, B)
        # Closed-form ground truth for verification
        Xstar_closed = inner_optimum_closed(prob, n, B)
        dθ_closed    = bilevel_grad_closed(prob, n, B)

        case = "n=$n B=$B"

        # --- serial CPU ---
        let
            r = timed_with_warmup(samples=2) do
                run_serial_cpu(prob, n, B)
            end
            Xstar, dθ, fwd_t, pb_t = r.result
            dθ_err = norm(dθ .- dθ_closed) / max(one(T), norm(dθ_closed))
            Xstar_err = norm(Xstar .- Xstar_closed) / max(one(T), norm(Xstar_closed))
            row = Dict(
                "kind"=>"measurement", "n"=>n, "B"=>B, "T"=>string(T),
                "condition"=>"serial_cpu",
                "forward_s"=>fwd_t, "pullback_s"=>pb_t,
                "total_s"=>fwd_t + pb_t,
                "dtheta_rel_err"=>Float64(dθ_err),
                "xstar_rel_err"=>Float64(Xstar_err),
            )
            println(out_io, jsonl_record(row)); flush(out_io)
            @printf("%-15s %-5s %-10s | %12.4f %12.4f | %14.2e %12.2e\n",
                    case, string(T), "serial", fwd_t, pb_t, dθ_err, Xstar_err)
        end

        # --- batched CPU ---
        let
            r = timed_with_warmup(samples=2) do
                run_batched_cpu(prob, n, B)
            end
            Xstar, dθ, fwd_t, pb_t = r.result
            dθ_err = norm(dθ .- dθ_closed) / max(one(T), norm(dθ_closed))
            Xstar_err = norm(Xstar .- Xstar_closed) / max(one(T), norm(Xstar_closed))
            row = Dict(
                "kind"=>"measurement", "n"=>n, "B"=>B, "T"=>string(T),
                "condition"=>"batched_cpu",
                "forward_s"=>fwd_t, "pullback_s"=>pb_t,
                "total_s"=>fwd_t + pb_t,
                "dtheta_rel_err"=>Float64(dθ_err),
                "xstar_rel_err"=>Float64(Xstar_err),
            )
            println(out_io, jsonl_record(row)); flush(out_io)
            @printf("%-15s %-5s %-10s | %12.4f %12.4f | %14.2e %12.2e\n",
                    case, string(T), "batched", fwd_t, pb_t, dθ_err, Xstar_err)
        end

        # --- batched GPU (only F32 with Metal) ---
        if GPU_BACKEND !== nothing
            metal_no_f64 = (GPU_BACKEND.name == "Metal" && T === Float64)
            if metal_no_f64
                row = Dict(
                    "kind"=>"measurement", "n"=>n, "B"=>B, "T"=>string(T),
                    "condition"=>"batched_$(lowercase(GPU_BACKEND.name))",
                    "skipped"=>true, "skip_reason"=>"Metal does not support Float64",
                )
                println(out_io, jsonl_record(row)); flush(out_io)
                @printf("%-15s %-5s %-10s | %12s %12s | %14s %12s\n",
                        case, string(T), lowercase(GPU_BACKEND.name)[1:min(6,end)],
                        "—", "—", "—", "—")
            else
                r = try
                    timed_with_warmup(samples=2) do
                        run_batched_gpu(prob, n, B)
                    end
                catch e
                    @warn "GPU run failed for $case $T" exception=(e, catch_backtrace())
                    nothing
                end
                if r !== nothing
                    Xstar, dθ, fwd_t, pb_t = r.result
                    dθ_err = norm(Vector(dθ) .- dθ_closed) / max(one(T), norm(dθ_closed))
                    Xstar_err = norm(Xstar .- Xstar_closed) / max(one(T), norm(Xstar_closed))
                    row = Dict(
                        "kind"=>"measurement", "n"=>n, "B"=>B, "T"=>string(T),
                        "condition"=>"batched_$(lowercase(GPU_BACKEND.name))",
                        "forward_s"=>fwd_t, "pullback_s"=>pb_t,
                        "total_s"=>fwd_t + pb_t,
                        "dtheta_rel_err"=>Float64(dθ_err),
                        "xstar_rel_err"=>Float64(Xstar_err),
                    )
                    println(out_io, jsonl_record(row)); flush(out_io)
                    @printf("%-15s %-5s %-10s | %12.4f %12.4f | %14.2e %12.2e\n",
                            case, string(T), lowercase(GPU_BACKEND.name)[1:min(6,end)],
                            fwd_t, pb_t, dθ_err, Xstar_err)
                end
            end
        end
    end

    println("\n", "="^120)
end

# ── Main ─────────────────────────────────────────────────────────────

function main()
    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)
    config_tag = USE_ACCEL ? "accelerate" : "openblas"
    gpu_tag = GPU_BACKEND === nothing ? "nogpu" : lowercase(GPU_BACKEND.name)
    ts = Dates.format(now(), "yyyymmdd-HHMMSS")
    out_path = joinpath(results_dir,
        "bench_batched_bilevel_$(GRID)_$(config_tag)_$(gpu_tag)_$(ts).jsonl")
    open(out_path, "w") do io
        run_sweep(io)
    end
    println("\nWrote: $out_path")
end

main()
