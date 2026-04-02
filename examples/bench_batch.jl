# Batched Frank-Wolfe Benchmark
#
# Compares sequential solve() in a loop vs batch_solve() for various
# oracle types and problem sizes.
#
# Usage:
#   julia --project=. examples/bench_batch.jl
#
# On Apple Silicon with Metal.jl:
#   julia --project=. examples/bench_batch.jl --metal

using Marguerite
using LinearAlgebra: dot, I
using BenchmarkTools
using Random: Xoshiro
using Printf

const USE_METAL = "--metal" in ARGS

if USE_METAL
    using Metal
    @info "Metal.jl loaded — GPU benchmarks enabled"
end

# ── Problem generators ──────────────────────────────────────────────

function make_batch_qp(n, B; seed=42)
    rng = Xoshiro(seed)
    A = randn(rng, n, n)
    Q = A'A + 0.1I
    C = randn(rng, n, B)   # per-problem linear terms
    X0 = rand(rng, n, B)
    return Q, C, X0
end

# ── Sequential baseline ────────────────────────────────────────────

function bench_sequential(Q, C, X0, lmo; max_iters=2000, tol=1e-4)
    n, B = size(X0)
    for b in 1:B
        c_b = C[:, b]
        fb(x) = 0.5 * dot(x, Q * x) + dot(c_b, x)
        gradb!(g, x) = (g .= Q * x .+ c_b)
        solve(fb, lmo, X0[:, b]; grad=gradb!, max_iters=max_iters, tol=tol)
    end
end

# ── Batched ─────────────────────────────────────────────────────────

struct BatchQP{T, MQ<:AbstractMatrix{T}, MC<:AbstractMatrix{T}}
    Q::MQ
    C::MC
    buf::Matrix{T}   # (n, B) workspace for Q*X
end

function batch_obj(bqp::BatchQP{T}, X) where T
    mul!(bqp.buf, bqp.Q, X)
    B = size(X, 2)
    obj = Vector{T}(undef, B)
    @inbounds for b in 1:B
        s = zero(T)
        @simd for i in 1:size(X, 1)
            s += X[i, b] * (0.5 * bqp.buf[i, b] + bqp.C[i, b])
        end
        obj[b] = s
    end
    return obj
end

function batch_grad!(bqp::BatchQP, G, X)
    mul!(G, bqp.Q, X)
    G .+= bqp.C
end

function bench_batched(bqp, lmo, X0; max_iters=2000, tol=1e-4)
    batch_solve(X -> batch_obj(bqp, X), lmo, X0;
                grad_batch=(G, X) -> batch_grad!(bqp, G, X),
                max_iters=max_iters, tol=tol)
end

# ── Scenarios ───────────────────────────────────────────────────────

using LinearAlgebra: mul!

scenarios = [
    ("ScalarBox n=100 B=64",    100,  64,  Box(0.0, 1.0)),
    ("ScalarBox n=500 B=256",   500,  256, Box(0.0, 1.0)),
    ("ScalarBox n=1000 B=512",  1000, 512, Box(0.0, 1.0)),
    ("ProbSimplex n=100 B=64",  100,  64,  ProbSimplex()),
    ("ProbSimplex n=100 B=256", 100,  256, ProbSimplex()),
    ("ProbSimplex n=500 B=256", 500,  256, ProbSimplex()),
]

function normalize_to_simplex!(X0)
    n, B = size(X0)
    for b in 1:B
        s = sum(@view(X0[:, b]))
        if s > 0
            @view(X0[:, b]) ./= s
        else
            X0[:, b] .= 1.0 / n
        end
    end
    X0
end

println("\n", "="^72)
println("  Marguerite.jl — Batched Frank-Wolfe Benchmark")
println("="^72)

for (name, n, B, lmo) in scenarios
    Q, C, X0 = make_batch_qp(n, B)
    if lmo isa Simplex
        normalize_to_simplex!(X0)
    end
    bqp = BatchQP(Q, C, similar(Q * X0))

    # Warmup
    bench_sequential(Q, C, X0[:, 1:min(2, B)], lmo; max_iters=5)
    bench_batched(bqp, lmo, X0[:, 1:min(2, B)]; max_iters=5)

    println("\n  ", name)
    println("  ", "-"^60)

    t_seq = @belapsed bench_sequential($Q, $C, $X0, $lmo) evals=1 samples=3
    t_bat = @belapsed bench_batched($bqp, $lmo, $X0) evals=1 samples=3
    speedup = t_seq / t_bat

    @printf("    Sequential:  %8.2f ms\n", t_seq * 1000)
    @printf("    Batched:     %8.2f ms\n", t_bat * 1000)
    @printf("    Speedup:     %8.2fx\n", speedup)
end

println("\n", "="^72)
println("  Done.")
println("="^72)
