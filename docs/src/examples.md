<!-- Copyright 2026 Samuel Talkington and contributors
   SPDX-License-Identifier: Apache-2.0 -->

# Examples

## Sparse Recovery on the Simplex

Consider sparse recovery on the probability simplex:

```math
\min_{x \in \Delta_n} \;\tfrac{1}{2} \| A x - b \|^2
```

where ``A \in \mathbb{R}^{m \times n}`` with ``m \ll n``, and the ground truth
``x^*`` has only a few nonzero entries.

Interior-point methods must form the Hessian ``Q = A^\top A \in \mathbb{R}^{n \times n}``
— that's ``O(n^2)`` storage and ``O(n^3)`` factorization per iteration.
Frank-Wolfe never touches ``Q``: each iteration costs ``O(mn)`` for the gradient
and ``O(n)`` for the simplex LMO (just `argmin`). The iterates are vertex-sparse
by construction, so the solution is sparse for free.

## Zero-allocation callable structs

We define objective and gradient as callable structs with pre-allocated workspace
buffers, then wrap them in closures for `solve`'s `::Function` dispatch:

```@example examples
using Marguerite, LinearAlgebra, Random, UnicodePlots

struct LeastSquaresObj{M<:AbstractMatrix, V<:AbstractVector}
    A::M
    b::V
    r::V  # workspace: residual A*x - b
end

LeastSquaresObj(A::Matrix{Float64}, b::Vector{Float64}) =
    LeastSquaresObj(A, b, similar(b))

function (obj::LeastSquaresObj)(x)
    mul!(obj.r, obj.A, x)
    obj.r .-= obj.b
    return 0.5 * dot(obj.r, obj.r)
end

struct LeastSquaresGrad!{M<:AbstractMatrix, V<:AbstractVector}
    A::M
    b::V
    r::V
end

LeastSquaresGrad!(A::Matrix{Float64}, b::Vector{Float64}) =
    LeastSquaresGrad!(A, b, similar(b))

function (∇f::LeastSquaresGrad!)(g, x)
    mul!(∇f.r, ∇f.A, x)
    ∇f.r .-= ∇f.b
    mul!(g, ∇f.A', ∇f.r)  # g = A'(Ax - b)
    return g
end

nothing  # hide
```

!!! note "Why closures?"
    Marguerite's `solve` dispatches on `∇f!::Function` to distinguish the
    4-argument form `solve(f, ∇f!, lmo, x0)` from the auto-diff form
    `solve(f, lmo, x0)`. Callable structs aren't `<: Function`, so we wrap
    them: `f = x -> f_obj(x)`.

## Problem setup

```@example examples
Random.seed!(42)

m, n = 50, 5000
A = randn(m, n)

# sparse ground truth: 5 active components on the simplex
x_true = zeros(n)
support = sort(randperm(n)[1:5])
x_true[support] .= 0.2
b = A * x_true

# wrap callable structs in closures for ::Function dispatch
f_obj = LeastSquaresObj(A, b)
∇f_obj = LeastSquaresGrad!(A, b)
f = x -> f_obj(x)
∇f! = (g, x) -> ∇f_obj(g, x)

lmo = ProbabilitySimplex()
# start from vertex e₁ so FW iterates stay sparse
x0 = zeros(n); x0[1] = 1.0

nothing  # hide
```

## Convergence trace

We trace the FW gap over iterations to see the ``O(1/t)`` decay, then report
the final solution quality:

```@example examples
# hand-written FW loop to collect gap history
x = copy(x0)
g_buf = zeros(n); v_buf = zeros(n); d_buf = zeros(n)
step = MonotonicStepSize()
max_iters = 2000
gaps = Vector{Float64}(undef, max_iters)

for t in 0:max_iters-1
    ∇f!(g_buf, x)
    lmo(v_buf, g_buf)
    d_buf .= x .- v_buf
    gaps[t+1] = dot(g_buf, d_buf)
    γ = step(t)
    x .= x .+ γ .* (v_buf .- x)
end

nnz_x = count(>(1e-8), x)
println("objective      = ", round(f(x); sigdigits=4))
println("FW gap         = ", round(gaps[end]; sigdigits=4))
println("nnz(x)         = ", nnz_x, " / ", n)
println("recovery error = ", round(norm(x - x_true); sigdigits=4))

lineplot(1:max_iters, gaps;
         yscale=:log10,
         title="FW Duality Gap  (m=$m, n=$n)",
         xlabel="iteration", ylabel="gap",
         name="FW gap", width=60)
```

## Sparsity visualization

```@example examples
idx = findall(>(1e-8), x)
barplot(string.(idx), x[idx];
        title="Nonzero components of x  ($(length(idx)) / $n)",
        xlabel="weight", width=60)
```

## Benchmark results

The table below compares Marguerite (Frank-Wolfe) against Clarabel (interior-point
QP solver via JuMP) on increasingly large problems. Results from
[`examples/bench_fw_advantage.jl`](https://github.com/samtalki/Marguerite.jl/blob/main/examples/bench_fw_advantage.jl).

| Problem Size | Marguerite FW | Clarabel QP |
|:-------------|:--------------|:------------|
| m=50, n=5,000 | **97 ms**, 157 KiB | 488 s, 23.5 GiB |
| m=100, n=20,000 | **3.8 s**, 625 KiB | prohibitive |
| m=100, n=50,000 | **15.5 s**, 1.5 MiB | prohibitive |

On the smallest problem, the Frank-Wolfe solver is approximately 5000× faster
and scales to dimensions where interior-point methods cannot even allocate the Hessian.

## Why the difference

Three factors drive the gap:

1. **Memory**: FW never forms ``A^\top A``. It keeps ``O(n)`` buffers via
   `Cache` — a few vectors vs. an ``n \times n`` dense matrix.

2. **Per-iteration cost**: Each FW step is an ``O(mn)`` matrix-vector product
   plus an ``O(n)`` LMO. Interior-point requires ``O(n^3)`` matrix factorizations.

3. **Sparsity**: FW iterates have at most ``t + 1`` nonzeros after ``t`` steps.
   IPM solutions are dense — every coordinate gets a small nonzero value from
   the barrier term.

## When to use Frank-Wolfe

**Frank-Wolfe is a good fit when:**
- ``n`` is large and the LMO is cheap (simplex, ``\ell_1``-ball, nuclear norm ball)
- You want sparse or structured solutions
- Memory is constrained — FW uses ``O(n)`` working memory
- Moderate accuracy suffices (``10^{-4}``–``10^{-6}`` primal gap)

**Interior-point methods may be better when:**
- High accuracy is critical (``10^{-8}``+)
- ``n`` is small enough that ``Q \in \mathbb{R}^{n \times n}`` fits in memory
- The constraint set doesn't have a cheap LMO
