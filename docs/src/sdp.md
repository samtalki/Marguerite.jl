# SDP Relaxation

## Quantum Density-Matrix Optimization

[`Spectraplex(n)`](@ref) exactly represents the real density-matrix feasible set

```math
\{ \rho \in \mathbb{S}_+^n : \operatorname{tr}(\rho) = 1 \}.
```

This page uses that set directly: optimize the energy of a real symmetric
Hamiltonian over density matrices. Problems such as the standard MaxCut SDP add
extra constraints like ``\operatorname{diag}(X) = 1`` and therefore require a
richer oracle than `Spectraplex` alone.

For simplicity we work with real symmetric matrices. The package stores the
matrix variable in vectorized form, so `x = vec(rho)`.

## Random Hamiltonian

```@example sdp
using Marguerite, LinearAlgebra, Random, UnicodePlots

Random.seed!(42)
n = 20
A = randn(n, n)
H = Symmetric((A + A') / 2)
H_vec = vec(Matrix(H))

println("Hilbert-space dimension = $n")
println("Smallest eigenvalue of H = ", round(eigmin(H); sigdigits=6))
nothing  # hide
```

## Linear Ground-State SDP

The ground-state problem is the linear SDP

```math
\min_{\rho \succeq 0,\; \operatorname{tr}(\rho)=1} \operatorname{tr}(H \rho).
```

Its optimum is the minimum eigenvalue of ``H``, attained by the rank-1 projector
onto a ground-state eigenvector. Frank-Wolfe finds that projector in one step.

```@example sdp
m = n * n
lmo = Spectraplex(n)

rho0 = Matrix{Float64}(I, n, n) / n
x0 = vec(rho0)

f_linear(x) = dot(H_vec, x)
∇f_linear!(g, x) = (g .= H_vec)

x_lin, res_lin = solve(f_linear, lmo, x0;
                       grad=∇f_linear!, max_iters=100, tol=1e-10, verbose=false)

println("FW energy            = ", round(res_lin.objective; sigdigits=6))
println("Ground-state energy  = ", round(eigmin(H); sigdigits=6))
println("FW gap               = ", round(res_lin.gap; sigdigits=4))
println("Iterations           = ", res_lin.iterations)
nothing  # hide
```

## Regularized Mixed-State Problem

Adding a Frobenius penalty produces a nontrivial convex SDP whose solution is
typically mixed rather than rank 1:

```math
\min_{\rho \succeq 0,\; \operatorname{tr}(\rho)=1}
\operatorname{tr}(H \rho) + \frac{\eta}{2}\|\rho\|_F^2.
```

```@example sdp
eta = 1.5

f_reg(x) = dot(H_vec, x) + (eta / 2) * dot(x, x)
∇f_reg!(g, x) = (g .= H_vec .+ eta .* x)

x_reg, res_reg = solve(f_reg, lmo, x0;
                       grad=∇f_reg!, max_iters=1000, tol=1e-8, verbose=false)

println("Objective  = ", round(res_reg.objective; sigdigits=6))
println("FW gap     = ", round(res_reg.gap; sigdigits=4))
println("Iterations = ", res_reg.iterations)
nothing  # hide
```

## Convergence Trace

The hand-written Frank-Wolfe loop below records the duality gap for the
regularized problem:

```@example sdp
max_iters = 500
x = copy(x0)
g_buf = zeros(m)
v_buf = zeros(m)
step = MonotonicStepSize()
gaps = Vector{Float64}(undef, max_iters)

for t in 0:max_iters-1
    ∇f_reg!(g_buf, x)
    lmo(v_buf, g_buf)
    gaps[t+1] = dot(g_buf, x .- v_buf)
    gamma = step(t)
    x .= x .+ gamma .* (v_buf .- x)
end

lineplot(1:max_iters, gaps;
         yscale=:log10,
         title="FW Gap — Regularized Density-Matrix SDP (n=$n, eta=$eta)",
         xlabel="iteration", ylabel="gap",
         name="FW gap", width=60)
```

## Solution Structure

The regularized solution is still PSD with unit trace, but it can spread mass
across several eigenvectors.

```@example sdp
rho_sol = reshape(x_reg, n, n)
rho_sym = Symmetric((rho_sol .+ rho_sol') ./ 2)
lambda = eigvals(rho_sym)
lambda_top = sort(lambda; rev=true)[1:min(10, n)]

println("Trace        = ", round(tr(rho_sym); sigdigits=6))
println("Min eig      = ", round(minimum(lambda); sigdigits=6))
println("Purity       = ", round(dot(x_reg, x_reg); sigdigits=6))
println("Numerical rank = ", count(>(1e-6), lambda), " / ", n)

barplot(["λ$i" for i in 1:length(lambda_top)], lambda_top;
        title="Largest eigenvalues of rho*",
        xlabel="value", width=60)
```

`Spectraplex` also supports implicit differentiation through these solves via a
compact [`active_set`](@ref) representation; see
[Bilevel Optimization](@ref) for the differentiated interface.

## Why Frank-Wolfe Fits This SDP

Frank-Wolfe is effective for density-matrix SDPs because:

1. **Cheap oracle**: the linear minimization oracle only needs the minimum
   eigenvector of the current gradient matrix.
2. **Low-rank iterates**: each FW step adds one rank-1 projector, so early
   iterates stay compact.
3. **No projections**: projection onto the spectraplex needs a full
   eigendecomposition, while the FW step only solves a leading-eigenvector
   subproblem.
4. **Memory efficiency**: the method stores a few dense matrices and vectors,
   not a full SDP interior-point system.
