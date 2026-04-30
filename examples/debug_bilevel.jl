using Marguerite
using Marguerite: solve, bilevel_solve, ProbSimplex
using LinearAlgebra
using Random: Xoshiro
using Printf

T = Float64
n = 4

rng = Xoshiro(42)
θ = 0.05 * randn(rng, T, n)
c = 0.05 * randn(rng, T, n)
x_target = max.(zero(T), 0.05 * randn(rng, T, n) .+ 1/n)
x_target ./= sum(x_target)

# Closed-form interior optimum
v = θ .+ c
m = sum(v) / n
x_closed = v .- m .+ 1/n
println("θ + c = ", v)
println("interior x* (closed form) = ", x_closed)
println("min x* = ", minimum(x_closed))
println("sum(x*) = ", sum(x_closed))

inner = (x, θ_) -> 0.5 * dot(x, x) - dot(θ_ .+ c, x)
grad! = (g, x, θ_) -> (g .= x .- θ_ .- c)
outer = x -> 0.5 * sum((x .- x_target).^2)

x0 = fill(1.0/n, n)
println("\n--- forward solve ---")
x_solver, res = solve(inner, ProbSimplex(1.0), copy(x0), θ;
                      grad=grad!, max_iters=10000, tol=1e-10)
println("x_solver = ", x_solver)
println("res.iters = ", res.iterations, ", gap = ", res.gap, ", converged = ", res.converged)
println("err = ", norm(x_solver .- x_closed))

println("\n--- bilevel_solve ---")
x_b, dθ_b, _ = bilevel_solve(outer, inner, ProbSimplex(1.0), copy(x0), θ;
                              grad=grad!, max_iters=10000, tol=1e-10)
println("dθ from bilevel_solve = ", dθ_b)

# Closed form bilevel gradient
diff = x_closed .- x_target
col_mean = sum(diff) / n
dθ_closed = diff .- col_mean
println("dθ closed form = ", dθ_closed)
println("dθ rel err = ", norm(dθ_b .- dθ_closed) / norm(dθ_closed))
