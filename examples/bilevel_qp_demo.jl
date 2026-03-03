using Marguerite, LinearAlgebra

H = [4.0 1.0 0.5; 1.0 3.0 0.8; 0.5 0.8 2.0]
f(x, θ) = 0.5 * dot(x, H * x) - dot(θ, x)
∇f!(g, x, θ) = (g .= H * x .- θ)

x_target = [0.6, 0.3, 0.1]
outer_loss(x) = sum((x .- x_target) .^ 2)
x0, θ = fill(1/3, 3), H * x_target

println("Starting bilevel optimization...")
t0 = time()

for k in 1:100
    x, ∇θ, cg = bilevel_solve(outer_loss, f, ∇f!,
                    ProbSimplex(), x0, θ; tol=1e-8, max_iters=10000)
    θ .-= 0.1 .* ∇θ
    if k % 10 == 0 || k == 1
        loss = outer_loss(x)
        elapsed = round(time() - t0; digits=1)
        println("  k=$k  loss=$(round(loss; sigdigits=3))  cg_iters=$(cg.iterations)  elapsed=$(elapsed)s")
    end
end

x_final, _ = solve(f, ∇f!, ProbSimplex(), x0, θ)
println("\nx*(θ)  = ", round.(x_final; digits=3))
println("target = ", x_target)
println("Total: ", round(time() - t0; digits=1), "s")
