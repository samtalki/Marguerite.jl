module LibMarguerite

using Marguerite: solve, Simplex, ProbSimplex, Box, Cache, MonotonicStepSize, _cg_solve
using LinearAlgebra: dot

# ── C-compatible result structs ────────────────────────────────────

struct CResult
    objective::Cdouble
    gap::Cdouble
    iterations::Cint
    converged::Cint   # bool as int32
    discards::Cint
    status::Cint      # 0 = ok, -1 = error
end

struct CBilevelResult
    inner_objective::Cdouble
    inner_gap::Cdouble
    inner_iterations::Cint
    inner_converged::Cint
    inner_discards::Cint
    cg_iterations::Cint
    cg_residual::Cdouble
    cg_converged::Cint
    status::Cint
end

# ── Internal helpers ───────────────────────────────────────────────

function _solve_and_copy!(f, ∇f!, lmo, x0, x_out_ptr, n, max_iters, tol, monotonic)
    try
        x, res = solve(f, ∇f!, lmo, x0;
                       max_iters=Int(max_iters),
                       tol=Float64(tol),
                       monotonic=monotonic != 0)
        unsafe_copyto!(x_out_ptr, pointer(x), n)
        return CResult(
            res.objective,
            res.gap,
            Cint(res.iterations),
            Cint(res.converged),
            Cint(res.discards),
            Cint(0),
        )
    catch e
        @error "marg_solve: Julia exception" exception=(e, catch_backtrace())
        return CResult(NaN, NaN, Cint(0), Cint(0), Cint(0), Cint(-1))
    end
end

# ── Callback wrappers ─────────────────────────────────────────────

function _wrap_obj(f_ptr, n, userdata)
    return x -> ccall(f_ptr, Cdouble, (Ptr{Cdouble}, Cint, Ptr{Cvoid}), x, n, userdata)
end

function _wrap_grad(grad_ptr, n, userdata)
    return function(g, x)
        ccall(grad_ptr, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cvoid}), g, x, n, userdata)
        return g
    end
end

function _wrap_lmo(lmo_ptr, n, userdata)
    return function(v, g)
        ccall(lmo_ptr, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cvoid}), v, g, n, userdata)
        return v
    end
end

# ── Generic solve (user-supplied f, grad, lmo callbacks) ────────────

Base.@ccallable function marg_solve(
    f_ptr::Ptr{Cvoid},
    grad_ptr::Ptr{Cvoid},
    lmo_ptr::Ptr{Cvoid},
    x0_ptr::Ptr{Cdouble},
    x_out_ptr::Ptr{Cdouble},
    n::Cint,
    max_iters::Cint,
    tol::Cdouble,
    monotonic::Cint,
    userdata::Ptr{Cvoid},
)::CResult
    x0 = unsafe_wrap(Array, x0_ptr, Int(n))
    f = _wrap_obj(f_ptr, n, userdata)
    ∇f! = _wrap_grad(grad_ptr, n, userdata)
    lmo = _wrap_lmo(lmo_ptr, n, userdata)
    return _solve_and_copy!(f, ∇f!, lmo, x0, x_out_ptr, Int(n), max_iters, tol, monotonic)
end

# ── Simplex convenience wrapper ─────────────────────────────────────

Base.@ccallable function marg_solve_simplex(
    f_ptr::Ptr{Cvoid},
    grad_ptr::Ptr{Cvoid},
    x0_ptr::Ptr{Cdouble},
    x_out_ptr::Ptr{Cdouble},
    n::Cint,
    radius::Cdouble,
    max_iters::Cint,
    tol::Cdouble,
    monotonic::Cint,
    userdata::Ptr{Cvoid},
)::CResult
    x0 = unsafe_wrap(Array, x0_ptr, Int(n))
    lmo = Simplex(Float64(radius))
    f = _wrap_obj(f_ptr, n, userdata)
    ∇f! = _wrap_grad(grad_ptr, n, userdata)
    return _solve_and_copy!(f, ∇f!, lmo, x0, x_out_ptr, Int(n), max_iters, tol, monotonic)
end

# ── Probability simplex convenience wrapper ──────────────────────────

Base.@ccallable function marg_solve_prob_simplex(
    f_ptr::Ptr{Cvoid},
    grad_ptr::Ptr{Cvoid},
    x0_ptr::Ptr{Cdouble},
    x_out_ptr::Ptr{Cdouble},
    n::Cint,
    radius::Cdouble,
    max_iters::Cint,
    tol::Cdouble,
    monotonic::Cint,
    userdata::Ptr{Cvoid},
)::CResult
    x0 = unsafe_wrap(Array, x0_ptr, Int(n))
    lmo = ProbSimplex(Float64(radius))
    f = _wrap_obj(f_ptr, n, userdata)
    ∇f! = _wrap_grad(grad_ptr, n, userdata)
    return _solve_and_copy!(f, ∇f!, lmo, x0, x_out_ptr, Int(n), max_iters, tol, monotonic)
end

# ── Box convenience wrapper ──────────────────────────────────────────

Base.@ccallable function marg_solve_box(
    f_ptr::Ptr{Cvoid},
    grad_ptr::Ptr{Cvoid},
    x0_ptr::Ptr{Cdouble},
    x_out_ptr::Ptr{Cdouble},
    n::Cint,
    lb_ptr::Ptr{Cdouble},
    ub_ptr::Ptr{Cdouble},
    max_iters::Cint,
    tol::Cdouble,
    monotonic::Cint,
    userdata::Ptr{Cvoid},
)::CResult
    x0 = unsafe_wrap(Array, x0_ptr, Int(n))
    lb = unsafe_wrap(Array, lb_ptr, Int(n))
    ub = unsafe_wrap(Array, ub_ptr, Int(n))
    lmo = Box(lb, ub)
    f = _wrap_obj(f_ptr, n, userdata)
    ∇f! = _wrap_grad(grad_ptr, n, userdata)
    return _solve_and_copy!(f, ∇f!, lmo, x0, x_out_ptr, Int(n), max_iters, tol, monotonic)
end

# ── Bilevel solve ────────────────────────────────────────────────────

Base.@ccallable function marg_bilevel_solve(
    inner_obj_ptr::Ptr{Cvoid},
    inner_grad_ptr::Ptr{Cvoid},
    lmo_ptr::Ptr{Cvoid},
    outer_obj_ptr::Ptr{Cvoid},
    outer_grad_ptr::Ptr{Cvoid},
    hvp_ptr::Ptr{Cvoid},
    cross_vjp_ptr::Ptr{Cvoid},
    x0_ptr::Ptr{Cdouble},
    x_out_ptr::Ptr{Cdouble},
    n::Cint,
    theta_ptr::Ptr{Cdouble},
    theta_grad_out_ptr::Ptr{Cdouble},
    ntheta::Cint,
    max_iters::Cint,
    tol::Cdouble,
    monotonic::Cint,
    cg_maxiter::Cint,
    cg_tol::Cdouble,
    cg_lambda::Cdouble,
    userdata::Ptr{Cvoid},
)::CBilevelResult
    try
        nn = Int(n)
        nt = Int(ntheta)
        x0 = unsafe_wrap(Array, x0_ptr, nn)
        θ = unsafe_wrap(Array, theta_ptr, nt)

        # wrap callbacks
        f(x) = ccall(inner_obj_ptr, Cdouble,
                      (Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Ptr{Cvoid}),
                      x, θ, n, ntheta, userdata)

        function ∇f!(g, x)
            ccall(inner_grad_ptr, Cvoid,
                  (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Ptr{Cvoid}),
                  g, x, θ, n, ntheta, userdata)
            return g
        end

        lmo = _wrap_lmo(lmo_ptr, n, userdata)

        # 1. solve inner FW problem
        x_star, inner_res = solve(f, ∇f!, lmo, x0;
                                  max_iters=Int(max_iters),
                                  tol=Float64(tol),
                                  monotonic=monotonic != 0)

        # 2. compute outer gradient x̄ = ∇_x L(x*)
        x̄ = Vector{Float64}(undef, nn)
        ccall(outer_grad_ptr, Cvoid,
              (Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cvoid}),
              x̄, x_star, n, userdata)

        # 3. CG solve (H + λI)u = x̄ using HVP callback
        function hvp_fn(p)
            Hp = Vector{Float64}(undef, nn)
            ccall(hvp_ptr, Cvoid,
                  (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Ptr{Cvoid}),
                  Hp, x_star, p, θ, n, ntheta, userdata)
            return Hp
        end
        u, cg_res = _cg_solve(hvp_fn, x̄;
                               maxiter=Int(cg_maxiter),
                               tol=Float64(cg_tol),
                               λ=Float64(cg_lambda))

        # 4. compute θ̄ = -(∂∇_x f/∂θ)^T u via cross_vjp callback, then negate
        θ̄ = Vector{Float64}(undef, nt)
        ccall(cross_vjp_ptr, Cvoid,
              (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Ptr{Cvoid}),
              θ̄, x_star, u, θ, n, ntheta, userdata)
        θ̄ .*= -1

        # copy outputs
        unsafe_copyto!(x_out_ptr, pointer(x_star), nn)
        unsafe_copyto!(theta_grad_out_ptr, pointer(θ̄), nt)

        return CBilevelResult(
            inner_res.objective, inner_res.gap,
            Cint(inner_res.iterations), Cint(inner_res.converged), Cint(inner_res.discards),
            Cint(cg_res.iterations), cg_res.residual_norm, Cint(cg_res.converged),
            Cint(0),
        )
    catch e
        @error "marg_bilevel_solve: Julia exception" exception=(e, catch_backtrace())
        return CBilevelResult(
            NaN, NaN, Cint(0), Cint(0), Cint(0),
            Cint(0), NaN, Cint(0),
            Cint(-1),
        )
    end
end

end # module LibMarguerite
