module LibMarguerite

using Marguerite: solve, Simplex, ProbSimplex, Box, Cache, MonotonicStepSize

# ── C-compatible result struct ──────────────────────────────────────

struct CResult
    objective::Cdouble
    gap::Cdouble
    iterations::Cint
    converged::Cint   # bool as int32
    discards::Cint
end

# ── Internal helpers ────────────────────────────────────────────────

function _solve_and_copy!(f, ∇f!, lmo, x0, x_out_ptr, n, max_iters, tol, monotonic)
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
    )
end

# ── Generic solve (user-supplied f, grad, lmo callbacks) ────────────

Base.@ccallable function marguerite_solve(
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

    f(x) = ccall(f_ptr, Cdouble, (Ptr{Cdouble}, Cint, Ptr{Cvoid}), x, n, userdata)

    function ∇f!(g, x)
        ccall(grad_ptr, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cvoid}), g, x, n, userdata)
        return g
    end

    function lmo(v, g)
        ccall(lmo_ptr, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cvoid}), v, g, n, userdata)
        return v
    end

    return _solve_and_copy!(f, ∇f!, lmo, x0, x_out_ptr, Int(n), max_iters, tol, monotonic)
end

# ── Simplex convenience wrapper ─────────────────────────────────────

Base.@ccallable function marguerite_solve_simplex(
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

    f(x) = ccall(f_ptr, Cdouble, (Ptr{Cdouble}, Cint, Ptr{Cvoid}), x, n, userdata)

    function ∇f!(g, x)
        ccall(grad_ptr, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cvoid}), g, x, n, userdata)
        return g
    end

    return _solve_and_copy!(f, ∇f!, lmo, x0, x_out_ptr, Int(n), max_iters, tol, monotonic)
end

# ── Probability simplex convenience wrapper ──────────────────────────

Base.@ccallable function marguerite_solve_prob_simplex(
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

    f(x) = ccall(f_ptr, Cdouble, (Ptr{Cdouble}, Cint, Ptr{Cvoid}), x, n, userdata)

    function ∇f!(g, x)
        ccall(grad_ptr, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cvoid}), g, x, n, userdata)
        return g
    end

    return _solve_and_copy!(f, ∇f!, lmo, x0, x_out_ptr, Int(n), max_iters, tol, monotonic)
end

# ── Box convenience wrapper ──────────────────────────────────────────

Base.@ccallable function marguerite_solve_box(
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

    f(x) = ccall(f_ptr, Cdouble, (Ptr{Cdouble}, Cint, Ptr{Cvoid}), x, n, userdata)

    function ∇f!(g, x)
        ccall(grad_ptr, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cvoid}), g, x, n, userdata)
        return g
    end

    return _solve_and_copy!(f, ∇f!, lmo, x0, x_out_ptr, Int(n), max_iters, tol, monotonic)
end

end # module LibMarguerite
