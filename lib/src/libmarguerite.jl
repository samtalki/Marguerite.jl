module LibMarguerite

using Marguerite: solve, Simplex, ProbSimplex, Box, MonotonicStepSize, AdaptiveStepSize, _cg_solve

# ── C-compatible result structs ────────────────────────────────────

struct CResult
    objective::Cdouble
    gap::Cdouble
    iterations::Cint
    converged::Cint   # bool as int32
    discards::Cint
    status::Cint      # MARG_OK = 0, MARG_ERROR = -1
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

const _ERROR_RESULT = CResult(NaN, NaN, Cint(0), Cint(0), Cint(0), Cint(-1))
const _ERROR_BILEVEL_RESULT = CBilevelResult(NaN, NaN, Cint(0), Cint(0), Cint(0), Cint(0), NaN, Cint(0), Cint(-1))

# ── Internal helpers ───────────────────────────────────────────────

function _to_cresult(x, res, x_out_ptr, n)
    GC.@preserve x unsafe_copyto!(x_out_ptr, pointer(x), n)
    return CResult(res.objective, res.gap,
                   Cint(res.iterations), Cint(res.converged), Cint(res.discards),
                   Cint(0))
end

function _solve_and_copy!(f, ∇f!, lmo, x0, x_out_ptr, n, max_iters, tol, step_rule_flag, L0)
    try
        mi = Int(max_iters)
        t = Float64(tol)
        if step_rule_flag == Cint(1)
            x, res = solve(f, ∇f!, lmo, x0; max_iters=mi, tol=t,
                           step_rule=AdaptiveStepSize(Float64(L0)))
        else
            x, res = solve(f, ∇f!, lmo, x0; max_iters=mi, tol=t)
        end
        return _to_cresult(x, res, x_out_ptr, n)
    catch e
        @error "marg_solve: Julia exception" exception=(e, catch_backtrace())
        return _ERROR_RESULT
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
    step_rule::Cint,
    L0::Cdouble,
    userdata::Ptr{Cvoid},
)::CResult
    (n <= 0 || any(==(C_NULL), (f_ptr, grad_ptr, lmo_ptr, x0_ptr, x_out_ptr))) && return _ERROR_RESULT
    x0 = unsafe_wrap(Array, x0_ptr, Int(n))
    f = _wrap_obj(f_ptr, n, userdata)
    ∇f! = _wrap_grad(grad_ptr, n, userdata)
    lmo = _wrap_lmo(lmo_ptr, n, userdata)
    return _solve_and_copy!(f, ∇f!, lmo, x0, x_out_ptr, Int(n), max_iters, tol,
                            step_rule, L0)
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
    step_rule::Cint,
    L0::Cdouble,
    userdata::Ptr{Cvoid},
)::CResult
    (n <= 0 || any(==(C_NULL), (f_ptr, grad_ptr, x0_ptr, x_out_ptr))) && return _ERROR_RESULT
    x0 = unsafe_wrap(Array, x0_ptr, Int(n))
    lmo = Simplex(Float64(radius))
    f = _wrap_obj(f_ptr, n, userdata)
    ∇f! = _wrap_grad(grad_ptr, n, userdata)
    return _solve_and_copy!(f, ∇f!, lmo, x0, x_out_ptr, Int(n), max_iters, tol,
                            step_rule, L0)
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
    step_rule::Cint,
    L0::Cdouble,
    userdata::Ptr{Cvoid},
)::CResult
    (n <= 0 || any(==(C_NULL), (f_ptr, grad_ptr, x0_ptr, x_out_ptr))) && return _ERROR_RESULT
    x0 = unsafe_wrap(Array, x0_ptr, Int(n))
    lmo = ProbSimplex(Float64(radius))
    f = _wrap_obj(f_ptr, n, userdata)
    ∇f! = _wrap_grad(grad_ptr, n, userdata)
    return _solve_and_copy!(f, ∇f!, lmo, x0, x_out_ptr, Int(n), max_iters, tol,
                            step_rule, L0)
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
    step_rule::Cint,
    L0::Cdouble,
    userdata::Ptr{Cvoid},
)::CResult
    (n <= 0 || any(==(C_NULL), (f_ptr, grad_ptr, x0_ptr, x_out_ptr, lb_ptr, ub_ptr))) && return _ERROR_RESULT
    x0 = unsafe_wrap(Array, x0_ptr, Int(n))
    lb = unsafe_wrap(Array, lb_ptr, Int(n))
    ub = unsafe_wrap(Array, ub_ptr, Int(n))
    lmo = Box(lb, ub)
    f = _wrap_obj(f_ptr, n, userdata)
    ∇f! = _wrap_grad(grad_ptr, n, userdata)
    return _solve_and_copy!(f, ∇f!, lmo, x0, x_out_ptr, Int(n), max_iters, tol,
                            step_rule, L0)
end

# ── Bilevel solve ────────────────────────────────────────────────────

Base.@ccallable function marg_bilevel_solve(
    inner_obj_ptr::Ptr{Cvoid},
    inner_grad_ptr::Ptr{Cvoid},
    lmo_ptr::Ptr{Cvoid},
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
    step_rule::Cint,
    L0::Cdouble,
    cg_maxiter::Cint,
    cg_tol::Cdouble,
    cg_lambda::Cdouble,
    userdata::Ptr{Cvoid},
)::CBilevelResult
    (n <= 0 || ntheta <= 0 ||
     any(==(C_NULL), (inner_obj_ptr, inner_grad_ptr, lmo_ptr, outer_grad_ptr,
                      hvp_ptr, cross_vjp_ptr, x0_ptr, x_out_ptr,
                      theta_ptr, theta_grad_out_ptr))) && return _ERROR_BILEVEL_RESULT
    # reimplements bilevel_solve logic because C function pointers are opaque
    # to Julia's AD backends -- the C user provides derivative callbacks directly
    try
        nn = Int(n)
        nt = Int(ntheta)
        mi = Int(max_iters)
        t = Float64(tol)
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
        if step_rule == Cint(1)
            x_star, inner_res = solve(f, ∇f!, lmo, x0; max_iters=mi, tol=t,
                                      step_rule=AdaptiveStepSize(Float64(L0)))
        else
            x_star, inner_res = solve(f, ∇f!, lmo, x0; max_iters=mi, tol=t)
        end

        # copy inner solution immediately (preserves result even if differentiation fails)
        GC.@preserve x_star unsafe_copyto!(x_out_ptr, pointer(x_star), nn)

        # 2. compute outer gradient x̄ = ∇_x L(x*)
        x̄ = Vector{Float64}(undef, nn)
        ccall(outer_grad_ptr, Cvoid,
              (Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cvoid}),
              x̄, x_star, n, userdata)

        # 3. CG solve (H + λI)u = x̄ using HVP callback
        Hp_buf = Vector{Float64}(undef, nn)
        function hvp_fn(p)
            ccall(hvp_ptr, Cvoid,
                  (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Ptr{Cvoid}),
                  Hp_buf, x_star, p, θ, n, ntheta, userdata)
            return Hp_buf
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

        # copy theta gradient
        GC.@preserve θ̄ unsafe_copyto!(theta_grad_out_ptr, pointer(θ̄), nt)

        return CBilevelResult(
            inner_res.objective, inner_res.gap,
            Cint(inner_res.iterations), Cint(inner_res.converged), Cint(inner_res.discards),
            Cint(cg_res.iterations), cg_res.residual_norm, Cint(cg_res.converged),
            Cint(0),
        )
    catch e
        @error "marg_bilevel_solve: Julia exception" exception=(e, catch_backtrace())
        return _ERROR_BILEVEL_RESULT
    end
end

# ── Precompile workload ──────────────────────────────────────────────
# JuliaC --trim=unsafe needs all reachable methods compiled into the image.
# Exercise every solve() code path with the same types the @ccallable
# wrappers will use at runtime.

let
    n = 2
    x0 = [0.5, 0.5]

    # mimic callback closures (same anonymous function structure as wrappers)
    f_obj = _wrap_obj(C_NULL, Cint(n), C_NULL)
    f_grad = _wrap_grad(C_NULL, Cint(n), C_NULL)
    f_lmo = _wrap_lmo(C_NULL, Cint(n), C_NULL)

    oracles = (f_lmo, Simplex(1.0), ProbSimplex(1.0), Box(zeros(2), ones(2)))
    step_rules = (MonotonicStepSize(), AdaptiveStepSize(1.0))

    for lmo in oracles, sr in step_rules
        try
            solve(f_obj, f_grad, lmo, x0; max_iters=1, tol=1.0, step_rule=sr)
        catch
        end
    end

    # exercise bilevel inner closures (different closure types from bilevel callbacks)
    θ = [0.5, 0.5]
    f_inner(x) = ccall(C_NULL, Cdouble,
                       (Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Ptr{Cvoid}),
                       x, θ, Cint(n), Cint(n), C_NULL)
    function g_inner(g, x)
        ccall(C_NULL, Cvoid,
              (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Ptr{Cvoid}),
              g, x, θ, Cint(n), Cint(n), C_NULL)
        return g
    end
    for sr in step_rules
        try
            solve(f_inner, g_inner, f_lmo, x0; max_iters=1, tol=1.0, step_rule=sr)
        catch
        end
    end
end

end # module LibMarguerite
