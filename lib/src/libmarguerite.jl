# Copyright 2026 Samuel Talkington and contributors
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

const _STEP_ADAPTIVE = Cint(1)

_make_step_rule(flag::Cint, L0) =
    flag == _STEP_ADAPTIVE ? AdaptiveStepSize(Float64(L0)) : MonotonicStepSize()

# ── Internal helpers ───────────────────────────────────────────────

function _to_cresult(x, res, x_out_ptr, n)
    GC.@preserve x unsafe_copyto!(x_out_ptr, pointer(x), n)
    return CResult(res.objective, res.gap,
                   Cint(res.iterations), Cint(res.converged), Cint(res.discards),
                   Cint(0))
end

function _solve_and_copy!(f_callable, grad_callable, lmo, x0, x_out_ptr, n, max_iters, tol, step_rule_flag, L0)
    try
        # only ∇f! needs a closure wrapper — solve() types it as ::Function
        _∇f!(g, x) = grad_callable(g, x)
        x, res = solve(f_callable, _∇f!, lmo, x0;
                        max_iters=Int(max_iters), tol=tol,
                        step_rule=_make_step_rule(step_rule_flag, L0))
        return _to_cresult(x, res, x_out_ptr, n)
    catch e
        @error "marg_solve: Julia exception" exception=(e, catch_backtrace())
        return _ERROR_RESULT
    end
end

# ── Callback wrappers ─────────────────────────────────────────────
# Callable structs (not closures) so the trimmer can trace the types.

struct WrappedObj
    f_ptr::Ptr{Cvoid}
    n::Cint
    userdata::Ptr{Cvoid}
end
function (w::WrappedObj)(x)
    w.f_ptr == C_NULL && return 0.0
    return ccall(w.f_ptr, Cdouble, (Ptr{Cdouble}, Cint, Ptr{Cvoid}), x, w.n, w.userdata)
end

struct WrappedGrad
    grad_ptr::Ptr{Cvoid}
    n::Cint
    userdata::Ptr{Cvoid}
end
function (w::WrappedGrad)(g, x)
    w.grad_ptr == C_NULL && return g
    ccall(w.grad_ptr, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cvoid}), g, x, w.n, w.userdata)
    return g
end

struct WrappedLMO
    lmo_ptr::Ptr{Cvoid}
    n::Cint
    userdata::Ptr{Cvoid}
end
function (w::WrappedLMO)(v, g)
    w.lmo_ptr == C_NULL && return v
    ccall(w.lmo_ptr, Cvoid, (Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cvoid}), v, g, w.n, w.userdata)
    return v
end

# bilevel callback wrappers
struct WrappedInnerObj
    ptr::Ptr{Cvoid}
    θ::Vector{Float64}
    n::Cint
    nθ::Cint
    userdata::Ptr{Cvoid}
end
function (w::WrappedInnerObj)(x)
    w.ptr == C_NULL && return 0.0
    return ccall(w.ptr, Cdouble,
        (Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Ptr{Cvoid}),
        x, w.θ, w.n, w.nθ, w.userdata)
end

struct WrappedInnerGrad
    ptr::Ptr{Cvoid}
    θ::Vector{Float64}
    n::Cint
    nθ::Cint
    userdata::Ptr{Cvoid}
end
function (w::WrappedInnerGrad)(g, x)
    w.ptr == C_NULL && return g
    ccall(w.ptr, Cvoid,
          (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Ptr{Cvoid}),
          g, x, w.θ, w.n, w.nθ, w.userdata)
    return g
end

struct WrappedHVP
    ptr::Ptr{Cvoid}
    x_star::Vector{Float64}
    θ::Vector{Float64}
    Hp_buf::Vector{Float64}
    n::Cint
    nθ::Cint
    userdata::Ptr{Cvoid}
end
function (w::WrappedHVP)(p)
    w.ptr == C_NULL && return w.Hp_buf
    ccall(w.ptr, Cvoid,
          (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint, Ptr{Cvoid}),
          w.Hp_buf, w.x_star, p, w.θ, w.n, w.nθ, w.userdata)
    return w.Hp_buf
end

# ── @ccallable shared helper ──────────────────────────────────────────

function _wrap_and_solve(f_ptr, grad_ptr, x0_ptr, x_out_ptr, n, lmo,
                         max_iters, tol, step_rule, L0, userdata)
    nn = Int(n)
    x0 = unsafe_wrap(Array, x0_ptr, nn)
    f = WrappedObj(f_ptr, n, userdata)
    ∇f! = WrappedGrad(grad_ptr, n, userdata)
    return _solve_and_copy!(f, ∇f!, lmo, x0, x_out_ptr, nn, max_iters, tol,
                            step_rule, L0)
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
    lmo = WrappedLMO(lmo_ptr, n, userdata)
    return _wrap_and_solve(f_ptr, grad_ptr, x0_ptr, x_out_ptr, n, lmo,
                           max_iters, tol, step_rule, L0, userdata)
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
    return _wrap_and_solve(f_ptr, grad_ptr, x0_ptr, x_out_ptr, n,
                           Simplex(radius),
                           max_iters, tol, step_rule, L0, userdata)
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
    return _wrap_and_solve(f_ptr, grad_ptr, x0_ptr, x_out_ptr, n,
                           ProbSimplex(radius),
                           max_iters, tol, step_rule, L0, userdata)
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
    lb = unsafe_wrap(Array, lb_ptr, Int(n))
    ub = unsafe_wrap(Array, ub_ptr, Int(n))
    return _wrap_and_solve(f_ptr, grad_ptr, x0_ptr, x_out_ptr, n,
                           Box{Float64}(lb, ub),
                           max_iters, tol, step_rule, L0, userdata)
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
        x0 = unsafe_wrap(Array, x0_ptr, nn)
        θ = unsafe_wrap(Array, theta_ptr, nt)

        # wrap callbacks as callable structs
        f_callable = WrappedInnerObj(inner_obj_ptr, θ, n, ntheta, userdata)
        g_callable = WrappedInnerGrad(inner_grad_ptr, θ, n, ntheta, userdata)
        lmo = WrappedLMO(lmo_ptr, n, userdata)

        # only ∇f! needs a closure wrapper — solve() types it as ::Function
        _∇f!(g, x) = g_callable(g, x)

        # 1. solve inner FW problem
        x_star::Vector{Float64}, inner_res = solve(
            f_callable, _∇f!, lmo, x0;
            max_iters=Int(max_iters), tol=tol, step_rule=_make_step_rule(step_rule, L0))

        # copy inner solution immediately (preserves result even if differentiation fails)
        GC.@preserve x_star unsafe_copyto!(x_out_ptr, pointer(x_star), nn)

        # 2. compute outer gradient x̄ = ∇_x L(x*)
        x̄ = Vector{Float64}(undef, nn)
        ccall(outer_grad_ptr, Cvoid,
              (Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cvoid}),
              x̄, x_star, n, userdata)

        # 3. CG solve (H + λI)u = x̄ using HVP callback
        Hp_buf = Vector{Float64}(undef, nn)
        hvp_fn = WrappedHVP(hvp_ptr, x_star, θ, Hp_buf, n, ntheta, userdata)
        u, cg_res = _cg_solve(hvp_fn, x̄;
                               maxiter=Int(cg_maxiter),
                               tol=cg_tol,
                               λ=cg_lambda)

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

# No precompile workload — @ccallable functions serve as trim roots.

end # module LibMarguerite
