/**
 * marguerite.h — C API for Marguerite.jl Frank-Wolfe solver
 *
 * Usage:
 *   void *lib = dlopen("path/to/libmarguerite.so", RTLD_NOW | RTLD_GLOBAL);
 *   marg_solve_prob_simplex_fn solve =
 *       (marg_solve_prob_simplex_fn)dlsym(lib, "marg_solve_prob_simplex");
 *   marg_result_t r = solve(f, grad, x0, x_out, n, 1.0, 1000, 1e-7, 1, NULL);
 *   if (r.status != 0) { /* handle error */ }
 */

#ifndef MARG_H
#define MARG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Result struct (matches Julia CResult layout) ────────────────── */

typedef struct {
    double   objective;
    double   gap;
    int32_t  iterations;
    int32_t  converged;   /* 0 = false, nonzero = true */
    int32_t  discards;
    int32_t  status;      /* 0 = success, -1 = Julia exception */
} marg_result_t;

/* ── Callback typedefs ───────────────────────────────────────────── */

/** Objective function: f(x, n, userdata) -> double */
typedef double (*marg_obj_fn)(const double *x, int32_t n, void *userdata);

/** Gradient: grad(g_out, x, n, userdata) — writes ∇f(x) into g_out */
typedef void (*marg_grad_fn)(double *g_out, const double *x, int32_t n, void *userdata);

/** LMO: lmo(v_out, g, n, userdata) — writes argmin <g,v> into v_out */
typedef void (*marg_lmo_fn)(double *v_out, const double *g, int32_t n, void *userdata);

/* ── Generic solve (user supplies f, grad, lmo) ──────────────────── */

marg_result_t marg_solve(
    marg_obj_fn   f,
    marg_grad_fn  grad,
    marg_lmo_fn   lmo,
    const double *x0,
    double       *x_out,
    int32_t       n,
    int32_t       max_iters,
    double        tol,
    int32_t       monotonic,
    void         *userdata
);

/* ── Simplex: min f(x) s.t. x >= 0, sum(x) <= radius ────────────── */

marg_result_t marg_solve_simplex(
    marg_obj_fn   f,
    marg_grad_fn  grad,
    const double *x0,
    double       *x_out,
    int32_t       n,
    double        radius,
    int32_t       max_iters,
    double        tol,
    int32_t       monotonic,
    void         *userdata
);

/* ── Probability simplex: min f(x) s.t. x >= 0, sum(x) = radius ── */

marg_result_t marg_solve_prob_simplex(
    marg_obj_fn   f,
    marg_grad_fn  grad,
    const double *x0,
    double       *x_out,
    int32_t       n,
    double        radius,
    int32_t       max_iters,
    double        tol,
    int32_t       monotonic,
    void         *userdata
);

/* ── Box: min f(x) s.t. lb <= x <= ub ────────────────────────────── */

marg_result_t marg_solve_box(
    marg_obj_fn   f,
    marg_grad_fn  grad,
    const double *x0,
    double       *x_out,
    int32_t       n,
    const double *lb,
    const double *ub,
    int32_t       max_iters,
    double        tol,
    int32_t       monotonic,
    void         *userdata
);

/* ── Bilevel callback typedefs ────────────────────────────────────── */

/** Inner objective with parameters: inner_obj(x, theta, n, ntheta, userdata) -> double */
typedef double (*marg_inner_obj_fn)(const double *x, const double *theta,
                                     int32_t n, int32_t ntheta, void *userdata);

/** Gradient of inner objective w.r.t. x: inner_grad(g_out, x, theta, n, ntheta, userdata) */
typedef void (*marg_inner_grad_fn)(double *g_out, const double *x, const double *theta,
                                    int32_t n, int32_t ntheta, void *userdata);

/** Hessian-vector product of inner objective: Hp_out = ∇²_xx f(x, θ) · p */
typedef void (*marg_hvp_fn)(double *Hp_out, const double *x, const double *p,
                             const double *theta, int32_t n, int32_t ntheta, void *userdata);

/** Cross-Hessian VJP: theta_grad_out = (∂(∇_x f)/∂θ)^T u */
typedef void (*marg_cross_vjp_fn)(double *theta_grad_out, const double *x, const double *u,
                                   const double *theta, int32_t n, int32_t ntheta, void *userdata);

/* ── Bilevel result struct ───────────────────────────────────────── */

typedef struct {
    /* inner FW solve diagnostics */
    double   inner_objective;
    double   inner_gap;
    int32_t  inner_iterations;
    int32_t  inner_converged;
    int32_t  inner_discards;
    /* CG solve diagnostics */
    int32_t  cg_iterations;
    double   cg_residual;
    int32_t  cg_converged;
    /* status */
    int32_t  status;          /* 0 = ok, -1 = error */
} marg_bilevel_result_t;

/* ── Bilevel solve ───────────────────────────────────────────────── */

marg_bilevel_result_t marg_bilevel_solve(
    marg_inner_obj_fn   inner_obj,      /* f(x, θ) — inner objective */
    marg_inner_grad_fn  inner_grad,     /* ∇_x f(x, θ) — gradient w.r.t. x */
    marg_lmo_fn         lmo,            /* linear minimization oracle */
    marg_obj_fn         outer_obj,      /* L(x) — outer objective */
    marg_grad_fn        outer_grad,     /* ∇_x L(x) — outer gradient */
    marg_hvp_fn         hvp,            /* ∇²_xx f · p — Hessian-vector product */
    marg_cross_vjp_fn   cross_vjp,      /* (∂∇_x f/∂θ)^T u — cross-Hessian VJP */
    const double       *x0,
    double             *x_out,          /* [n] output: x* */
    int32_t             n,
    const double       *theta,
    double             *theta_grad_out, /* [ntheta] output: ∇_θ L(x*(θ)) */
    int32_t             ntheta,
    int32_t             max_iters,
    double              tol,
    int32_t             monotonic,
    int32_t             cg_maxiter,
    double              cg_tol,
    double              cg_lambda,
    void               *userdata
);

#ifdef __cplusplus
}
#endif

#endif /* MARG_H */
