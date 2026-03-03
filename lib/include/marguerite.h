/**
 * marguerite.h — C API for Marguerite.jl Frank-Wolfe solver
 *
 * Usage:
 *   void *lib = dlopen("path/to/libmarguerite.so", RTLD_NOW | RTLD_GLOBAL);
 *   marguerite_solve_simplex_fn solve =
 *       (marguerite_solve_simplex_fn)dlsym(lib, "marguerite_solve_simplex");
 *   marguerite_result_t r = solve(f, grad, x0, x_out, n, 1.0, 1000, 1e-7, 1, NULL);
 */

#ifndef MARGUERITE_H
#define MARGUERITE_H

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
} marguerite_result_t;

/* ── Callback typedefs ───────────────────────────────────────────── */

/** Objective function: f(x, n, userdata) -> double */
typedef double (*marguerite_obj_fn)(const double *x, int32_t n, void *userdata);

/** Gradient: grad(g_out, x, n, userdata) — writes ∇f(x) into g_out */
typedef void (*marguerite_grad_fn)(double *g_out, const double *x, int32_t n, void *userdata);

/** LMO: lmo(v_out, g, n, userdata) — writes argmin <g,v> into v_out */
typedef void (*marguerite_lmo_fn)(double *v_out, const double *g, int32_t n, void *userdata);

/* ── Generic solve (user supplies f, grad, lmo) ──────────────────── */

marguerite_result_t marguerite_solve(
    marguerite_obj_fn  f,
    marguerite_grad_fn grad,
    marguerite_lmo_fn  lmo,
    const double      *x0,
    double            *x_out,
    int32_t            n,
    int32_t            max_iters,
    double             tol,
    int32_t            monotonic,
    void              *userdata
);

/* ── Simplex: min f(x) s.t. x >= 0, sum(x) <= radius ────────────── */

marguerite_result_t marguerite_solve_simplex(
    marguerite_obj_fn  f,
    marguerite_grad_fn grad,
    const double      *x0,
    double            *x_out,
    int32_t            n,
    double             radius,
    int32_t            max_iters,
    double             tol,
    int32_t            monotonic,
    void              *userdata
);

/* ── Probability simplex: min f(x) s.t. x >= 0, sum(x) = radius ── */

marguerite_result_t marguerite_solve_prob_simplex(
    marguerite_obj_fn  f,
    marguerite_grad_fn grad,
    const double      *x0,
    double            *x_out,
    int32_t            n,
    double             radius,
    int32_t            max_iters,
    double             tol,
    int32_t            monotonic,
    void              *userdata
);

/* ── Box: min f(x) s.t. lb <= x <= ub ────────────────────────────── */

marguerite_result_t marguerite_solve_box(
    marguerite_obj_fn  f,
    marguerite_grad_fn grad,
    const double      *x0,
    double            *x_out,
    int32_t            n,
    const double      *lb,
    const double      *ub,
    int32_t            max_iters,
    double             tol,
    int32_t            monotonic,
    void              *userdata
);

#ifdef __cplusplus
}
#endif

#endif /* MARGUERITE_H */
