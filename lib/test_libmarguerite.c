/**
 * Minimal test: minimize f(x) = 0.5 * ||x - target||^2
 * over the probability simplex {x >= 0, sum(x) = 1}.
 *
 * Compile:  cc -o test_libmarguerite test_libmarguerite.c -ldl
 * Run:      ./test_libmarguerite ./build/lib/libmarguerite.so
 */

#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <math.h>
#include "../lib/include/marguerite.h"

/* ── userdata carries the target vector ───────────────────────────── */
typedef struct {
    const double *target;
    int n;
} qp_data_t;

/* f(x) = 0.5 * sum_i (x_i - target_i)^2 */
static double qp_obj(const double *x, int32_t n, void *ud) {
    qp_data_t *d = (qp_data_t *)ud;
    double val = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = x[i] - d->target[i];
        val += 0.5 * diff * diff;
    }
    return val;
}

/* grad_i = x_i - target_i */
static void qp_grad(double *g, const double *x, int32_t n, void *ud) {
    qp_data_t *d = (qp_data_t *)ud;
    for (int i = 0; i < n; i++) {
        g[i] = x[i] - d->target[i];
    }
}

typedef marguerite_result_t (*solve_prob_simplex_fn)(
    marguerite_obj_fn, marguerite_grad_fn,
    const double *, double *, int32_t,
    double, int32_t, double, int32_t, void *
);

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <path-to-libmarguerite.so>\n", argv[0]);
        return 1;
    }

    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
    if (!lib) {
        fprintf(stderr, "dlopen: %s\n", dlerror());
        return 2;
    }

    solve_prob_simplex_fn solve = (solve_prob_simplex_fn)dlsym(lib, "marguerite_solve_prob_simplex");
    if (!solve) {
        fprintf(stderr, "dlsym: %s\n", dlerror());
        return 3;
    }

    /* 3D problem: project target=(0.5, 0.3, 0.2) onto probability simplex.
       Since target already sums to 1 and is non-negative, solution = target.
       Note: convergence flag uses relative tolerance (gap <= tol * |f(x)|),
       so when optimal f(x) ~ 0 convergence may not trigger. We check
       solution quality directly instead. */
    const int n = 3;
    double target[] = {0.5, 0.3, 0.2};
    double x0[]     = {1.0/3, 1.0/3, 1.0/3};  /* uniform start */
    double x_out[3] = {0};

    qp_data_t data = { .target = target, .n = n };

    marguerite_result_t res = solve(
        qp_obj, qp_grad,
        x0, x_out, n,
        1.0,     /* radius */
        10000,   /* max_iters */
        1e-7,    /* tol */
        1,       /* monotonic */
        &data
    );

    printf("iters: %d  objective: %.10e  gap: %.10e\n",
           res.iterations, res.objective, res.gap);
    printf("solution: [%.6f, %.6f, %.6f]\n", x_out[0], x_out[1], x_out[2]);

    /* Check solution quality */
    int ok = 1;
    for (int i = 0; i < n; i++) {
        if (fabs(x_out[i] - target[i]) > 1e-3) {
            fprintf(stderr, "FAIL: x[%d]=%.6f, expected %.6f\n", i, x_out[i], target[i]);
            ok = 0;
        }
    }
    if (res.objective > 1e-6) {
        fprintf(stderr, "FAIL: objective %.10e too large\n", res.objective);
        ok = 0;
    }

    if (ok) {
        printf("PASS\n");
    }

    dlclose(lib);
    return ok ? 0 : 1;
}
