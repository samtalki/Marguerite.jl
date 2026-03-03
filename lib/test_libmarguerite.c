/**
 * Tests for libmarguerite C API.
 *
 * Compile:  cc -o test_libmarguerite test_libmarguerite.c -ldl -lm
 * Run:      ./test_libmarguerite ./build/lib/libmarguerite.so
 */

#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <math.h>
#include "include/marguerite.h"

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, fmt, ...) do {                              \
    if (cond) { g_pass++; }                                     \
    else { g_fail++; fprintf(stderr, "FAIL: " fmt "\n", ##__VA_ARGS__); } \
} while (0)

/* ── Shared test problem: f(x) = 0.5 * ||x - target||^2 ────────── */

typedef struct {
    const double *target;
    int n;
} qp_data_t;

static double qp_obj(const double *x, int32_t n, void *ud) {
    qp_data_t *d = (qp_data_t *)ud;
    double val = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = x[i] - d->target[i];
        val += 0.5 * diff * diff;
    }
    return val;
}

static void qp_grad(double *g, const double *x, int32_t n, void *ud) {
    qp_data_t *d = (qp_data_t *)ud;
    for (int i = 0; i < n; i++) {
        g[i] = x[i] - d->target[i];
    }
}

/* ── Box LMO callback: project gradient onto box [0, 1]^n ────────── */

static void box_lmo(double *v_out, const double *g, int32_t n, void *ud) {
    (void)ud;
    for (int i = 0; i < n; i++) {
        v_out[i] = (g[i] < 0.0) ? 1.0 : 0.0;
    }
}

/* ── Bilevel callbacks for f(x, θ) = 0.5 ||x||² - θ'x ───────────── */

typedef struct {
    int n;
    int ntheta;
} bilevel_data_t;

/* inner objective: f(x, θ) = 0.5 ||x||² - θ'x */
static double bilevel_inner_obj(const double *x, const double *theta,
                                 int32_t n, int32_t ntheta, void *ud) {
    (void)ntheta; (void)ud;
    double val = 0.0;
    for (int i = 0; i < n; i++) {
        val += 0.5 * x[i] * x[i] - theta[i] * x[i];
    }
    return val;
}

/* inner gradient: ∇_x f = x - θ */
static void bilevel_inner_grad(double *g_out, const double *x, const double *theta,
                                int32_t n, int32_t ntheta, void *ud) {
    (void)ntheta; (void)ud;
    for (int i = 0; i < n; i++) {
        g_out[i] = x[i] - theta[i];
    }
}

/* HVP: ∇²_xx f · p = I · p = p  (Hessian is identity) */
static void bilevel_hvp(double *Hp_out, const double *x, const double *p,
                         const double *theta, int32_t n, int32_t ntheta, void *ud) {
    (void)x; (void)theta; (void)ntheta; (void)ud;
    for (int i = 0; i < n; i++) {
        Hp_out[i] = p[i];
    }
}

/* cross-Hessian VJP: (∂(∇_x f)/∂θ)^T u = (-I)^T u = -u */
static void bilevel_cross_vjp(double *theta_grad_out, const double *x, const double *u,
                               const double *theta, int32_t n, int32_t ntheta, void *ud) {
    (void)x; (void)theta; (void)ntheta; (void)ud;
    for (int i = 0; i < n; i++) {
        theta_grad_out[i] = -u[i];
    }
}

/* outer loss: L(x) = sum(x) */
static double bilevel_outer_obj(const double *x, int32_t n, void *ud) {
    (void)ud;
    double val = 0.0;
    for (int i = 0; i < n; i++) val += x[i];
    return val;
}

/* outer gradient: ∇_x L = ones */
static void bilevel_outer_grad(double *g_out, const double *x, int32_t n, void *ud) {
    (void)x; (void)ud;
    for (int i = 0; i < n; i++) g_out[i] = 1.0;
}

/* ── Prob simplex LMO for bilevel test ──────────────────────────── */

static void prob_simplex_lmo(double *v_out, const double *g, int32_t n, void *ud) {
    (void)ud;
    /* find argmin_i g_i, put all mass there */
    int min_idx = 0;
    for (int i = 1; i < n; i++) {
        if (g[i] < g[min_idx]) min_idx = i;
    }
    for (int i = 0; i < n; i++) v_out[i] = 0.0;
    v_out[min_idx] = 1.0;
}

/* ── Function pointer typedefs for dlsym ─────────────────────────── */

typedef marg_result_t (*solve_fn)(
    marg_obj_fn, marg_grad_fn, marg_lmo_fn,
    const double *, double *, int32_t,
    int32_t, double, int32_t, void *
);

typedef marg_result_t (*solve_simplex_fn)(
    marg_obj_fn, marg_grad_fn,
    const double *, double *, int32_t,
    double, int32_t, double, int32_t, void *
);

typedef marg_result_t (*solve_prob_simplex_fn)(
    marg_obj_fn, marg_grad_fn,
    const double *, double *, int32_t,
    double, int32_t, double, int32_t, void *
);

typedef marg_result_t (*solve_box_fn)(
    marg_obj_fn, marg_grad_fn,
    const double *, double *, int32_t,
    const double *, const double *,
    int32_t, double, int32_t, void *
);

typedef marg_bilevel_result_t (*bilevel_solve_fn)(
    marg_inner_obj_fn, marg_inner_grad_fn, marg_lmo_fn,
    marg_obj_fn, marg_grad_fn,
    marg_hvp_fn, marg_cross_vjp_fn,
    const double *, double *, int32_t,
    const double *, double *, int32_t,
    int32_t, double, int32_t,
    int32_t, double, double,
    void *
);

/* ── Tests ───────────────────────────────────────────────────────── */

static void test_solve(void *lib) {
    solve_fn fn = (solve_fn)dlsym(lib, "marg_solve");
    if (!fn) { g_fail++; fprintf(stderr, "FAIL: dlsym marg_solve: %s\n", dlerror()); return; }

    /* min 0.5 ||x - target||² over [0,1]^3 with box LMO callback.
       target = (0.5, 0.3, 0.8) is in [0,1]^3, so solution = target. */
    const int n = 3;
    double target[] = {0.5, 0.3, 0.8};
    double x0[] = {0.0, 0.0, 0.0};
    double x_out[3] = {0};
    qp_data_t data = { .target = target, .n = n };

    marg_result_t res = fn(qp_obj, qp_grad, box_lmo, x0, x_out, n, 10000, 1e-7, 1, &data);

    printf("[marg_solve] iters=%d obj=%.6e status=%d\n", res.iterations, res.objective, res.status);
    CHECK(res.status == 0, "marg_solve status=%d", res.status);
    for (int i = 0; i < n; i++) {
        CHECK(fabs(x_out[i] - target[i]) < 1e-2,
              "marg_solve x[%d]=%.6f expected %.6f", i, x_out[i], target[i]);
    }
}

static void test_solve_simplex(void *lib) {
    solve_simplex_fn fn = (solve_simplex_fn)dlsym(lib, "marg_solve_simplex");
    if (!fn) { g_fail++; fprintf(stderr, "FAIL: dlsym marg_solve_simplex: %s\n", dlerror()); return; }

    /* min 0.5 ||x - target||² over simplex {x >= 0, sum(x) <= 1}.
       target = (0.5, 0.3, 0.1) sums to 0.9 <= 1 and all non-neg, so solution = target. */
    const int n = 3;
    double target[] = {0.5, 0.3, 0.1};
    double x0[] = {0.0, 0.0, 0.0};
    double x_out[3] = {0};
    qp_data_t data = { .target = target, .n = n };

    marg_result_t res = fn(qp_obj, qp_grad, x0, x_out, n, 1.0, 10000, 1e-7, 1, &data);

    printf("[marg_solve_simplex] iters=%d obj=%.6e status=%d\n", res.iterations, res.objective, res.status);
    CHECK(res.status == 0, "marg_solve_simplex status=%d", res.status);
    for (int i = 0; i < n; i++) {
        CHECK(fabs(x_out[i] - target[i]) < 1e-2,
              "marg_solve_simplex x[%d]=%.6f expected %.6f", i, x_out[i], target[i]);
    }
}

static void test_solve_prob_simplex(void *lib) {
    solve_prob_simplex_fn fn = (solve_prob_simplex_fn)dlsym(lib, "marg_solve_prob_simplex");
    if (!fn) { g_fail++; fprintf(stderr, "FAIL: dlsym marg_solve_prob_simplex: %s\n", dlerror()); return; }

    /* min 0.5 ||x - target||² over probability simplex {x >= 0, sum(x) = 1}.
       target = (0.5, 0.3, 0.2) already on simplex, so solution = target. */
    const int n = 3;
    double target[] = {0.5, 0.3, 0.2};
    double x0[] = {1.0/3, 1.0/3, 1.0/3};
    double x_out[3] = {0};
    qp_data_t data = { .target = target, .n = n };

    marg_result_t res = fn(qp_obj, qp_grad, x0, x_out, n, 1.0, 10000, 1e-7, 1, &data);

    printf("[marg_solve_prob_simplex] iters=%d obj=%.6e status=%d\n", res.iterations, res.objective, res.status);
    CHECK(res.status == 0, "marg_solve_prob_simplex status=%d", res.status);
    for (int i = 0; i < n; i++) {
        CHECK(fabs(x_out[i] - target[i]) < 1e-3,
              "marg_solve_prob_simplex x[%d]=%.6f expected %.6f", i, x_out[i], target[i]);
    }
}

static void test_solve_box(void *lib) {
    solve_box_fn fn = (solve_box_fn)dlsym(lib, "marg_solve_box");
    if (!fn) { g_fail++; fprintf(stderr, "FAIL: dlsym marg_solve_box: %s\n", dlerror()); return; }

    /* min 0.5 ||x - target||² over [0, 1]^3.
       target = (0.5, 0.3, 0.8) is in [0,1]^3, so solution = target. */
    const int n = 3;
    double target[] = {0.5, 0.3, 0.8};
    double x0[] = {0.0, 0.0, 0.0};
    double x_out[3] = {0};
    double lb[] = {0.0, 0.0, 0.0};
    double ub[] = {1.0, 1.0, 1.0};
    qp_data_t data = { .target = target, .n = n };

    marg_result_t res = fn(qp_obj, qp_grad, x0, x_out, n, lb, ub, 10000, 1e-7, 1, &data);

    printf("[marg_solve_box] iters=%d obj=%.6e status=%d\n", res.iterations, res.objective, res.status);
    CHECK(res.status == 0, "marg_solve_box status=%d", res.status);
    for (int i = 0; i < n; i++) {
        CHECK(fabs(x_out[i] - target[i]) < 1e-2,
              "marg_solve_box x[%d]=%.6f expected %.6f", i, x_out[i], target[i]);
    }
}

static void test_bilevel_solve(void *lib) {
    bilevel_solve_fn fn = (bilevel_solve_fn)dlsym(lib, "marg_bilevel_solve");
    if (!fn) { g_fail++; fprintf(stderr, "FAIL: dlsym marg_bilevel_solve: %s\n", dlerror()); return; }

    /*
     * Inner: min_x f(x, θ) = 0.5 ||x||² - θ'x  over probability simplex
     * Outer: L(x) = sum(x)
     *
     * Hessian = I, so HVP is identity.
     * ∇_x f = x - θ, so ∂(∇_x f)/∂θ = -I, and cross-VJP = -u.
     *
     * With θ = (0.5, 0.3, 0.2):
     *   x*(θ) = projection of θ onto prob simplex = θ (since θ is on simplex)
     *   x̄ = ∇_x L = ones(3)
     *   (H + λI)u = x̄  =>  (1+λ)u = 1  =>  u = 1/(1+λ) * ones
     *   θ̄ = -(−u) = u = 1/(1+λ) * ones
     */
    const int n = 3;
    const int ntheta = 3;
    double theta[] = {0.5, 0.3, 0.2};
    double x0[] = {1.0/3, 1.0/3, 1.0/3};
    double x_out[3] = {0};
    double theta_grad[3] = {0};

    double cg_lambda = 1e-4;

    marg_bilevel_result_t res = fn(
        bilevel_inner_obj, bilevel_inner_grad, prob_simplex_lmo,
        bilevel_outer_obj, bilevel_outer_grad,
        bilevel_hvp, bilevel_cross_vjp,
        x0, x_out, n,
        theta, theta_grad, ntheta,
        10000,      /* max_iters */
        1e-7,       /* tol */
        1,          /* monotonic */
        50,         /* cg_maxiter */
        1e-6,       /* cg_tol */
        cg_lambda,  /* cg_lambda */
        NULL
    );

    printf("[marg_bilevel_solve] inner_iters=%d cg_iters=%d status=%d\n",
           res.inner_iterations, res.cg_iterations, res.status);
    printf("  x*   = [%.6f, %.6f, %.6f]\n", x_out[0], x_out[1], x_out[2]);
    printf("  grad = [%.6f, %.6f, %.6f]\n", theta_grad[0], theta_grad[1], theta_grad[2]);

    CHECK(res.status == 0, "marg_bilevel_solve status=%d", res.status);
    CHECK(res.cg_converged != 0, "marg_bilevel_solve CG did not converge");

    /* x* should be close to theta */
    for (int i = 0; i < n; i++) {
        CHECK(fabs(x_out[i] - theta[i]) < 1e-2,
              "bilevel x[%d]=%.6f expected %.6f", i, x_out[i], theta[i]);
    }

    /* θ̄ should be close to 1/(1+λ) * ones ≈ ones */
    double expected_grad = 1.0 / (1.0 + cg_lambda);
    for (int i = 0; i < ntheta; i++) {
        CHECK(fabs(theta_grad[i] - expected_grad) < 1e-2,
              "bilevel grad[%d]=%.6f expected %.6f", i, theta_grad[i], expected_grad);
    }
}

/* ── Main ────────────────────────────────────────────────────────── */

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

    test_solve(lib);
    test_solve_simplex(lib);
    test_solve_prob_simplex(lib);
    test_solve_box(lib);
    test_bilevel_solve(lib);

    dlclose(lib);

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
