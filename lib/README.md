# Marguerite C API

C shared library for the Marguerite.jl Frank-Wolfe solver, built with [JuliaC.jl](https://github.com/JuliaLang/JuliaC.jl).

## Building

Requires Julia 1.12+.

```bash
# install dependencies
julia --project=lib -e 'using Pkg; Pkg.instantiate()'

# build shared library
julia --project=lib lib/build.jl
```

Output lands in `lib/build/lib/libmarguerite.{so,dylib,dll}`.

## Testing

```bash
cd lib
cc -o test_libmarguerite test_libmarguerite.c -ldl -lm
./test_libmarguerite ./build/lib/libmarguerite.so
```

## Loading the library

The library is loaded at runtime via `dlopen`/`dlsym` (or `LoadLibrary` on Windows):

```c
#include <dlfcn.h>
#include "include/marguerite.h"

void *lib = dlopen("path/to/build/lib/libmarguerite.so", RTLD_NOW | RTLD_GLOBAL);
```

When deploying, ship the entire `build/` directory -- it contains the Julia runtime dependencies alongside the shared library.

## API reference

### Status codes and step rules

```c
#define MARG_OK        0
#define MARG_ERROR   (-1)

#define MARG_STEP_MONOTONIC  0   /* γ_t = 2/(t+2) */
#define MARG_STEP_ADAPTIVE   1   /* backtracking line-search */
```

### Result types

| Type | Fields |
|------|--------|
| `marg_result_t` | `objective`, `gap`, `iterations`, `converged`, `discards`, `status` |
| `marg_bilevel_result_t` | `inner_objective`, `inner_gap`, `inner_iterations`, `inner_converged`, `inner_discards`, `cg_iterations`, `cg_residual`, `cg_converged`, `status` |

`status` is `MARG_OK` on success, `MARG_ERROR` if a Julia exception occurred or input validation failed.

### Callback signatures

| Typedef | Signature | Description |
|---------|-----------|-------------|
| `marg_obj_fn` | `(x, n, userdata) -> double` | Objective value |
| `marg_grad_fn` | `(g_out, x, n, userdata) -> void` | Gradient into `g_out` |
| `marg_lmo_fn` | `(v_out, g, n, userdata) -> void` | LMO: argmin <g,v> into `v_out` |
| `marg_inner_obj_fn` | `(x, theta, n, ntheta, userdata) -> double` | Parameterized objective |
| `marg_inner_grad_fn` | `(g_out, x, theta, n, ntheta, userdata) -> void` | Parameterized gradient |
| `marg_hvp_fn` | `(Hp_out, x, p, theta, n, ntheta, userdata) -> void` | Hessian-vector product |
| `marg_cross_vjp_fn` | `(theta_grad_out, x, u, theta, n, ntheta, userdata) -> void` | Cross-Hessian VJP |

### Forward solve functions

All four variants share the same pattern: provide an objective, gradient, and constraint set (via LMO or built-in oracle), and get back a `marg_result_t`.

```c
/* generic -- user supplies LMO callback */
marg_result_t marg_solve(f, grad, lmo, x0, x_out, n, max_iters, tol, step_rule, L0, userdata);

/* simplex: x >= 0, sum(x) <= radius */
marg_result_t marg_solve_simplex(f, grad, x0, x_out, n, radius, max_iters, tol, step_rule, L0, userdata);

/* probability simplex: x >= 0, sum(x) = radius */
marg_result_t marg_solve_prob_simplex(f, grad, x0, x_out, n, radius, max_iters, tol, step_rule, L0, userdata);

/* box: lb <= x <= ub */
marg_result_t marg_solve_box(f, grad, x0, x_out, n, lb, ub, max_iters, tol, step_rule, L0, userdata);
```

Parameters:
- `step_rule` -- `MARG_STEP_MONOTONIC` (default) or `MARG_STEP_ADAPTIVE`
- `L0` -- initial Lipschitz estimate for adaptive step size (ignored when monotonic)

### Bilevel solve

`marg_bilevel_solve` solves an inner Frank-Wolfe problem and computes `d L(x*(theta)) / d theta` via implicit differentiation. The C user provides all derivative information via callbacks -- 6 in total.

```c
marg_bilevel_result_t marg_bilevel_solve(
    inner_obj, inner_grad, lmo,
    outer_grad,
    hvp, cross_vjp,
    x0, x_out, n,
    theta, theta_grad_out, ntheta,
    max_iters, tol, step_rule, L0,
    cg_maxiter, cg_tol, cg_lambda,
    userdata
);
```

**Why so many callbacks?** Julia's AD backends need LLVM IR to differentiate through; a C function pointer is opaque at runtime. So all derivative information must be provided explicitly. See the follow-up issue on Enzyme.jl for a potential future path to auto-differentiation from C.

## Example: forward solve

```c
#include <dlfcn.h>
#include "include/marguerite.h"

/* f(x) = 0.5 ||x - target||² */
double my_obj(const double *x, int32_t n, void *ud) {
    double *target = (double *)ud;
    double val = 0.0;
    for (int i = 0; i < n; i++) {
        double d = x[i] - target[i];
        val += 0.5 * d * d;
    }
    return val;
}

void my_grad(double *g, const double *x, int32_t n, void *ud) {
    double *target = (double *)ud;
    for (int i = 0; i < n; i++)
        g[i] = x[i] - target[i];
}

int main() {
    void *lib = dlopen("build/lib/libmarguerite.so", RTLD_NOW | RTLD_GLOBAL);

    typedef marg_result_t (*fn_t)(marg_obj_fn, marg_grad_fn,
        const double *, double *, int32_t, double, int32_t, double,
        int32_t, double, void *);
    fn_t solve = (fn_t)dlsym(lib, "marg_solve_prob_simplex");

    double target[] = {0.5, 0.3, 0.2};
    double x0[] = {1.0/3, 1.0/3, 1.0/3};
    double x_out[3];

    marg_result_t r = solve(my_obj, my_grad, x0, x_out, 3,
                            1.0, 1000, 1e-7, MARG_STEP_MONOTONIC, 1.0, target);

    if (r.status != MARG_OK) { /* handle error */ }
    /* x_out now contains the solution */
}
```

## Error handling

All functions return a result struct with a `status` field:
- `MARG_OK` (`0`) -- success
- `MARG_ERROR` (`-1`) -- a Julia exception occurred or input validation failed (details printed to stderr)

Invalid inputs (NULL pointers, `n <= 0`) return an error result immediately. Always check `status` before using the result. On error, `objective` and `gap` are `NaN`.

## Linking against the library

The C user compiles their `main.c` and links against `libmarguerite.so`:

```bash
cc -o myapp myapp.c -Llib/build/lib -lmarguerite -Wl,-rpath,lib/build/lib
```

When deploying, ship the entire `build/` bundle directory (which includes Julia runtime dependencies).
