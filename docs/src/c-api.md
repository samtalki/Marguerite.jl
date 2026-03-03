<!-- Copyright 2026 Samuel Talkington and contributors
   SPDX-License-Identifier: Apache-2.0 -->

# C API

Marguerite provides a C shared library built with [JuliaC.jl](https://github.com/JuliaLang/JuliaC.jl), exposing the Frank-Wolfe solver to C/C++ consumers. The library is loaded at runtime via `dlopen`/`dlsym`.

## Building

Requires Julia 1.12+.

```bash
julia --project=lib -e 'using Pkg; Pkg.instantiate()'
julia --project=lib lib/build.jl
```

Output: `lib/build/lib/libmarguerite.{so,dylib,dll}`

## Result types

```c
typedef struct {
    double   objective;
    double   gap;
    int32_t  iterations;
    int32_t  converged;
    int32_t  discards;
    int32_t  status;      /* MARG_OK or MARG_ERROR */
} marg_result_t;

typedef struct {
    double   inner_objective;
    double   inner_gap;
    int32_t  inner_iterations;
    int32_t  inner_converged;
    int32_t  inner_discards;
    int32_t  cg_iterations;
    double   cg_residual;
    int32_t  cg_converged;
    int32_t  status;
} marg_bilevel_result_t;
```

## Status codes and step rules

```c
#define MARG_OK        0
#define MARG_ERROR   (-1)

#define MARG_STEP_MONOTONIC  0   /* γ_t = 2/(t+2) */
#define MARG_STEP_ADAPTIVE   1   /* backtracking line-search with Lipschitz estimation */
```

## Callback signatures

The API uses C function pointers with a `void *userdata` parameter for passing context.

```c
typedef double (*marg_obj_fn)(const double *x, int32_t n, void *userdata);
typedef void (*marg_grad_fn)(double *g_out, const double *x, int32_t n, void *userdata);
typedef void (*marg_lmo_fn)(double *v_out, const double *g, int32_t n, void *userdata);
```

Bilevel callbacks take additional `theta` and `ntheta` parameters:

```c
typedef double (*marg_inner_obj_fn)(const double *x, const double *theta,
                                     int32_t n, int32_t ntheta, void *userdata);
typedef void (*marg_inner_grad_fn)(double *g_out, const double *x, const double *theta,
                                    int32_t n, int32_t ntheta, void *userdata);
typedef void (*marg_hvp_fn)(double *Hp_out, const double *x, const double *p,
                             const double *theta, int32_t n, int32_t ntheta, void *userdata);
typedef void (*marg_cross_vjp_fn)(double *theta_grad_out, const double *x, const double *u,
                                   const double *theta, int32_t n, int32_t ntheta, void *userdata);
```

## Forward solve

Four variants are available, differing only in the constraint set:

| Function | Constraint |
|----------|-----------|
| `marg_solve` | User-supplied LMO callback |
| `marg_solve_simplex` | ``x \geq 0, \sum x_i \leq r`` |
| `marg_solve_prob_simplex` | ``x \geq 0, \sum x_i = r`` |
| `marg_solve_box` | ``\ell \leq x \leq u`` |

Common parameters:
- `x0` -- initial feasible point (not modified)
- `x_out` -- output buffer for the solution (length `n`)
- `max_iters` -- maximum Frank-Wolfe iterations
- `tol` -- convergence tolerance (gap ``\leq`` tol ``\cdot |f(x)|``)
- `step_rule` -- `MARG_STEP_MONOTONIC` (default ``\gamma_t = 2/(t+2)``) or `MARG_STEP_ADAPTIVE` (backtracking line-search)
- `L0` -- initial Lipschitz estimate for adaptive step size (ignored when `step_rule = MARG_STEP_MONOTONIC`)

## Bilevel solve

[`marg_bilevel_solve`](@id c-bilevel) computes ``\nabla_\theta L(x^*(\theta))`` via implicit differentiation. It requires 6 callbacks:

| Callback | Mathematical meaning |
|----------|---------------------|
| `inner_obj` | ``f(x, \theta)`` |
| `inner_grad` | ``\nabla_x f(x, \theta)`` |
| `lmo` | Linear minimization oracle |
| `outer_grad` | ``\nabla_x L(x)`` |
| `hvp` | ``\nabla^2_{xx} f \cdot p`` |
| `cross_vjp` | ``(\partial \nabla_x f / \partial \theta)^T u`` |

Additional parameters:
- `theta` -- parameter vector (length `ntheta`)
- `theta_grad_out` -- output buffer for ``\nabla_\theta L`` (length `ntheta`)
- `step_rule`, `L0` -- step size rule for the inner solve (same as forward solve)
- `cg_maxiter`, `cg_tol`, `cg_lambda` -- conjugate gradient settings for the Hessian solve

### Why 6 callbacks?

Julia's AD backends operate on LLVM IR. A C function pointer is opaque at runtime, so the solver cannot differentiate through it automatically. All derivative information must be provided explicitly via callbacks.

A future enhancement may use [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) to differentiate C code compiled with `clang -fembed-bitcode`, which would reduce the callback count.

## Error handling

All functions return a struct with a `status` field:
- `MARG_OK` (`0`) -- success
- `MARG_ERROR` (`-1`) -- Julia exception (details printed to stderr)

Invalid inputs (NULL pointers, `n <= 0`) return an error result immediately without entering the solver. Always check `status` before using results. On error, floating-point fields are `NaN`.

## Example

```c
#include <dlfcn.h>
#include "include/marguerite.h"

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
        const double *, double *, int32_t,
        double, int32_t, double, int32_t, double, void *);
    fn_t solve = (fn_t)dlsym(lib, "marg_solve_prob_simplex");

    double target[] = {0.5, 0.3, 0.2};
    double x0[] = {1.0/3, 1.0/3, 1.0/3};
    double x_out[3];

    marg_result_t r = solve(my_obj, my_grad, x0, x_out, 3,
                            1.0, 1000, 1e-7, MARG_STEP_MONOTONIC, 1.0, target);

    if (r.status != MARG_OK) { /* handle error */ }
    /* x_out contains the solution */
}
```

## Deployment

Ship the entire `lib/build/` directory. It contains `libmarguerite.so` alongside Julia runtime dependencies. Link with:

```bash
cc -o myapp myapp.c -Llib/build/lib -lmarguerite -Wl,-rpath,lib/build/lib
```
