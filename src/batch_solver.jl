# Copyright 2026 Samuel Talkington
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

"""
    batch_solve(f_batch, lmo, X0; kwargs...) -> (X, BatchResult)

Solve `B` independent Frank-Wolfe problems in lockstep:

```math
\\min_{x_b \\in C} f_b(x_b) \\quad b = 1, \\ldots, B
```

`X0` is an `(n, B)` matrix whose columns are initial points.

# Arguments
- `f_batch(X::AbstractMatrix) -> Vector`: returns per-problem objectives (length `B`)
- `lmo`: oracle (applied column-wise)
- `X0`: `(n, B)` initial point matrix

# Keyword Arguments
- `grad_batch`: in-place batched gradient `grad_batch(G, X)` writing `(n, B)` gradients.
  **Required** — no auto-gradient for batched mode.
- `max_iters::Int = 10000`: maximum lockstep iterations
- `tol::Real = 1e-4`: convergence tolerance per problem
- `monotonic::Bool = true`: reject non-improving updates
- `verbose::Bool = false`: print progress
- `cache::Union{BatchCache, Nothing} = nothing`: pre-allocated buffers
"""
function batch_solve(f_batch, lmo, X0::AbstractMatrix;
                     grad_batch=nothing,
                     max_iters::Int=10000, tol::Real=1e-4,
                     monotonic::Bool=true,
                     verbose::Bool=false,
                     cache::Union{BatchCache, Nothing}=nothing)
    grad_batch === nothing && throw(ArgumentError(
        "batch_solve requires grad_batch keyword argument. " *
        "Provide grad_batch=(G, X) -> ... that writes (n, B) gradients into G in-place."))
    oracle = lmo isa AbstractOracle ? lmo : FunctionOracle(lmo)
    return _batch_solve_core(f_batch, grad_batch, oracle, X0;
                              max_iters=max_iters, tol=tol,
                              monotonic=monotonic, verbose=verbose, cache=cache)
end

function _batch_solve_core(f_batch::F, ∇f_batch!::G, lmo::L, X0::AbstractMatrix;
                           max_iters::Int=10000, tol::Real=1e-4,
                           monotonic::Bool=true, verbose::Bool=false,
                           cache::Union{BatchCache, Nothing}=nothing) where {F, G, L<:AbstractOracle}
    X = copy(X0)
    T = eltype(X)
    n, B = size(X)

    c = something(cache, BatchCache(X0))

    # Reset mutable state
    c.active .= true
    c.discards .= 0

    # Initial objectives
    c.objective .= f_batch(X)
    fill!(c.gap, T(Inf))

    converged_flags = falses(B)
    final_iter = max_iters

    if max_iters <= 0
        ∇f_batch!(c.gradient, X)
        _batch_lmo_and_gap!(lmo, c, X)
        @inbounds for b in 1:B
            converged_flags[b] = c.gap[b] <= tol * (one(T) + abs(c.objective[b]))
        end
        return BatchSolveResult(X, BatchResult(copy(c.objective), copy(c.gap), 0, converged_flags, copy(c.discards)))
    end

    if verbose
        @printf("  %6s   %13s   %13s   %6s\n", "Iter", "Max Obj", "Max Gap", "Active")
        println("  ──────   ─────────────   ─────────────   ──────")
    end

    @inbounds for t in 0:(max_iters - 1)
        # 1. Batched gradient
        ∇f_batch!(c.gradient, X)

        # 2. Batched LMO + gap
        _batch_lmo_and_gap!(lmo, c, X)

        # 3. Convergence check
        all_done = true
        for b in 1:B
            if c.active[b] && c.gap[b] <= tol * (one(T) + abs(c.objective[b]))
                c.active[b] = false
                converged_flags[b] = true
            end
            all_done &= !c.active[b]
        end
        if all_done
            final_iter = t
            break
        end

        # 4. Step size (MonotonicStepSize: same scalar for all)
        γ = T(2) / T(t + 2)

        # 5. Trial update: x_trial[:,b] = (1-γ)*X[:,b] + γ*vertex[:,b]
        omγ = one(T) - γ
        for b in 1:B
            @simd for i in 1:n
                c.x_trial[i, b] = omγ * X[i, b] + γ * c.vertex[i, b]
            end
        end

        # 6. Objective evaluation
        obj_trial = f_batch(c.x_trial)

        # 7. Acceptance (masked)
        for b in 1:B
            c.active[b] || continue
            ot = obj_trial[b]
            if !isfinite(ot)
                c.discards[b] += 1
                continue
            end
            if monotonic && ot > c.objective[b] + n * eps(T) * max(one(T), abs(c.objective[b]))
                c.discards[b] += 1
                continue
            end
            @simd for i in 1:n
                X[i, b] = c.x_trial[i, b]
            end
            c.objective[b] = ot
        end

        if verbose && (t % 50 == 0 || t == max_iters - 1)
            max_obj = maximum(c.objective)
            max_gap = maximum(c.gap)
            n_active = count(c.active)
            @printf("  %6d   %13.6e   %13.4e   %6d\n", t, max_obj, max_gap, n_active)
        end
    end

    if verbose
        n_conv = count(converged_flags)
        if n_conv == B
            @printf("  All %d problems converged in %d iterations\n", B, final_iter)
        else
            @printf("  %d/%d problems converged after %d iterations\n", n_conv, B, max_iters)
        end
    end

    return BatchSolveResult(X, BatchResult(
        copy(c.objective), copy(c.gap), final_iter, converged_flags, copy(c.discards)))
end
