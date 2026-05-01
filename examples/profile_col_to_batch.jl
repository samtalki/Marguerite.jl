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

# Profile the bilevel pullback allocation pattern (PR-B precondition)
#
# Hypothesis (from validation review): `_col_to_batch` allocates an (n, B)
# zero-matrix per per-column gradient call inside the rrule pullback. For
# moderate B and n, this dominates the pullback wall time and obscures any
# GPU speedup we'd expect to measure in Experiment 1.

using Marguerite
using Marguerite: solve, batch_solve, batch_bilevel_solve, ProbSimplex
using LinearAlgebra
using Random: Xoshiro
using ChainRulesCore: rrule
using Printf

function make_problem(n, B; seed=42)
    rng = Xoshiro(seed)
    A = randn(rng, n, n)
    Q = (A'A) ./ n + 0.1 * I
    Q = Matrix(Q)
    target = randn(rng, n, B)
    target ./= sum(target, dims=1)  # simplex-ish
    X0 = fill(1.0/n, n, B)

    f_param = (X, θ) -> [0.5 * dot(X[:, b], Q * X[:, b]) - dot(θ, X[:, b]) for b in 1:B]
    grad_param! = (G, X, θ) -> (G .= Q * X .- θ)
    return Q, target, X0, f_param, grad_param!
end

function profile_pullback(n, B)
    Q, target, X0, f_param, grad_param! = make_problem(n, B)
    θ = randn(n)
    lmo = ProbSimplex()

    println("\n=== n=$n, B=$B ===")
    # Forward solve
    Y, _ = batch_solve(f_param, lmo, X0, θ; grad_batch=grad_param!,
                      max_iters=100, tol=1e-3)
    print("Forward only:                ")
    @time batch_solve(f_param, lmo, X0, θ; grad_batch=grad_param!,
                      max_iters=100, tol=1e-3)

    # rrule + pullback
    print("rrule build (incl forward):  ")
    @time br, pb = rrule(batch_solve, f_param, lmo, X0, θ;
                         grad_batch=grad_param!,
                         max_iters=100, tol=1e-3)
    X2, _ = br  # BatchSolveResult unpacks to (X, result)
    dY = randn(size(X2))
    pb(dY)  # warmup
    print("Pullback call:               ")
    @time pb(dY)
    print("Pullback call (2nd):         ")
    @time pb(dY)
end

profile_pullback(100, 8)
profile_pullback(100, 64)
profile_pullback(1000, 16)
