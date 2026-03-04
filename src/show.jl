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

# ------------------------------------------------------------------
# Pretty printing for user-facing types
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Result
# ------------------------------------------------------------------

function Base.show(io::IO, r::Result)
    print(io, "Result(")
    @printf(io, "f=%.4e, gap=%.4e", r.objective, r.gap)
    print(io, ", iters=", r.iterations)
    print(io, r.converged ? ", converged" : ", not converged")
    r.discards > 0 && print(io, ", discards=", r.discards)
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", r::Result{T}) where T
    println(io, "Frank-Wolfe Result")
    @printf(io, "  objective:  %.6e\n", r.objective)
    @printf(io, "  FW gap:     %.4e\n", r.gap)
    println(io, "  iterations: ", r.iterations)
    println(io, "  converged:  ", r.converged)
    r.discards > 0 && print(io, "  discards:   ", r.discards)
end

# ------------------------------------------------------------------
# CGResult
# ------------------------------------------------------------------

function Base.show(io::IO, r::CGResult)
    print(io, "CGResult(")
    print(io, "iters=", r.iterations)
    @printf(io, ", res=%.4e", r.residual_norm)
    print(io, r.converged ? ", converged" : ", not converged")
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", r::CGResult{T}) where T
    println(io, "CG Result")
    println(io, "  iterations:    ", r.iterations)
    @printf(io, "  residual norm: %.4e\n", r.residual_norm)
    print(io, "  converged:     ", r.converged)
end

# ------------------------------------------------------------------
# Step size rules
# ------------------------------------------------------------------

function Base.show(io::IO, ::MonotonicStepSize)
    print(io, "MonotonicStepSize (γₜ = 2/(t+2))")
end

function Base.show(io::IO, s::AdaptiveStepSize)
    @printf(io, "AdaptiveStepSize(L=%.4g, η=%.4g)", s.L, s.η)
end

# ------------------------------------------------------------------
# Oracles
# ------------------------------------------------------------------

function Base.show(io::IO, lmo::Simplex{T, Equality}) where {T, Equality}
    op = Equality ? "=" : "≤"
    name = Equality ? "ProbSimplex" : "Simplex"
    print(io, name, "{", T, "} {x ≥ 0 : ∑xᵢ ", op, " ", lmo.r, "}")
end

function Base.show(io::IO, lmo::Box{T}) where T
    n = length(lmo.lb)
    if n ≤ 4
        print(io, "Box{", T, "} ", lmo.lb, " ≤ x ≤ ", lmo.ub)
    else
        print(io, "Box{", T, "}(dim=", n, ")")
    end
end

function Base.show(io::IO, lmo::Knapsack)
    n = length(lmo.perm)
    print(io, "Knapsack(budget=", lmo.k, ", dim=", n, ")")
end

function Base.show(io::IO, lmo::MaskedKnapsack)
    n = length(lmo.is_masked)
    m = count(lmo.is_masked)
    total = lmo.k + m
    print(io, "MaskedKnapsack(budget=", total, ", masked=", m, ", dim=", n, ")")
end

function Base.show(io::IO, lmo::WeightedSimplex{T}) where T
    n = length(lmo.α)
    print(io, "WeightedSimplex{", T, "}(dim=", n, ", budget=", lmo.β, ")")
end

# ------------------------------------------------------------------
# ActiveConstraints
# ------------------------------------------------------------------

function Base.show(io::IO, ac::ActiveConstraints{T}) where T
    nb = length(ac.bound_indices)
    nf = length(ac.free_indices)
    ne = length(ac.eq_normals)
    print(io, "ActiveConstraints{", T, "}: ", nb, " bound, ", nf, " free, ", ne, " equality")
end
