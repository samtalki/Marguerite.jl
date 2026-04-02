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

import MathOptInterface as MOI
import SparseArrays: sparse, spzeros, SparseMatrixCSC

# ------------------------------------------------------------------
# Solution storage
# ------------------------------------------------------------------

mutable struct _MOISolution
    x::Vector{Float64}
    objective::Float64
    gap::Float64
    iterations::Int
    converged::Bool
    solve_time_sec::Float64

    _MOISolution() = new(Float64[], NaN, NaN, 0, false, 0.0)
end

# ------------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------------

"""
    Optimizer()

MathOptInterface optimizer for the Marguerite Frank-Wolfe solver.

Solves convex problems of the form:

```math
\\min_{x \\in C} \\frac{1}{2} x^T Q x + c^T x
```

where ``C`` is a constraint set supported by a Marguerite oracle
(Box, Simplex, ProbSimplex, WeightedSimplex).

# Usage with JuMP

```julia
using JuMP, Marguerite
model = Model(Marguerite.Optimizer)
set_attribute(model, "tol", 1e-6)
set_attribute(model, "max_iters", 5000)
@variable(model, 0 <= x[1:3] <= 1)
@objective(model, Min, x[1]^2 + x[2]^2 + x[3]^2 - x[1])
optimize!(model)
```
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    # Problem data
    Q::Union{Nothing, SparseMatrixCSC{Float64,Int}}
    c::Vector{Float64}
    objective_constant::Float64
    sense::MOI.OptimizationSense
    oracle::Union{Nothing, AbstractOracle}
    n::Int

    # Solver settings
    max_iters::Int
    tol::Float64
    monotonic::Bool
    verbose::Bool

    # Solution
    sol::_MOISolution

    function Optimizer(; kwargs...)
        opt = new(
            nothing, Float64[], 0.0, MOI.MIN_SENSE, nothing, 0,
            10000, 1e-4, true, false,
            _MOISolution())
        for (key, val) in kwargs
            MOI.set(opt, MOI.RawOptimizerAttribute(String(key)), val)
        end
        return opt
    end
end

# ------------------------------------------------------------------
# Basic optimizer interface
# ------------------------------------------------------------------

MOI.get(::Optimizer, ::MOI.SolverName) = "Marguerite"
MOI.get(::Optimizer, ::MOI.SolverVersion) = "0.1.3"

function MOI.is_empty(opt::Optimizer)
    return opt.oracle === nothing && opt.n == 0
end

function MOI.empty!(opt::Optimizer)
    opt.Q = nothing
    opt.c = Float64[]
    opt.objective_constant = 0.0
    opt.sense = MOI.MIN_SENSE
    opt.oracle = nothing
    opt.n = 0
    opt.sol = _MOISolution()
    return
end

function MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{F}) where {F}
    return F <: Union{MOI.ScalarAffineFunction{Float64}, MOI.ScalarQuadraticFunction{Float64}}
end

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true

# ------------------------------------------------------------------
# Supported constraint types
# ------------------------------------------------------------------

# We support variable bounds (Interval, GreaterThan, LessThan, EqualTo)
# and a single linear constraint (for simplex detection)
function MOI.supports_constraint(::Optimizer,
    ::Type{MOI.VariableIndex}, ::Type{S}) where {S<:Union{
        MOI.GreaterThan{Float64}, MOI.LessThan{Float64},
        MOI.Interval{Float64}, MOI.EqualTo{Float64}}}
    return true
end

function MOI.supports_constraint(::Optimizer,
    ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{S}) where {S<:Union{
        MOI.EqualTo{Float64}, MOI.LessThan{Float64}, MOI.GreaterThan{Float64}}}
    return true
end

# ------------------------------------------------------------------
# Attribute setting
# ------------------------------------------------------------------

function MOI.set(opt::Optimizer, ::MOI.Silent, value::Bool)
    opt.verbose = !value
    return
end

MOI.get(opt::Optimizer, ::MOI.Silent) = !opt.verbose

function MOI.set(opt::Optimizer, param::MOI.RawOptimizerAttribute, value)
    name = Symbol(param.name)
    if name === :tol
        opt.tol = Float64(value)
    elseif name === :max_iters
        opt.max_iters = Int(value)
    elseif name === :monotonic
        opt.monotonic = Bool(value)
    elseif name === :verbose
        opt.verbose = Bool(value)
    else
        throw(MOI.UnsupportedAttribute(param))
    end
    return
end

function MOI.get(opt::Optimizer, param::MOI.RawOptimizerAttribute)
    name = Symbol(param.name)
    name === :tol && return opt.tol
    name === :max_iters && return opt.max_iters
    name === :monotonic && return opt.monotonic
    name === :verbose && return opt.verbose
    throw(MOI.UnsupportedAttribute(param))
end

# ------------------------------------------------------------------
# Two-level optimize! dispatch (following Moreau.jl pattern)
# ------------------------------------------------------------------

function MOI.optimize!(dest::Optimizer, src::MOI.ModelLike)
    # Copy model to our optimizer, then solve
    index_map = MOI.Utilities.identity_index_map(src)

    # Extract problem dimensions
    vis = MOI.get(src, MOI.ListOfVariableIndices())
    n = length(vis)
    n > 0 || throw(MOI.EmptyModel())
    dest.n = n

    # Extract objective
    dest.sense = MOI.get(src, MOI.ObjectiveSense())
    _extract_objective!(dest, src)

    # Extract constraints → oracle
    dest.oracle = _detect_oracle(src, vis)

    # Solve
    _do_solve!(dest)

    return index_map, false
end

# ------------------------------------------------------------------
# Objective extraction
# ------------------------------------------------------------------

function _extract_objective!(opt::Optimizer, src::MOI.ModelLike)
    n = opt.n
    obj_type = MOI.get(src, MOI.ObjectiveFunctionType())

    if obj_type <: MOI.ScalarQuadraticFunction{Float64}
        obj = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}())
        # Build Q (symmetric, stored as full matrix for mul!)
        I_idx = Int[]
        J_idx = Int[]
        V_val = Float64[]
        for qt in obj.quadratic_terms
            i = qt.variable_1.value
            j = qt.variable_2.value
            coeff = qt.coefficient  # MOI stores 2*Q[i,j] for i≠j, Q[i,i] for i==j
            if i == j
                push!(I_idx, i); push!(J_idx, j); push!(V_val, coeff)
            else
                push!(I_idx, i); push!(J_idx, j); push!(V_val, coeff / 2)
                push!(I_idx, j); push!(J_idx, i); push!(V_val, coeff / 2)
            end
        end
        opt.Q = isempty(I_idx) ? nothing : sparse(I_idx, J_idx, V_val, n, n)
        # Build c
        opt.c = zeros(n)
        for at in obj.affine_terms
            opt.c[at.variable.value] += at.coefficient
        end
        opt.objective_constant = obj.constant

    elseif obj_type <: MOI.ScalarAffineFunction{Float64}
        obj = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
        opt.Q = nothing
        opt.c = zeros(n)
        for at in obj.terms
            opt.c[at.variable.value] += at.coefficient
        end
        opt.objective_constant = obj.constant

    else
        throw(MOI.UnsupportedAttribute(MOI.ObjectiveFunction{obj_type}(),
            "Marguerite.Optimizer supports ScalarAffineFunction and ScalarQuadraticFunction objectives only."))
    end

    # Negate for MAX sense
    if opt.sense == MOI.MAX_SENSE
        if opt.Q !== nothing
            opt.Q = -opt.Q
        end
        opt.c = -opt.c
    end
end

# ------------------------------------------------------------------
# Constraint detection → Marguerite oracle
# ------------------------------------------------------------------

function _detect_oracle(src::MOI.ModelLike, vis::Vector{MOI.VariableIndex})
    n = length(vis)

    # Collect variable bounds
    lb = fill(-Inf, n)
    ub = fill(Inf, n)
    _collect_variable_bounds!(lb, ub, src, vis)

    # Collect scalar affine constraints
    affine_constraints = _collect_affine_constraints(src, n)

    # Detect oracle from bounds + affine constraints
    return _build_oracle(lb, ub, affine_constraints, n)
end

function _collect_variable_bounds!(lb, ub, src::MOI.ModelLike, vis)
    # Interval bounds
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.Interval{Float64}}())
        set = MOI.get(src, MOI.ConstraintSet(), ci)
        f = MOI.get(src, MOI.ConstraintFunction(), ci)
        i = f.value
        lb[i] = max(lb[i], set.lower)
        ub[i] = min(ub[i], set.upper)
    end

    # GreaterThan bounds
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.GreaterThan{Float64}}())
        set = MOI.get(src, MOI.ConstraintSet(), ci)
        f = MOI.get(src, MOI.ConstraintFunction(), ci)
        lb[f.value] = max(lb[f.value], set.lower)
    end

    # LessThan bounds
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.LessThan{Float64}}())
        set = MOI.get(src, MOI.ConstraintSet(), ci)
        f = MOI.get(src, MOI.ConstraintFunction(), ci)
        ub[f.value] = min(ub[f.value], set.upper)
    end

    # EqualTo (fixed variables)
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VariableIndex, MOI.EqualTo{Float64}}())
        set = MOI.get(src, MOI.ConstraintSet(), ci)
        f = MOI.get(src, MOI.ConstraintFunction(), ci)
        lb[f.value] = set.value
        ub[f.value] = set.value
    end
end

struct _AffineConstraint
    coeffs::Vector{Float64}    # length n
    sense::Symbol              # :eq, :leq, :geq
    rhs::Float64
end

function _collect_affine_constraints(src::MOI.ModelLike, n::Int)
    constraints = _AffineConstraint[]

    for (F, S) in [(MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}),
                    (MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}),
                    (MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64})]
        for ci in MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
            f = MOI.get(src, MOI.ConstraintFunction(), ci)
            set = MOI.get(src, MOI.ConstraintSet(), ci)
            coeffs = zeros(n)
            for term in f.terms
                coeffs[term.variable.value] += term.coefficient
            end
            rhs_val = if S <: MOI.EqualTo
                set.value - f.constant
            elseif S <: MOI.LessThan
                set.upper - f.constant
            else
                set.lower - f.constant
            end
            sense_sym = S <: MOI.EqualTo ? :eq : (S <: MOI.LessThan ? :leq : :geq)
            push!(constraints, _AffineConstraint(coeffs, sense_sym, rhs_val))
        end
    end
    return constraints
end

function _build_oracle(lb, ub, affine_constraints, n)
    has_finite_lb = all(isfinite, lb)
    has_finite_ub = all(isfinite, ub)

    if isempty(affine_constraints)
        # Pure box constraints
        has_finite_lb && has_finite_ub || throw(ArgumentError(
            "Marguerite.Optimizer: all variables must have finite bounds when no affine constraints are present."))
        # Check if all bounds are identical → ScalarBox
        if all(lb[i] == lb[1] for i in 1:n) && all(ub[i] == ub[1] for i in 1:n)
            return ScalarBox{Float64}(lb[1], ub[1])
        end
        return Box(lb, ub)
    end

    if length(affine_constraints) == 1
        ac = affine_constraints[1]
        all_nonneg = all(lb[i] >= 0.0 for i in 1:n)
        all_lb_zero = all(lb[i] == 0.0 for i in 1:n)
        all_ub_inf = all(ub[i] == Inf for i in 1:n)
        is_unit_sum = all(ac.coeffs[i] ≈ 1.0 for i in 1:n)

        # ProbSimplex: x >= 0, sum(x) = r
        if all_lb_zero && all_ub_inf && is_unit_sum && ac.sense == :eq
            return ProbSimplex(ac.rhs)
        end

        # Capped Simplex: x >= 0, sum(x) <= r
        if all_lb_zero && all_ub_inf && is_unit_sum && ac.sense == :leq
            return Simplex(ac.rhs)
        end

        # WeightedSimplex: x >= lb, α'x <= β
        if has_finite_lb && all_ub_inf && ac.sense == :leq && all(ac.coeffs[i] > 0 for i in 1:n)
            return WeightedSimplex(ac.coeffs, ac.rhs, lb)
        end
    end

    throw(ArgumentError(
        "Marguerite.Optimizer: constraint pattern not recognized as a supported oracle.\n" *
        "Supported patterns:\n" *
        "  - Variable bounds only → Box\n" *
        "  - x ≥ 0, Σx = r → ProbSimplex\n" *
        "  - x ≥ 0, Σx ≤ r → Simplex\n" *
        "  - x ≥ lb, α'x ≤ β → WeightedSimplex\n" *
        "For arbitrary constraint sets, use Marguerite.solve() directly with a custom oracle."))
end

# ------------------------------------------------------------------
# Solve
# ------------------------------------------------------------------

function _do_solve!(opt::Optimizer)
    n = opt.n
    Q = opt.Q
    c = opt.c
    oracle = opt.oracle::AbstractOracle

    # Build objective and gradient closures
    if Q !== nothing
        f = let Q=Q, c=c
            x -> 0.5 * dot(x, Q * x) + dot(c, x)
        end
        grad! = let Q=Q, c=c
            (g, x) -> (mul!(g, Q, x); g .+= c; g)
        end
    else
        f = let c=c
            x -> dot(c, x)
        end
        grad! = let c=c
            (g, x) -> (copyto!(g, c); g)
        end
    end

    # Generate feasible initial point
    x0 = _initial_point(oracle, n)

    # Solve
    t0 = time()
    x, result = solve(f, oracle, x0;
        grad=grad!, max_iters=opt.max_iters, tol=opt.tol,
        monotonic=opt.monotonic, verbose=opt.verbose)
    solve_time = time() - t0

    # Store solution (un-negate for MAX)
    opt.sol.x = x
    obj_val = result.objective
    if opt.sense == MOI.MAX_SENSE
        obj_val = -obj_val
    end
    opt.sol.objective = obj_val + opt.objective_constant
    opt.sol.gap = result.gap
    opt.sol.iterations = result.iterations
    opt.sol.converged = result.converged
    opt.sol.solve_time_sec = solve_time
end

function _initial_point(oracle::ScalarBox{T}, n) where T
    fill((oracle.lb + oracle.ub) / 2, n)
end

function _initial_point(oracle::Box{T}, n) where T
    (oracle.lb .+ oracle.ub) ./ 2
end

function _initial_point(oracle::Simplex{T, Equality}, n) where {T, Equality}
    fill(T(oracle.r) / n, n)
end

function _initial_point(oracle::WeightedSimplex{T}, n) where T
    copy(oracle.lb)
end

function _initial_point(::AbstractOracle, n)
    fill(1.0 / n, n)
end

# ------------------------------------------------------------------
# Result queries
# ------------------------------------------------------------------

function MOI.get(opt::Optimizer, ::MOI.TerminationStatus)
    opt.sol.x === nothing || isempty(opt.sol.x) && return MOI.OPTIMIZE_NOT_CALLED
    opt.sol.converged && return MOI.OPTIMAL
    return MOI.ITERATION_LIMIT
end

function MOI.get(opt::Optimizer, attr::MOI.PrimalStatus)
    attr.result_index == 1 || return MOI.NO_SOLUTION
    isempty(opt.sol.x) && return MOI.NO_SOLUTION
    opt.sol.converged && return MOI.FEASIBLE_POINT
    return MOI.NEARLY_FEASIBLE_POINT
end

MOI.get(opt::Optimizer, ::MOI.DualStatus) = MOI.NO_SOLUTION
MOI.get(opt::Optimizer, ::MOI.ResultCount) = isempty(opt.sol.x) ? 0 : 1

function MOI.get(opt::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(opt, attr)
    return opt.sol.objective
end

function MOI.get(opt::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(opt, attr)
    return opt.sol.x[vi.value]
end

function MOI.get(opt::Optimizer, attr::MOI.VariablePrimal, vis::Vector{MOI.VariableIndex})
    MOI.check_result_index_bounds(opt, attr)
    return [opt.sol.x[vi.value] for vi in vis]
end

MOI.get(opt::Optimizer, ::MOI.SolveTimeSec) = opt.sol.solve_time_sec
MOI.get(opt::Optimizer, ::MOI.RawStatusString) = opt.sol.converged ? "converged" : "iteration_limit"

function MOI.get(opt::Optimizer, ::MOI.NumberOfVariables)
    return opt.n
end
