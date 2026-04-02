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

# Shared test utilities for verification tests.

using LinearAlgebra: dot, mul!, I
using Test: @test, isapprox

"""
    random_qp_data(rng, n; epsilon=0.1)

Generate a random QP with positive-definite Hessian Q and linear term c.
Returns `(Q, c)` where `Q = A'A + εI`.
"""
function random_qp_data(rng, n; epsilon=0.1)
    A = randn(rng, n, n)
    Q = A' * A + epsilon * I
    c = randn(rng, n)
    return Q, c
end

"""
    make_qp(Q, c)

Returns `(f, ∇f!)` for the QP `min 0.5 x'Qx + c'x`.
"""
function make_qp(Q, c)
    f(x) = 0.5 * dot(x, Q, x) + dot(c, x)
    function ∇f!(g, x)
        mul!(g, Q, x)
        g .+= c
        return g
    end
    return f, ∇f!
end

"""
    check_match(f, x_fw, x_jump, obj_jump; obj_atol, x_atol)

Assert that Frank-Wolfe solution matches a JuMP reference.
"""
function check_match(f, x_fw, x_jump, obj_jump; obj_atol, x_atol)
    @test isapprox(f(x_fw), obj_jump; atol=obj_atol)
    @test isapprox(x_fw, x_jump; atol=x_atol)
end
