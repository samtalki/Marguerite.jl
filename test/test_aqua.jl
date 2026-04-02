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

using Aqua, Marguerite, Test

@testset "Aqua.jl package quality" begin
    Aqua.test_all(Marguerite;
        ambiguities=true,
        unbound_args=true,
        undefined_exports=true,
        project_extras=true,
        stale_deps=true,
        deps_compat=true,
        piracies=true,
    )
end
