#!/usr/bin/env julia
#
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
#
# Build libmarguerite shared library using JuliaC.
#
# Usage:
#   julia --project=lib lib/build.jl
#
# Output lands in lib/build/{lib,bin}/libmarguerite.{so,dylib,dll}

using JuliaC

const LIB_DIR = @__DIR__
const BUILD_DIR = joinpath(LIB_DIR, "build")
mkpath(BUILD_DIR)

img = ImageRecipe(
    file       = joinpath(LIB_DIR, "src", "libmarguerite.jl"),
    project    = LIB_DIR,
    output_type = "--output-lib",
    add_ccallables = true,
    trim_mode  = "unsafe",
    verbose    = true,
    julia_args = ["--experimental"],
)

link = LinkRecipe(
    image_recipe = img,
    outname      = joinpath(BUILD_DIR, "libmarguerite"),
    rpath        = JuliaC.RPATH_BUNDLE,
)

bun = BundleRecipe(
    link_recipe = link,
    output_dir  = BUILD_DIR,
)

println("── Compiling...")
compile_products(img)

println("── Linking...")
link_products(link)

println("── Bundling...")
bundle_products(bun)

dlext = Base.BinaryPlatforms.platform_dlext()
libroot = Sys.iswindows() ? "bin" : "lib"
libpath = joinpath(BUILD_DIR, libroot, "libmarguerite." * dlext)

if isfile(libpath)
    println("── Built: $libpath")
else
    error("Build failed — library not found at $libpath")
end
