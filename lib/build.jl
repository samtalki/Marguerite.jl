#!/usr/bin/env julia
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
    trim_mode  = "unsafe-warn",
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
