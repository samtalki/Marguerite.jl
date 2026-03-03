# Marguerite C API

C shared library for the Marguerite.jl Frank-Wolfe solver, built with [JuliaC.jl](https://github.com/JuliaLang/JuliaC.jl).

Full documentation: [C API reference](https://samtalki.github.io/Marguerite.jl/dev/c-api/)

## Building

Requires Julia 1.12+.

```bash
julia --project=lib -e 'using Pkg; Pkg.instantiate()'
julia --project=lib lib/build.jl
```

Output lands in `lib/build/lib/libmarguerite.{so,dylib,dll}`.

## Testing

```bash
cd lib
cc -o test_libmarguerite test_libmarguerite.c -ldl -lm
./test_libmarguerite ./build/lib/libmarguerite.so
```
