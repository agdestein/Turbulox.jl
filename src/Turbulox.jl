"Simulate turbulence in a box."
module Turbulox

using Adapt
using FFTW # For pressure solver and spectrum computation
using KernelAbstractions # For CPU/GPU kernels
using KernelAbstractions.Extras.LoopInfo: @unroll
using LinearAlgebra
using Random # For reproducibility
using StaticArrays # For tensors

"Get value from `Val`."
getval(::Val{x}) where {x} = x

"Staggered grid of order `o` and dimension `d`."
struct Grid{o,d}
    "Number of grid points in each dimension."
    n::Int
    Grid(; order = 2, dim = 2, n) = new{order,dim}(n)
end

"Get order of grid."
@inline order(::Grid{o,d}) where {o,d} = o

"Get physical dimension."
@inline dim(::Grid{o,d}) where {o,d} = d

"Get unit index in dimension `α`."
@inline e(g::Grid, α) = CartesianIndex(ntuple(β -> β == α, dim(g)))

# Extend index periodically so that it stays within the domain.
@inline (g::Grid)(i::Integer) = mod1(i, g.n)
@inline (g::Grid)(I::CartesianIndex) = CartesianIndex(mod1.(I.I, g.n))

"""
Problem setup.

## Kwargs

- `grid::Grid`: Grid setup.
- `visc::Real`: Viscosity. This value is also used to infer the floating point type, so make sure it is a `Float32` or `Float64`.
- `backend = CPU()`: KernelAbstractions.jl backend. For Nvidia GPUs, do `using CUDA` and set to `CUDABackend()`.
- `workgroupsize = 64`: Kernel work group size.
"""
problem_setup(; grid, visc, backend = CPU(), workgroupsize = 64) =
    (; grid, visc, backend, workgroupsize)

include("initializers.jl")
include("operators.jl")
include("tensors.jl")
include("closures.jl")
include("time.jl")

end
