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

"""
Problem setup.

## Kwargs

- `D = Val(2)`: Physical dimension. This is wrapped in a `Val` to make it known at compile time.
- `n::Int`: Number of grid points in each dimension.
- `visc::Real`: Viscosity. This value is also used to infer the floating point type, so make sure it is a `Float32` or `Float64`.
- `backend = CPU()`: KernelAbstractions.jl backend. For Nvidia GPUs, do `using CUDA` and set to `CUDABackend()`.
"""
problem_setup(; D = Val(2), n, visc, backend = CPU(), workgroupsize = 64) =
    (; D, n, visc, backend, workgroupsize)

"Allocate empty scalar field."
scalarfield(setup) = KernelAbstractions.zeros(
    setup.backend,
    typeof(setup.visc),
    ntuple(Returns(setup.n), setup.D),
)

"Allocate empty vector field."
vectorfield(setup) = KernelAbstractions.zeros(
    setup.backend,
    typeof(setup.visc),
    ntuple(Returns(setup.n), setup.D)...,
    getval(setup.D),
)

"Allocate empty tensor field."
function tensorfield(setup)
    (; backend, D, n, visc) = setup
    d = getval(D)
    d2 = d * d
    T = typeof(visc)
    KernelAbstractions.zeros(backend, SMatrix{d,d,T,d2}, ntuple(Returns(n), D))
end

include("operators.jl")
include("time.jl")

end
