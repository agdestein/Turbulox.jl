"Get value from `Val`."
getval(::Val{x}) where {x} = x

"Staggered grid position."
struct Stag end

"Collocated grid position."
struct Coll end

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

"Get unit index in dimension `i`."
@inline e(grid, i) = CartesianIndex(ntuple(==(i), dim(grid)))

# Extend index periodically so that it stays within the domain.
@inline (g::Grid)(i::Integer) = mod1(i, g.n)
@inline (g::Grid)(x::CartesianIndex) = CartesianIndex(mod1.(x.I, g.n))

get_axis(grid, ::Stag) = range(0, 1, grid.n+1)[2:end]
get_axis(grid, ::Coll) = range(0, 1, grid.n+1)[2:end] .- 1 / 2 / grid.n

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

"""
Apply `kernel!` on `setup.grid, args...` over the entire domain.
The `args` are typically input and output fields.
The kernel should be of the form
```julia
using KernelAbstractions
@kernel function kernel!(grid, args...)
    # content
end
```
"""
function apply!(kernel!, setup, args...; ndrange = nothing)
    (; grid, backend, workgroupsize) = setup
    if isnothing(ndrange)
        ndrange = ntuple(Returns(grid.n), dim(grid))
    end
    kernel!(backend, workgroupsize)(grid, args...; ndrange)
    KernelAbstractions.synchronize(backend)
    nothing
end
