"Get value from `Val`."
getval(::Val{x}) where {x} = x

"Staggered grid position."
struct Stag end

"Collocated grid position."
struct Coll end

"Staggered grid of order `o` and dimension `d`."
struct Grid{o,d,T,B}
    "Domain side length."
    L::T

    "Number of grid points in each dimension."
    n::Int

    """
    KernelAbstractions.jl hardware backend.
    For Nvidia GPUs, do `using CUDA` and set to `CUDABackend()`.
    """
    backend::B

    "Kernel work group size."
    workgroupsize::Int

    Grid(; order = 2, dim = 2, L, n, backend = CPU(), workgroupsize = 64) =
        new{order,dim,typeof(L),typeof(backend)}(L, n, backend, workgroupsize)
end

"Get order of grid."
@inline order(::Grid{o,d}) where {o,d} = o

"Get physical dimension."
@inline dim(::Grid{o,d}) where {o,d} = d

"Get grid spacing."
@inline dx(g::Grid) = g.L / g.n

"Get unit index in dimension `i`."
@inline e(grid, i) = CartesianIndex(ntuple(==(i), dim(grid)))

# Extend index periodically so that it stays within the domain.
@inline (g::Grid)(i::Integer) = mod1(i, g.n)
@inline (g::Grid)(x::CartesianIndex) = CartesianIndex(mod1.(x.I, g.n))

"Get grid points along axis."
get_axis(g::Grid, ::Stag) = range(0, g.L, g.n + 1)[2:end]
get_axis(g::Grid, ::Coll) = range(0, g.L, g.n + 1)[2:end] .- dx(g) / 2

"""
Apply `kernel!` on `grid, args...` over the entire domain.
The `args` are typically input and output fields.
The kernel should be of the form
```julia
using KernelAbstractions
@kernel function kernel!(grid, args...)
    # content
end
```
"""
function apply!(kernel!, g::Grid, args...; ndrange = default_ndrange(g))
    (; backend, workgroupsize) = g
    kernel!(backend, workgroupsize)(g, args...; ndrange)
    KernelAbstractions.synchronize(backend)
    nothing
end

default_ndrange(g::Grid) = ntuple(Returns(g.n), dim(g))
