"Get value from `Val`."
@inline getval(::Val{x}) where {x} = x

"Staggered grid position."
struct Stag end

"Collocated grid position."
struct Coll end

"Staggered grid of half order `ho`."
@kwdef struct Grid{H,T,B}
    "Number of terms in discretization (half of order). Default is `Val(1)`."
    ho::Val{H} = Val(1)

    "Domain side length."
    L::T

    "Number of grid points in each dimension."
    n::Int

    """
    KernelAbstractions.jl hardware backend.
    For Nvidia GPUs, do `using CUDA` and set to `CUDABackend()`.
    """
    backend::B = CPU()

    "Kernel work group size."
    workgroupsize::Int = 64
end

"Get order of grid."
@inline order(g::Grid) = 2 * getval(g.ho)

"Get grid spacing."
@inline dx(g::Grid) = g.L / g.n

"Get unit index in dimension `i`."
@inline e(i) = CartesianIndex(ntuple(==(i), 3))

# Extend index periodically so that it stays within the domain.
@inline (g::Grid)(i::Integer) = mod1(i, g.n)
# @inline (g::Grid)(x::CartesianIndex) = CartesianIndex(map(g, x.I))

"Get grid points along axis."
get_axis(g::Grid, ::Stag) = range(0, g.L, g.n + 1)[2:end]
get_axis(g::Grid, ::Coll) = range(0, g.L, g.n + 1)[2:end] .- dx(g) / 2

"Scalar field."
struct ScalarField{T,G,D} <: AbstractArray{T,3}
    grid::G
    data::D
    ScalarField(
        grid,
        data = KernelAbstractions.zeros(
            grid.backend,
            typeof(grid.L),
            grid.n,
            grid.n,
            grid.n,
        ),
    ) = new{eltype(data),typeof(grid),typeof(data)}(grid, data)
end

"Staggered vector field."
struct VectorField{T,G,D} <: AbstractArray{T,4}
    grid::G
    data::D
    VectorField(
        grid,
        data = KernelAbstractions.zeros(
            grid.backend,
            typeof(grid.L),
            grid.n,
            grid.n,
            grid.n,
            3,
        ),
    ) = new{eltype(data),typeof(grid),typeof(data)}(grid, data)
end

"Staggered tensor field."
struct TensorField{T,G,D} <: AbstractArray{T,5}
    grid::G
    data::D
    TensorField(
        grid,
        data = KernelAbstractions.zeros(
            grid.backend,
            typeof(grid.L),
            grid.n,
            grid.n,
            grid.n,
            3,
            3,
        ),
    ) = new{eltype(data),typeof(grid),typeof(data)}(grid, data)
end

Base.size(u::ScalarField) = size(u.data)
Base.size(u::VectorField) = size(u.data)
Base.size(u::TensorField) = size(u.data)

Base.getindex(u::ScalarField, i::Int) = u.data[i]
Base.getindex(u::VectorField, i::Int) = u.data[i]
Base.getindex(u::TensorField, i::Int) = u.data[i]

Base.getindex(u::ScalarField, I::Vararg{Int,3}) = u.data[map(u.grid, I)...]
Base.getindex(u::VectorField, I::Vararg{Int,4}) = u.data[map(u.grid, I[1:3])..., I[4]]
Base.getindex(u::TensorField, I::Vararg{Int,5}) = u.data[map(u.grid, I[1:3])..., I[4], I[5]]

Base.setindex!(u::ScalarField, val, i::Int) = setindex!(u.data, val, i)
Base.setindex!(u::VectorField, val, i::Int) = setindex!(u.data, val, i)
Base.setindex!(u::TensorField, val, i::Int) = setindex!(u.data, val, i)

Base.setindex!(u::ScalarField, val, I::Vararg{Int,3}) =
    setindex!(u.data, val, map(u.grid, I)...)
Base.setindex!(u::VectorField, val, I::Vararg{Int,4}) =
    setindex!(u.data, val, map(u.grid, I[1:3])..., I[4])
Base.setindex!(u::TensorField, val, I::Vararg{Int,5}) =
    setindex!(u.data, val, map(u.grid, I[1:3])..., I[4], I[5])

function Base.similar(u::ScalarField{T}, ::Type{T}, dims::Vararg{Int,3}) where {T}
    @assert dims == size(u.data) "Scalar field must have same size as grid."
    ScalarField(u.grid, similar(u.data))
end
function Base.similar(u::VectorField{T}, ::Type{T}, dims::Vararg{Int,4}) where {T}
    @assert dims == size(u.data) "Vector field must have same size as grid."
    VectorField(u.grid, similar(u.data))
end
function Base.similar(u::TensorField{T}, ::Type{T}, dims::Vararg{Int,5}) where {T}
    @assert dims == size(u.data) "Tensor field must have same size as grid."
    TensorField(u.grid, similar(u.data))
end

"""
Apply `kernel!` on `args...` over the entire domain.
The `args` are typically input and output fields.
The kernel should be of the form
```julia
using KernelAbstractions
@kernel function kernel!(args...)
    # content
end
```
"""
function apply!(kernel!, g::Grid, args...; ndrange = default_ndrange(g))
    (; backend, workgroupsize) = g
    kernel!(backend, workgroupsize)(args...; ndrange)
    KernelAbstractions.synchronize(backend)
    nothing
end

default_ndrange(g::Grid) = g.n, g.n, g.n
