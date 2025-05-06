"Get value from `Val`."
@inline getval(::Val{x}) where {x} = x

"1D staggered grid position (at the right boundary of a volume in a given direction)."
struct Stag end

"1D collocated grid position (at the volume center in a given direction)."
struct Coll end

"Position in a 3D finite volume."
struct Position{P1,P2,P3} end
const Center = Position{Coll,Coll,Coll}
const Face1 = Position{Stag,Coll,Coll}
const Face2 = Position{Coll,Stag,Coll}
const Face3 = Position{Coll,Coll,Stag}
const Edge1 = Position{Coll,Stag,Stag}
const Edge2 = Position{Stag,Coll,Stag}
const Edge3 = Position{Stag,Stag,Coll}
const Corner = Position{Stag,Stag,Stag}

const X = Val{1}
const Y = Val{2}
const Z = Val{3}

Base.getindex(::Position{P1,P2,P3}, ::X) where {P1,P2,P3} = P1()
Base.getindex(::Position{P1,P2,P3}, ::Y) where {P1,P2,P3} = P2()
Base.getindex(::Position{P1,P2,P3}, ::Z) where {P1,P2,P3} = P3()

@inline directions() = Val(1), Val(2), Val(3)

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

Base.show(io::IO, g::Grid) =
    print(io, "Grid($(g.ho), $(g.L), $(g.n), $(g.backend), $(g.workgroupsize))")

"Get order of grid."
@inline order(g::Grid) = 2 * getval(g.ho)

"Get grid spacing."
@inline dx(g::Grid) = g.L / g.n

"Get unit index in dimension `i`."
@inline e(::Val{1}) = CartesianIndex(1, 0, 0)
@inline e(::Val{2}) = CartesianIndex(0, 1, 0)
@inline e(::Val{3}) = CartesianIndex(0, 0, 1)

# Extend index periodically so that it stays within the domain.
@inline (g::Grid)(i::Integer) = mod1(i, g.n)
# @inline (g::Grid)(x::CartesianIndex) = CartesianIndex(map(g, x.I))

"Get grid points along axis."
get_axis(g::Grid, ::Stag) = range(0, g.L, g.n + 1)[2:end]
get_axis(g::Grid, ::Coll) = range(0, g.L, g.n + 1)[2:end] .- dx(g) / 2

"Scalar field."
struct ScalarField{T,P,G,D} <: AbstractArray{T,3}
    position::P
    grid::G
    data::D
    ScalarField(
        position::Position,
        grid::Grid,
        data = KernelAbstractions.zeros(
            grid.backend,
            typeof(grid.L),
            grid.n,
            grid.n,
            grid.n,
        ),
    ) = new{eltype(data),typeof(position),typeof(grid),typeof(data)}(position, grid, data)
    ScalarField(g::Grid, args...) = ScalarField(Center(), g, args...)
end

"Vector field."
struct VectorField{T,G,D} <: AbstractArray{T,4}
    grid::G
    data::D
    VectorField(
        grid::Grid,
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

"Tensor field."
struct TensorField{T,G,D} <: AbstractArray{T,5}
    grid::G
    data::D
    TensorField(
        grid::Grid,
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

Adapt.adapt_structure(to, u::ScalarField) =
    ScalarField(u.position, u.grid, adapt(to, u.data))
Adapt.adapt_structure(to, u::VectorField) = VectorField(u.grid, adapt(to, u.data))
Adapt.adapt_structure(to, u::TensorField) = TensorField(u.grid, adapt(to, u.data))

Base.size(u::ScalarField) = size(u.data)
Base.size(u::VectorField) = size(u.data)
Base.size(u::TensorField) = size(u.data)

Base.show(io::IO, u::ScalarField) =
    print(io, "ScalarField($(u.position), $(u.grid), ::$(typeof(u.data)))")
Base.show(io::IO, u::VectorField) = print(io, "VectorField($(u.grid), ::$(typeof(u.data)))")
Base.show(io::IO, u::TensorField) = print(io, "TensorField($(u.grid), ::$(typeof(u.data)))")
Base.show(io::IO, ::MIME"text/plain", u::ScalarField) =
    print(io, join(map(string, size(u.data)), "×"), " ", u)
Base.show(io::IO, ::MIME"text/plain", u::VectorField) =
    print(io, join(map(string, size(u.data)), "×"), " ", u)
Base.show(io::IO, ::MIME"text/plain", u::TensorField) =
    print(io, join(map(string, size(u.data)), "×"), " ", u)

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

Base.getindex(u::VectorField, ::X) = ScalarField(Face1(), u.grid, view(u.data,:,:,:,1))
Base.getindex(u::VectorField, ::Y) = ScalarField(Face2(), u.grid, view(u.data,:,:,:,2))
Base.getindex(u::VectorField, ::Z) = ScalarField(Face3(), u.grid, view(u.data,:,:,:,3))

Base.getindex(u::TensorField, ::X, ::X) =
    ScalarField(Center(), u.grid, view(u.data,:,:,:,1,1))
Base.getindex(u::TensorField, ::Y, ::X) =
    ScalarField(Edge3(), u.grid, view(u.data,:,:,:,2,1))
Base.getindex(u::TensorField, ::Z, ::X) =
    ScalarField(Edge2(), u.grid, view(u.data,:,:,:,3,1))
Base.getindex(u::TensorField, ::X, ::Y) =
    ScalarField(Edge3(), u.grid, view(u.data,:,:,:,1,2))
Base.getindex(u::TensorField, ::Y, ::Y) =
    ScalarField(Center(), u.grid, view(u.data,:,:,:,2,2))
Base.getindex(u::TensorField, ::Z, ::Y) =
    ScalarField(Edge1(), u.grid, view(u.data,:,:,:,3,2))
Base.getindex(u::TensorField, ::X, ::Z) =
    ScalarField(Edge2(), u.grid, view(u.data,:,:,:,1,3))
Base.getindex(u::TensorField, ::Y, ::Z) =
    ScalarField(Edge1(), u.grid, view(u.data,:,:,:,2,3))
Base.getindex(u::TensorField, ::Z, ::Z) =
    ScalarField(Center(), u.grid, view(u.data,:,:,:,3,3))

function Base.similar(u::ScalarField, ::Type{T}, dims::Dims) where {T}
    @assert dims == size(u.data) "Scalar field must have same size as grid."
    ScalarField(u.position, u.grid, similar(u.data, T))
end
function Base.similar(u::VectorField, ::Type{T}, dims::Dims) where {T}
    @assert dims == size(u.data) "Vector field must have same size as grid."
    VectorField(u.grid, similar(u.data, T))
end
function Base.similar(u::TensorField, ::Type{T}, dims::Dims) where {T}
    @assert dims == size(u.data) "Tensor field must have same size as grid."
    TensorField(u.grid, similar(u.data, T))
end

Base.copyto!(v::ScalarField, u::ScalarField) = (copyto!(v.data, u.data); v)
Base.copyto!(v::VectorField, u::VectorField) = (copyto!(v.data, u.data); v)
Base.copyto!(v::TensorField, u::TensorField) = (copyto!(v.data, u.data); v)

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
