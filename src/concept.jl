"Get value from `Val`."
@inline getval(::Val{x}) where {x} = x

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
    print(io, "Grid(", join((g.ho, g.L, g.n, g.backend, g.workgroupsize), ", ")..., ")")

"Get order of grid."
@inline order(g::Grid) = 2 * getval(g.ho)

"Get grid spacing."
@inline dx(g::Grid) = g.L / g.n

# Extend index periodically so that it stays within the domain.
@inline (g::Grid)(i::Integer) = mod1(i, g.n)
# @inline (g::Grid)(x::CartesianIndex) = CartesianIndex(map(g, x.I))

"Cardinal direction `i`."
struct Direction{i} end
@inline Direction(i) = Direction{i}()

const X = Direction{1}
const Y = Direction{2}
const Z = Direction{3}

@inline directions() = X(), Y(), Z()

@inline orthogonal(::X) = Y(), Z()
@inline orthogonal(::Y) = X(), Z()
@inline orthogonal(::Z) = X(), Y()

@inline orthogonal(::X, ::Y) = Z()
@inline orthogonal(::Y, ::X) = Z()
@inline orthogonal(::X, ::Z) = Y()
@inline orthogonal(::Z, ::X) = Y()
@inline orthogonal(::Y, ::Z) = X()
@inline orthogonal(::Z, ::Y) = X()

"1D staggered grid position (at the right boundary of a volume in a given direction)."
struct Stag end

"1D collocated grid position (at the volume center in a given direction)."
struct Coll end

"Position in a 3D finite volume."
struct Position{P1,P2,P3} end
const Center = Position{Coll,Coll,Coll}
const FaceX = Position{Stag,Coll,Coll}
const FaceY = Position{Coll,Stag,Coll}
const FaceZ = Position{Coll,Coll,Stag}
const EdgeX = Position{Coll,Stag,Stag}
const EdgeY = Position{Stag,Coll,Stag}
const EdgeZ = Position{Stag,Stag,Coll}
const Corner = Position{Stag,Stag,Stag}

Base.getindex(::Position{PX,PY,PZ}, ::X) where {PX,PY,PZ} = PX()
Base.getindex(::Position{PX,PY,PZ}, ::Y) where {PX,PY,PZ} = PY()
Base.getindex(::Position{PX,PY,PZ}, ::Z) where {PX,PY,PZ} = PZ()

@inline vectorposition(::X) = FaceX()
@inline vectorposition(::Y) = FaceY()
@inline vectorposition(::Z) = FaceZ()

@inline tensorposition(::Direction{i}, ::Direction{i}) where {i} = Center()
@inline tensorposition(::X, ::Y) = EdgeZ()
@inline tensorposition(::Y, ::X) = EdgeZ()
@inline tensorposition(::X, ::Z) = EdgeY()
@inline tensorposition(::Z, ::X) = EdgeY()
@inline tensorposition(::Y, ::Z) = EdgeX()
@inline tensorposition(::Z, ::Y) = EdgeX()

"Get unit index in direction `i`."
@inline e(::X) = CartesianIndex(1, 0, 0)
@inline e(::Y) = CartesianIndex(0, 1, 0)
@inline e(::Z) = CartesianIndex(0, 0, 1)

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
struct VectorField{G,D}
    grid::G
    data::D
end
VectorField(g::Grid) =
    VectorField(g, KernelAbstractions.zeros(g.backend, typeof(g.L), g.n, g.n, g.n, 3))

"Tensor field."
struct TensorField{G,D}
    grid::G
    data::D
end
TensorField(g::Grid) =
    TensorField(g, KernelAbstractions.zeros(g.backend, typeof(g.L), g.n, g.n, g.n, 3, 3))

"""
Lazily computed scalar field.
Entries at index `I` are given by `func(stuff..., I)`.
"""
struct LazyScalarField{T,P,G,F,S} <: AbstractArray{T,3}
    position::P
    grid::G
    func::F
    stuff::S
    LazyScalarField(position, grid, func, stuff...) =
        new{typeof(grid.L),typeof(position),typeof(grid),typeof(func),typeof(stuff)}(
            position,
            grid,
            func,
            stuff,
        )
    LazyScalarField(g::Grid, func, stuff...) = LazyScalarField(Center(), g, func, stuff...)
end

struct LazyVectorField{G,F,S}
    grid::G
    func::F
    stuff::S
    LazyVectorField(g::Grid, func, stuff...) =
        new{typeof(g),typeof(func),typeof(stuff)}(g, func, stuff)
end

struct LazyTensorField{G,F,S}
    grid::G
    func::F
    stuff::S
    LazyTensorField(g::Grid, func, stuff...) =
        new{typeof(g),typeof(func),typeof(stuff)}(g, func, stuff)
end

Adapt.adapt_structure(to, u::ScalarField) =
    ScalarField(u.position, u.grid, adapt(to, u.data))
Adapt.adapt_structure(to, u::VectorField) = VectorField(u.grid, adapt(to, u.data))
Adapt.adapt_structure(to, u::TensorField) = TensorField(u.grid, adapt(to, u.data))
Adapt.adapt_structure(to, u::LazyScalarField) =
    LazyScalarField(u.position, u.grid, u.func, adapt(to, u.stuff)...)
Adapt.adapt_structure(to, u::LazyVectorField) =
    LazyVectorField(u.grid, u.func, adapt(to, u.stuff)...)
Adapt.adapt_structure(to, u::LazyTensorField) =
    LazyTensorField(u.grid, u.func, adapt(to, u.stuff)...)

Base.size(u::ScalarField) = size(u.data)
Base.size(u::VectorField) = size(u.data)
Base.size(u::TensorField) = size(u.data)
Base.size(u::LazyScalarField) = u.grid.n, u.grid.n, u.grid.n
Base.size(u::LazyVectorField) = u.grid.n, u.grid.n, u.grid.n, 3
Base.size(u::LazyTensorField) = u.grid.n, u.grid.n, u.grid.n, 3, 3

Base.show(io::IO, u::ScalarField) =
    print(io, "ScalarField($(u.position), $(u.grid), ::$(typeof(u.data)))")
Base.show(io::IO, u::VectorField) = print(io, "VectorField($(u.grid), ::$(typeof(u.data)))")
Base.show(io::IO, u::TensorField) = print(io, "TensorField($(u.grid), ::$(typeof(u.data)))")
Base.show(io::IO, u::LazyScalarField) = print(
    io,
    "LazyScalarField(",
    join((u.position, u.grid, u.func, map(s -> "::$(typeof(s))", u.stuff)...), ", ")...,
    ")",
)
# Base.show(io::IO, u::LazyVectorField) =
#     print(io, "LazyVectorField(", join((u.grid, u.fields...), ", ")..., ")")
# Base.show(io::IO, u::LazyTensorField) =
#     print(io, "LazyTensorField(", join((u.grid, u.fields...), ", ")..., ")")

Base.show(io::IO, ::MIME"text/plain", u::ScalarField) =
    print(io, join(map(string, size(u)), "×")..., " ", u)
Base.show(io::IO, ::MIME"text/plain", u::VectorField) =
    print(io, join(map(string, size(u)), "×")..., " ", u)
Base.show(io::IO, ::MIME"text/plain", u::TensorField) =
    print(io, join(map(string, size(u)), "×")..., " ", u)
Base.show(io::IO, ::MIME"text/plain", u::LazyScalarField) =
    print(io, join(map(string, size(u)), "×")..., " ", u)
# Base.show(io::IO, ::MIME"text/plain", u::LazyVectorField) =
#     print(io, join(map(string, size(u)), "×")..., " ", u)
# Base.show(io::IO, ::MIME"text/plain", u::LazyTensorField) =
#     print(io, join(map(string, size(u)), "×")..., " ", u)

@inline Base.getindex(u::ScalarField, I::Vararg{Int,3}) = u.data[map(u.grid, I)...]
Base.getindex(u::LazyScalarField, I::Vararg{Int,3}) = u.func(u.stuff..., CartesianIndex(I))

@inline Base.setindex!(u::ScalarField, val, I::Vararg{Int,3}) =
    setindex!(u.data, val, map(u.grid, I)...)
# No modifying of lazy fields.

@inline Base.getindex(u::VectorField, ii::Direction{i}) where {i} =
    ScalarField(vectorposition(ii), u.grid, view(u.data,:,:,:,i))
@inline Base.getindex(u::TensorField, ii::Direction{i}, jj::Direction{j}) where {i,j} =
    ScalarField(tensorposition(ii, jj), u.grid, view(u.data,:,:,:,i,j))
@inline Base.getindex(u::LazyVectorField, i) =
    LazyScalarField(vectorposition(i), u.grid, u.stuff..., i)
@inline Base.getindex(u::LazyTensorField, i, j) =
    LazyScalarField(tensorposition(i, j), u.grid, u.func, u.stuff..., i, j)

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

# Eltype for non-`AbstractArray`s defaults to `Any`, which we never want.
Base.eltype(u::VectorField) = eltype(u.data)
Base.eltype(u::TensorField) = eltype(u.data)
Base.eltype(u::LazyVectorField) = error("Please infer this.")
Base.eltype(u::LazyTensorField) = error("Please infer this.")

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
