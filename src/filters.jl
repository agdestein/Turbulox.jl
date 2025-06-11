"Gaussian filter kernel."
function gaussian(grid, compression, Δ)
    (; backend) = grid
    (; L, n, backend) = grid
    T = typeof(Δ)
    # Note:
    #     The standard deviation is σ = Δ / sqrt(12).
    #     This gives Δ ≈ 3.5σ. At x = Δ, the Gaussian is already quite small,
    #     and there is point in including points outside this bound.
    r = round(Int, Δ / 2 * n / L)
    r = 2r # Corresponds to x = Δ, sufficient to cover tail
    @assert isodd(compression)
    w = let
        x = ((-r):r) .* (L / n)
        y = reshape(x, 1, :)
        z = reshape(x, 1, 1, :)
        @. exp(-6 * (x^2 + y^2 + z^2) / Δ^2)
    end
    w = w ./ sum(w) # Normalize
    weights = adapt(backend, w)
    R = CartesianIndex((r, r, r))
    indices = (-R):R # In 3D: (2r+1)^3 points in kernel
    (; weights, indices, compression)
end

function tophat(grid, compression, Δ)
    (; L, n, backend) = grid
    T = typeof(Δ)
    h = L / n
    r = round(Int, Δ / h / 2)
    @assert isodd(compression)
    @assert 2r / n ≈ Δ # This ensures that r is the same in every direction
    w = fill(T(1) / (2r + 1)^3, ntuple(Returns(2r + 1), 3))
    weights = adapt(backend, w)
    R = CartesianIndex((r, r, r))
    indices = (-R):R
    (; weights, indices, compression)
end

@kernel function convolve!(v, u, kernel, offset)
    x = @index(Global, Cartesian) # Coarse grid index
    (; weights, indices, compression) = kernel
    k = 0 # Linear weight index 
    vx = zero(eltype(v))
    xfine = compression * x + offset # Fine grid index
    while k < length(weights)
        k += 1
        y = xfine + indices[k] # Translate index
        y = CartesianIndex(mod1.(y.I, compression * u.grid.n)) # Periodic extension
        vx += weights[k] * u[y] # Convolution
    end
    v[x] = vx
end

"Filter scalar field."
function applyfilter!(v, u, grid, filter, compression, Δ)
    (; backend, workgroupsize) = grid
    kernel = filter(grid, compression, Δ)
    r = div(compression, 2)
    offset = CartesianIndex(ntuple(Returns(-r), 3))
    convolve!(backend, workgroupsize)(v, u, kernel, offset; ndrange = size(v))
    KernelAbstractions.synchronize(backend)
    kernel
end

"Filter staggered vector field."
function applyfilter!(v, u, grid, filter, compression, Δ, ::Stag)
    (; backend, workgroupsize) = grid
    kernel = filter(grid, compression, Δ)
    r = div(compression, 2)
    for i = 1:3
        offset = CartesianIndex(ntuple(j -> j == i ? 0 : -r, 3))
        vi, ui = selectdim(v, 4, i), selectdim(u, 4, i)
        convolve!(backend, workgroupsize)(grid, vi, ui, kernel, offset; ndrange = size(vi))
    end
    KernelAbstractions.synchronize(backend)
    kernel
end

"Filter staggered tensor field."
function applyfilter!(v, u, grid, filter, compression, Δ, ::Stag, ::Stag)
    (; backend, workgroupsize) = grid
    kernel = filter(grid, compression, Δ)
    r = div(compression, 2)
    for j = 1:3, i = 1:3
        offset = CartesianIndex(ntuple(k -> (i != j) && (k == i || k == j) ? 0 : -r, 3))
        v_ij, u_ij = view(v,:,:,:,i,j), view(u,:,:,:,i,j)
        convolve!(backend, workgroupsize)(v_ij, u_ij, kernel, offset; ndrange = size(v_ij))
    end
    KernelAbstractions.synchronize(backend)
    kernel
end

"Get fine grid offset indices corresponding to coarse grid index at position."
fineline(::Stag, comp) = (a = div(comp, 2); (a+1):(comp+a))
fineline(::Coll, comp) = 1:comp

finepoint(::Stag, comp) = comp:comp
finepoint(::Coll, comp) = (a = div(comp, 2); (a+1):(a+1))

function volumefilter!(uH::ScalarField, uh, comp)
    (; grid, position) = uH
    (; n, backend, workgroupsize) = grid
    @kernel function Φ!(uH, uh, volume)
        x = @index(Global, Cartesian)
        y = comp * (x - oneunit(x))
        s = zero(eltype(uH))
        for r in volume
            s += uh[y+r]
        end
        uH[x] = s / comp^3
    end
    ndrange = n, n, n
    volume = CartesianIndices(map(i -> fineline(position[i], comp), directions()))
    Φ!(backend, workgroupsize)(uH, uh, volume; ndrange)
    uH
end
volumefilter!(uH::VectorField, uh, comp) =
    for i in directions()
        volumefilter!(uH[i], uh[i], comp)
    end
volumefilter!(uH::TensorField, uh, comp) =
    for j in directions(), i in directions()
        volumefilter!(uH[i, j], uh[i, j], comp)
    end

"Filter without reducing the grid size."
function volumefilter_explicit!(uH::ScalarField, uh, comp)
    (; grid, position) = uH
    (; n, backend, workgroupsize) = grid
    @kernel function Φ!(uH, uh, volume)
        x = @index(Global, Cartesian)
        s = zero(eltype(uH))
        for r in volume
            s += uh[x+r]
        end
        uH[x] = s / comp^3
    end
    ndrange = n, n, n
    a = div(comp, 2)
    volume = CartesianIndices(ntuple(Returns((-a):a), 3))
    Φ!(backend, workgroupsize)(uH, uh, volume; ndrange)
    uH
end
volumefilter_explicit!(uH::VectorField, uh, comp) =
    for i in directions()
        volumefilter_explicit!(uH[i], uh[i], comp)
    end
volumefilter_explicit!(uH::TensorField, uh, comp) =
    for j in directions(), i in directions()
        volumefilter_explicit!(uH[i, j], uh[i, j], comp)
    end

function surfacefilter!(uH::ScalarField, uh, comp, i)
    (; grid, position) = uH
    (; n, backend, workgroupsize) = grid
    @kernel function Φ!(uH, uh, face)
        x = @index(Global, Cartesian)
        y = comp * (x - oneunit(x))
        s = zero(eltype(uH))
        for r in face
            s += uh[y+r]
        end
        uH[x] = s / comp^2
    end
    ndrange = n, n, n
    face = CartesianIndices(
        map(
            j -> j == i ? finepoint(position[j], comp) : fineline(position[j], comp),
            directions(),
        ),
    )
    Φ!(backend, workgroupsize)(uH, uh, face; ndrange)
    uH
end
surfacefilter!(uH::VectorField, uh, comp) =
    for i in directions()
        surfacefilter!(uH[i], uh[i], comp, i)
    end
surfacefilter!(uH::TensorField, uh, comp, filter_i) =
    for j in directions(), i in directions()
        k = filter_i ? i : j
        surfacefilter!(uH[i, j], uh[i, j], comp, k)
    end

function linefilter!(uH::ScalarField, uh, comp, k)
    (; grid, position) = uH
    (; n, backend, workgroupsize) = grid
    @kernel function Φ!(uH, uh, line)
        x = @index(Global, Cartesian)
        y = comp * (x - oneunit(x))
        s = zero(eltype(uH))
        for r in line
            s += uh[y+r]
        end
        uH[x] = s / comp
    end
    ndrange = n, n, n
    line = CartesianIndices(
        map(
            j -> j == k ? fineline(position[j], comp) : finepoint(position[j], comp),
            directions(),
        ),
    )
    Φ!(backend, workgroupsize)(uH, uh, line; ndrange)
    uH
end
linefilter!(uH::VectorField, uh, comp) =
    for i in directions()
        linefilter!(uH[i], uh[i], comp, i)
    end
linefilter!(uH::TensorField, uh, comp) =
    for j in directions(), i in directions()
        if i != j
            k = orthogonal(i, j)
            linefilter!(uH[i, j], uh[i, j], comp, k)
        end
    end
