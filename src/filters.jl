"Gaussian filter kernel."
function gaussian(grid, compression, Δ)
    (; backend) = grid
    (; L, n, backend) = grid
    d = dim(grid)
    T = typeof(Δ)
    # Note:
    #     The standard deviation is σ = Δ / sqrt(12).
    #     This gives Δ ≈ 3.5σ. At x = Δ, the Gaussian is already quite small,
    #     and there is point in including points outside this bound.
    r = round(Int, Δ / 2 * n / L)
    r = 2r # Corresponds to x = Δ, sufficient to cover tail
    @assert isodd(compression)
    w = let
        x = (-r:r) .* (L / n)
        y = reshape(x, 1, :)
        z = reshape(x, 1, 1, :)
        if d == 2
            @. exp(-6 * (x^2 + y^2) / Δ^2)
        else
            @. exp(-6 * (x^2 + y^2 + z^2) / Δ^2)
        end
    end
    w = w ./ sum(w) # Normalize
    weights = adapt(backend, w)
    R = ntuple(Returns(r), d) |> splat(CartesianIndex)
    indices = -R:R # In 3D: (2r+1)^3 points in kernel
    (; weights, indices, compression)
end

function tophat(grid, compression, Δ)
    (; L, n, backend) = grid
    d = dim(grid)
    T = typeof(Δ)
    h = L / n
    r = round(Int, Δ / h / 2)
    @assert isodd(compression)
    @assert 2r / n ≈ Δ # This ensures that r is the same in every direction
    w = fill(T(1) / (2r + 1)^d, ntuple(Returns(2r + 1), d))
    weights = adapt(backend, w)
    R = ntuple(Returns(r), d) |> splat(CartesianIndex)
    indices = -R:R
    (; weights, indices, compression)
end

@kernel function convolve!(g::Grid, v, u, kernel, offset)
    x = @index(Global, Cartesian) # Coarse grid index
    (; weights, indices, compression) = kernel
    k = 0 # Linear weight index 
    vx = zero(eltype(v))
    xfine = compression * x + offset # Fine grid index
    while k < length(weights)
        k += 1
        y = xfine + indices[k] # Translate index
        y = CartesianIndex(mod1.(y.I, compression * g.n)) # Periodic extension
        vx += weights[k] * u[y] # Convolution
    end
    v[x] = vx
end

"Filter scalar field."
function applyfilter!(v, u, grid, filter, compression, Δ)
    (; backend, workgroupsize) = grid
    kernel = filter(grid, compression, Δ)
    r = div(compression, 2)
    offset = CartesianIndex(ntuple(Returns(-r), dim(grid)))
    convolve!(backend, workgroupsize)(grid, v, u, kernel, offset; ndrange = size(v))
    KernelAbstractions.synchronize(backend)
    kernel
end

"Filter staggered vector field."
function applyfilter!(v, u, grid, filter, compression, Δ, ::Stag)
    (; backend, workgroupsize) = grid
    d = dim(grid)
    kernel = filter(grid, compression, Δ)
    r = div(compression, 2)
    for i = 1:d
        offset = CartesianIndex(ntuple(j -> j == i ? 0 : -r, d))
        vi, ui = selectdim(v, d + 1, i), selectdim(u, d + 1, i)
        convolve!(backend, workgroupsize)(grid, vi, ui, kernel, offset; ndrange = size(vi))
    end
    KernelAbstractions.synchronize(backend)
    kernel
end

"Filter staggered tensor field."
function applyfilter!(v, u, grid, filter, compression, Δ, ::Stag, ::Stag)
    (; backend, workgroupsize) = grid
    d = dim(grid)
    @assert d == 3
    kernel = filter(grid, compression, Δ)
    r = div(compression, 2)
    for j = 1:d, i = 1:d
        offset = CartesianIndex(ntuple(k -> (i != j) && (k == i || k == j) ? 0 : -r, dim(grid)))
        v_ij, u_ij = view(v, :, :, :, i, j), view(u, :, :, :, i, j)
        convolve!(backend, workgroupsize)(grid, v_ij, u_ij, kernel, offset; ndrange = size(v_ij))
    end
    KernelAbstractions.synchronize(backend)
    kernel
end
