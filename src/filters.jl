function gaussian!(ubar, u, Δ, setup)
    (; backend, workgroupsize, grid) = setup
    n = grid.n
    d = dim(grid)
    T = eltype(u)
    r = round(Int, 2Δ / 2 * n)
    # a = (6 / π / Δ^2)^(3 / 2) |> T
    w = let
        x = (-r:r) ./ T(n)
        y = reshape(x, 1, :)
        z = reshape(x, 1, 1, :)
        if d == 2
            @. exp(-6 * (x^2 + y^2) / Δ^2)
        else
            @. exp(-6 * (x^2 + y^2 + z^2) / Δ^2)
        end
    end
    w = w ./ sum(w) # Normalize
    w = adapt(backend, w)
    R = ntuple(Returns(r), d) |> splat(CartesianIndex)
    J = -R:R
    @kernel function kernel!(ubar, u, w)
        x..., i = @index(Global, NTuple)
        X = CartesianIndex(x...)
        res = zero(eltype(ubar))
        k = 0
        while k < length(w)
            k += 1
            Y = X + J[k]
            y = mod1.(Y.I, n) # Periodic extension
            Y = CartesianIndex(y..., i)
            res += w[k] * u[Y]
        end
        ubar[x..., i] = res
    end
    kernel!(backend, workgroupsize)(ubar, u, w; ndrange = size(u))
    ubar, w
end
