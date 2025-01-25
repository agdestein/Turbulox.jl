"Interpolate staggered tensor to volume centers."
@inline function pol_tensor_collocated(g::Grid{o,2}, σ, x) where {o}
    e1, e2 = e(g, 1), e(g, 2)
    σ11 = σ[x, 1, 1]
    σ22 = σ[x, 2, 2]
    σ12 = (σ[x, 1, 2] + σ[x-e1|>g, 1, 2] + σ[x-e2|>g, 1, 2] + σ[x-e1-e2|>g, 1, 2]) / 4
    σ21 = (σ[x, 2, 1] + σ[x-e1|>g, 2, 1] + σ[x-e2|>g, 2, 1] + σ[x-e1-e2|>g, 2, 1]) / 4
    SMatrix{2,2,eltype(σ),4}(σ11, σ21, σ12, σ22)
end
@inline function pol_tensor_collocated(g::Grid{o,3}, σ, x) where {o}
    e1, e2, e3 = e(g, 1), e(g, 2), e(g, 3)
    σ11 = σ[x, 1, 1]
    σ22 = σ[x, 2, 2]
    σ33 = σ[x, 3, 3]
    σ12 = (σ[x, 1, 2] + σ[x-e1|>g, 1, 2] + σ[x-e2|>g, 1, 2] + σ[x-e1-e2|>g, 1, 2]) / 4
    σ21 = (σ[x, 2, 1] + σ[x-e1|>g, 2, 1] + σ[x-e2|>g, 2, 1] + σ[x-e1-e2|>g, 2, 1]) / 4
    σ13 = (σ[x, 1, 3] + σ[x-e1|>g, 1, 3] + σ[x-e3|>g, 1, 3] + σ[x-e1-e3|>g, 1, 3]) / 4
    σ31 = (σ[x, 3, 1] + σ[x-e1|>g, 3, 1] + σ[x-e3|>g, 3, 1] + σ[x-e1-e3|>g, 3, 1]) / 4
    σ23 = (σ[x, 2, 3] + σ[x-e2|>g, 2, 3] + σ[x-e3|>g, 2, 3] + σ[x-e2-e3|>g, 2, 3]) / 4
    σ32 = (σ[x, 3, 2] + σ[x-e2|>g, 3, 2] + σ[x-e3|>g, 3, 2] + σ[x-e2-e3|>g, 3, 2]) / 4
    SMatrix{3,3,eltype(σ),9}(σ11, σ21, σ31, σ12, σ22, σ32, σ13, σ23, σ33)
end

@inline function δ_collocated(g::Grid, u, x, i, j)
    ei, ej = e(g, i), e(g, j)
    if i == j
        δ(g, u, x, i, j)
    else
        (
            δ(g, u, x, i, j) +
            δ(g, u, x - ej, i, j) +
            δ(g, u, x - ei, i, j) +
            δ(g, u, x - ei - ej, i, j)
        ) / 4
    end
end

@inline ∇_collocated(g::Grid, u, x::CartesianIndex{2}) = SMatrix{2,2,eltype(u),4}(
    δ_collocated(g, u, x, 1, 1),
    δ_collocated(g, u, x, 2, 1),
    δ_collocated(g, u, x, 1, 2),
    δ_collocated(g, u, x, 2, 2),
)
@inline ∇_collocated(g::Grid, u, x::CartesianIndex{3}) = SMatrix{3,3,eltype(u),9}(
    δ_collocated(g, u, x, 1, 1),
    δ_collocated(g, u, x, 2, 1),
    δ_collocated(g, u, x, 3, 1),
    δ_collocated(g, u, x, 1, 2),
    δ_collocated(g, u, x, 2, 2),
    δ_collocated(g, u, x, 3, 2),
    δ_collocated(g, u, x, 1, 3),
    δ_collocated(g, u, x, 2, 3),
    δ_collocated(g, u, x, 3, 3),
)

@inline idtensor(::Grid{o,2}) where {o} = SMatrix{2,2,Bool,4}(1, 0, 0, 1)
@inline idtensor(::Grid{o,3}) where {o} = SMatrix{3,3,Bool,9}(1, 0, 0, 0, 1, 0, 0, 0, 1)

@inline unittensor(g::Grid, i, j) =
    SVector(ntuple(k -> i == k, dim(g))) * SVector(ntuple(k -> j == k, dim(g)))

@kernel function velocitygradient!(grid, ∇u, u)
    x = @index(Global, Cartesian)
    dims = 1:dim(grid)
    @unroll for i in dims
        @unroll for j in dims
            ∇u[x, i, j] = δ(grid, u, x, i, j)
        end
    end
end

@kernel function velocitygradient_collocated!(grid, ∇u, u)
    x = @index(Global, Cartesian)
    ∇u[x] = ∇_collocated(grid, u, x)
end

@kernel function tensorproduct!(grid, uv, u, v)
    (; n) = grid
    if dim(grid) == 2
        x, y = @index(Global, NTuple)
        uvec =
            SVector((u[x, y, 1] + u[x-1|>g, y, 1]) / 2, (u[x, y, 2] + u[x, y-1|>g, 2]) / 2)
        vvec =
            SVector((v[x, y, 1] + v[x-1|>g, y, 1]) / 2, (v[x, y, 2] + v[x, y-1|>g, 2]) / 2)
        uv[x, y] = uvec * vvec'
    else
        x, y, z = @index(Global, NTuple)
        uvec = SVector(
            (u[x, y, z, 1] + u[x-1|>g, y, z, 1]) / 2,
            (u[x, y, z, 2] + u[x, y-1|>g, z, 2]) / 2,
            (u[x, y, z, 3] + u[x, y, z-1|>g, 3]) / 2,
        )
        vvec = SVector(
            (v[x, y, z, 1] + v[x-1|>g, y, z, 1]) / 2,
            (v[x, y, z, 2] + v[x, y-1|>g, z, 2]) / 2,
            (v[x, y, z, 3] + v[x, y, z-1|>g, 3]) / 2,
        )
        uv[x, y, z] = uvec * vvec'
    end
end

@kernel function dissipation!(grid, ϵ, u, visc)
    x = @index(Global, Cartesian)
    ∇u = ∇_collocated(grid, u, x)
    S = (∇u + ∇u') / 2
    ϵ[x] = 2 * visc * dot(S, S)
end

@kernel function tensorbasis!(grid, B, V, ∇u)
    x = @index(Global, Cartesian)
    S = (∇u[x] + ∇u[x]') / 2
    R = (∇u[x] - ∇u[x]') / 2
    if dim(grid) == 2
        B[x, 1] = idtensor(grid)
        B[x, 2] = S
        B[x, 3] = S * R - R * S
        V[x, 1] = dot(S, S)
        V[x, 2] = dot(R, R)
    else
        B[x, 1] = idtensor(grid)
        B[x, 2] = S
        B[x, 3] = S * R - R * S
        B[x, 4] = S * S
        B[x, 5] = R * R
        B[x, 6] = S * S * R - R * S * S
        B[x, 7] = S * R * R + R * R * S
        B[x, 8] = R * S * R * R - R * R * S * R
        B[x, 9] = S * R * S * S - S * S * R * S
        B[x, 10] = S * S * R * R + R * R * S * S
        B[x, 11] = R * S * S * R * R - R * R * S * S * R
        V[x, 1] = tr(S * S)
        V[x, 2] = tr(R * R)
        V[x, 3] = tr(S * S * S)
        V[x, 4] = tr(S * R * R)
        V[x, 5] = tr(S * S * R * R)
    end
end

@kernel function strain!(grid, S, u)
    x = @index(Global, Cartesian)
    @unroll for i = 1:dim(grid)
        @unroll for j = 1:dim(grid)
            Aij = δ(grid, u, x, i, j)
            Aji = δ(grid, u, x, j, i)
            S[x, i, j] = (Aij + Aji) / 2
        end
    end
end

@kernel function compute_qr!(grid, q, r, ∇u)
    x = @index(Global, Cartesian)
    A = ∇u[x]
    q[x] = -tr(A * A) / 2
    r[x] = -tr(A * A * A) / 3
end

"""
Divergence of staggered tensor field.
Add result to existing force field `f`.
"""
@kernel function tensordivergence!(g::Grid, f, σ)
    x = @index(Global, Cartesian)
    dims = 1:dim(g)
    @unroll for i in dims
        div = f[x, i]
        @unroll for j in dims
            ei, ej = e(g, i), e(g, j)
            div += g.n * (σ[x+(i==j)*ej, i, j] - σ[x-(i!=j)*ej, i, j])
        end
        f[x, i] = div
    end
end

"""
Divergence of collocated tensor field.
First interpolate to staggered points.
Add result to existing force field `f`.
"""
@kernel function tensordivergence_collocated!(g::Grid, f, σ)
    x = @index(Global, Cartesian)
    dims = 1:dim(g)
    @unroll for i in dims
        div = f[x, i]
        @unroll for j in dims
            ei, ej = e(g, i), e(g, j)
            if i == j
                σa = σ[x][i, j]
                σb = σ[x+ei|>g][i, j]
            else
                σa =
                    (
                        σ[x][i, j] +
                        σ[x-ej|>g][i, j] +
                        σ[x+ei|>g][i, j] +
                        σ[x+ei-ej|>g][i, j]
                    ) / 4
                σb =
                    (
                        σ[x][i, j] +
                        σ[x+ei+ej|>g][i, j] +
                        σ[x+ei|>g][i, j] +
                        σ[x+ej|>g][i, j]
                    ) / 4
            end
            div += g.n * (σb - σa)
        end
        f[x, i] = div
    end
end
