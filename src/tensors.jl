"Copy symmtric tensor to normal tensor"
desymmetrize!(::Grid{o,2}, r, rsym) where {o} = @. begin
    r[:, :, 1, 1] = rsym[:, :, 1]
    r[:, :, 2, 2] = rsym[:, :, 2]
    r[:, :, 1, 2] = rsym[:, :, 3]
    r[:, :, 2, 1] = rsym[:, :, 3]
end
desymmetrize!(::Grid{o,3}, r, rsym) where {o} = @. begin
    r[:, :, :, 1, 1] = rsym[:, :, :, 1]
    r[:, :, :, 2, 2] = rsym[:, :, :, 2]
    r[:, :, :, 3, 3] = rsym[:, :, :, 3]
    r[:, :, :, 1, 2] = rsym[:, :, :, 4]
    r[:, :, :, 2, 1] = rsym[:, :, :, 4]
    r[:, :, :, 1, 3] = rsym[:, :, :, 5]
    r[:, :, :, 3, 1] = rsym[:, :, :, 5]
    r[:, :, :, 2, 3] = rsym[:, :, :, 6]
    r[:, :, :, 3, 2] = rsym[:, :, :, 6]
end

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

"Interpolate collocated tensor to staggered tensor."
@inline function pol_tensor_stag(g::Grid, σ, x, i, j)
    ei, ej = e(g, i), e(g, j)
    if i == j
        σ[x][i, j]
    else
        (σ[x][i, j] + σ[x+ei|>g][i, j] + σ[x+ej|>g][i, j] + σ[x+ei+ej|>g][i, j]) / 4
    end
end

@kernel function pol_tensor_stag!(g::Grid, σ_stag, σ_coll)
    x = @index(Global, Cartesian)
    @unroll for j = 1:dim(g)
        @unroll for i = 1:dim(g)
            σ_stag[x, i, j] = pol_tensor_stag(g, σ_coll, x, i, j)
        end
    end
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
    SVector(ntuple(k -> i == k, dim(g))) * SVector(ntuple(k -> j == k, dim(g)))'

@kernel function velocitygradient!(grid, ∇u, u)
    x = @index(Global, Cartesian)
    dims = 1:dim(grid)
    @unroll for i in dims
        @unroll for j in dims
            ∇u[x, i, j] = δ(grid, u, x, i, j)
        end
    end
end

@kernel function velocitygradient_coll!(grid, ∇u, u)
    x = @index(Global, Cartesian)
    ∇u[x] = ∇_collocated(grid, u, x)
end

"Compute ``u v^T`` in the collocated points."
@kernel function tensorproduct_coll!(g::Grid, uv, u, v)
    x = @index(Global, Cartesian)
    if dim(g) == 2
        uvec = SVector(pol(g, u, x, 1, 1), pol(g, u, x, 2, 2))
        vvec = SVector(pol(g, v, x, 1, 1), pol(g, v, x, 2, 2))
    else
        uvec = SVector(pol(g, u, x, 1, 1), pol(g, u, x, 2, 2), pol(g, u, x, 3, 3))
        vvec = SVector(pol(g, v, x, 1, 1), pol(g, v, x, 2, 2), pol(g, v, x, 3, 3))
    end
    uv[x] = uvec * vvec'
end

"Compute ``u v^T`` in the staggered points."
@kernel function tensorproduct_stag!(g::Grid, uv, u, v)
    x = @index(Global, Cartesian)
    if dim(g) == 2
        u11, v11 = pol(g, u, x, 1, 1), pol(g, v, x, 1, 1)
        u12, v12 = pol(g, u, x, 1, 2), pol(g, v, x, 1, 2)
        u21, v21 = pol(g, u, x, 2, 1), pol(g, v, x, 2, 2)
        u22, v22 = pol(g, u, x, 2, 2), pol(g, v, x, 2, 2)
        uv[x, 1, 1] = u11 * v11
        uv[x, 2, 1] = u21 * v12
        uv[x, 1, 2] = u12 * v21
        uv[x, 2, 2] = u22 * v22
    else
        u11, v11 = pol(g, u, x, 1, 1), pol(g, v, x, 1, 1)
        u21, v21 = pol(g, u, x, 2, 1), pol(g, v, x, 2, 1)
        u31, v31 = pol(g, u, x, 3, 1), pol(g, v, x, 3, 1)
        u12, v12 = pol(g, u, x, 1, 2), pol(g, v, x, 1, 2)
        u22, v22 = pol(g, u, x, 2, 2), pol(g, v, x, 2, 2)
        u32, v32 = pol(g, u, x, 3, 2), pol(g, v, x, 3, 2)
        u13, v13 = pol(g, u, x, 1, 3), pol(g, v, x, 1, 3)
        u23, v23 = pol(g, u, x, 2, 3), pol(g, v, x, 2, 3)
        u33, v33 = pol(g, u, x, 3, 3), pol(g, v, x, 3, 3)
        uv[x, 1, 1] = u11 * v11
        uv[x, 2, 1] = u21 * v12
        uv[x, 3, 1] = u31 * v13
        uv[x, 1, 2] = u12 * v21
        uv[x, 2, 2] = u22 * v22
        uv[x, 3, 2] = u32 * v23
        uv[x, 1, 3] = u13 * v31
        uv[x, 2, 3] = u23 * v32
        uv[x, 3, 3] = u33 * v33
    end
end

@kernel function dissipation!(grid, ϵ, u, visc)
    x = @index(Global, Cartesian)
    ∇u = ∇_collocated(grid, u, x)
    S = (∇u + ∇u') / 2
    ϵ[x] = 2 * visc * dot(S, S)
end

@kernel function tensordissipation!(grid, diss, σ, ∇u)
    x = @index(Global, Cartesian)
    G = ∇u[x]
    S = (G + G') / 2
    diss[x] = -dot(σ[x], S)
end

@kernel function tensordissipation_staggered!(g::Grid, diss, σ, u)
    x = @index(Global, Cartesian)
    d = zero(eltype(diss))
    @unroll for i = 1:dim(g)
        @unroll for j = 1:dim(g)
            ei, ej = e(g, i), e(g, j)
            if i == j
                d -= σ[x, i, j] * δ(g, u, x, i, j)
            else
                d -= (
                    σ[x, i, j] * strain(g, u, x, i, j) +
                    σ[x - ei, i, j] * strain(g, u, x - ei, i, j) +
                    σ[x - ej, i, j] * strain(g, u, x - ej, i, j) +
                    σ[x - ei - ej, i, j] * strain(g, u, x - ei - ej, i, j)
                ) / 4
            end
        end
    end
    diss[x] = d
end

@inline function invariants(::Grid{o,2}, ∇u) where {o}
    S = (∇u + ∇u') / 2
    R = (∇u - ∇u') / 2
    tr(S * S), tr(R * R)
end

@inline function invariants(::Grid{o,3}, ∇u) where {o}
    S = (∇u + ∇u') / 2
    R = (∇u - ∇u') / 2
    tr(S * S), tr(R * R), tr(S * S * S), tr(S * R * R), tr(S * S * R * R)
end

"Compute Pope's tensor basis [popeTurbulentFlows2000](@cite)."
tensorbasis

"Compute deviatoric part of a tensor."
deviator(σ) = σ - tr(σ) / 3 * oneunit(σ)

@inline function tensorbasis(g::Grid{o,2}, ∇u) where {o}
    S = (∇u + ∇u') / 2
    R = (∇u - ∇u') / 2
    oneunit(S), S, S * R - R * S
end

@inline function tensorbasis(g::Grid{o,3}, ∇u) where {o}
    T = eltype(∇u)
    S = (∇u + ∇u') / 2
    R = (∇u - ∇u') / 2
    (
        S,
        S * R - R * S,
        deviator(S * S),
        deviator(R * R),
        R * S * S - S * S * R,
        deviator(S * R * R + R * R * S),
        R * S * R * R - R * R * S * R,
        S * R * S * S - S * S * R * S,
        deviator(R * R * S * S + S * S * R * R),
        R * S * S * R * R - R * R * S * S * R,
    )
end

invariant_scalers(g::Grid{o,2}, t) where {o} = t^2, t^2
invariant_scalers(g::Grid{o,3}, t) where {o} = t^2, t^2, t^3, t^3, t^4
tensorbasis_scalers(g::Grid{o,2}, t) where {o} = one(t), t, t^2
tensorbasis_scalers(g::Grid{o,3}, t) where {o} =
    t, t^2, t^2, t^2, t^3, t^3, t^4, t^4, t^4, t^5

@inline ninvariant(::Grid{o,2}) where {o} = 2
@inline ninvariant(::Grid{o,3}) where {o} = 5
@inline ntensorbasis(::Grid{o,2}) where {o} = 3
@inline ntensorbasis(::Grid{o,3}) where {o} = 10

"Fill `V` with invariants."
@kernel function fill_invariants!(grid, V, ∇u)
    x = @index(Global, Cartesian)
    v = invariants(grid, ∇u[x])
    @unroll for i = 1:ninvariant(grid)
        V[i, x] = v[i]
    end
end

"Fill `B` with basis tensors."
@kernel function fill_tensorbasis!(grid, B, ∇u)
    x = @index(Global, Cartesian)
    b = tensorbasis(grid, ∇u[x])
    if dim(grid) == 2
        B[1, x] = b[1]
        B[2, x] = b[2]
        B[3, x] = b[3]
    else
        B[1, x] = b[1]
        B[2, x] = b[2]
        B[3, x] = b[3]
        B[4, x] = b[4]
        B[5, x] = b[5]
        B[6, x] = b[6]
        B[7, x] = b[7]
        B[8, x] = b[8]
        B[9, x] = b[9]
        B[10, x] = b[10]
    end
end

"Expand tensor basis `B` and fill `τ[x] = c[i, x] * B[i, x]` (sum over `i`)."
@kernel function expand_tensorbasis!(grid, τ, c, ∇u)
    x = @index(Global, Cartesian)
    b = tensorbasis(grid, ∇u[x])
    τx = zero(eltype(τ))
    # @unroll for i = 1:ntensorbasis(grid)
    #     τx += c[i, x] * b[i]
    # end
    if dim(grid) == 2
        τx += c[1, x] * b[1]
        τx += c[2, x] * b[2]
        τx += c[3, x] * b[3]
    else
        τx += c[1, x] * b[1]
        τx += c[2, x] * b[2]
        τx += c[3, x] * b[3]
        τx += c[4, x] * b[4]
        τx += c[5, x] * b[5]
        τx += c[6, x] * b[6]
        τx += c[7, x] * b[7]
        τx += c[8, x] * b[8]
        τx += c[9, x] * b[9]
        τx += c[10, x] * b[10]
    end
    τ[x] = τx
end

@inline strain(g::Grid, u, x, i, j) = (δ(g, u, x, i, j) + δ(g, u, x, j, i)) / 2

@kernel function strain!(grid, S, u)
    x = @index(Global, Cartesian)
    @unroll for i = 1:dim(grid)
        @unroll for j = 1:dim(grid)
            S[x, i, j] = strain(g, u, x, i, j)
        end
    end
end

@kernel function compute_qr!(grid, q, r, ∇u)
    x = @index(Global, Cartesian)
    G = ∇u[x]
    q[x] = -tr(G * G) / 2
    r[x] = -tr(G * G * G) / 3
end

@kernel function compute_q!(grid, q, ∇u)
    x = @index(Global, Cartesian)
    G = ∇u[x]
    q[x] = -tr(G * G) / 2
end

"""
Divergence of staggered tensor field ``σ``.
Subtract result from existing force field ``f``.
The operation is ``f_i \\leftarrow f_i - ∂_j σ_{i j}``.
"""
@kernel function tensordivergence!(g::Grid, f, σ)
    x = @index(Global, Cartesian)
    dims = 1:dim(g)
    @unroll for i in dims
        div = f[x, i]
        @unroll for j in dims
            ei, ej = e(g, i), e(g, j)
            div -= (σ[x+(i==j)*ej|>g, i, j] - σ[x-(i!=j)*ej|>g, i, j]) / dx(g)
        end
        f[x, i] = div
    end
end

"""
Divergence of collocated tensor field ``\\sigma``.
First interpolate to staggered points.
Subtract result from existing force field ``f``.
The operation is ``f_i \\leftarrow f_i - ∂_j σ_{i j}``.
"""
@kernel function tensordivergence_collocated!(g::Grid, f, σ)
    x = @index(Global, Cartesian)
    dims = 1:dim(g)
    @unroll for i in dims
        div = f[x, i] # add closure to existing force
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
            div -= (σb - σa) / dx(g)
        end
        f[x, i] = div
    end
end

"Divergence first, then interpolate"
@kernel function tensordivergence_collocated_2!(g::Grid, f, σ)
    x = @index(Global, Cartesian)
    dims = 1:dim(g)
    @unroll for i in dims
        div = f[x, i] # add closure to existing force
        @unroll for j in dims
            ei, ej = e(g, i), e(g, j)
            if i == j
                div -= (σ[x+ej|>g][i, j] - σ[x][i, j]) / dx(g)
            else
                div -=
                    (
                        (σ[x][i, j] - σ[x-ej|>g][i, j]) / dx(g) +
                        (σ[x+ej|>g][i, j] - σ[x][i, j]) / dx(g) +
                        (σ[x+ei|>g][i, j] - σ[x+ei-ej|>g][i, j]) / dx(g) +
                        (σ[x+ei+ej|>g][i, j] - σ[x+ei|>g][i, j]) / dx(g)
                    ) / 4
            end
        end
        f[x, i] = div
    end
end
