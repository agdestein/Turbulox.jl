"Copy symmetric tensor to normal tensor"
desymmetrize!(r, rsym) = @. begin
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
@inline function pol_tensor_collocated(σ, I)
    x, y, z = X(), Y(), Z()
    ex, ey, ez = e(x), e(y), e(z)
    σxx = σ[x, y][I]
    σyy = σ[x, y][I]
    σzz = σ[x, y][I]
    σxy = (σ[x, y][I] + σ[x, y][I-ex] + σ[x, y][I-ey] + σ[x, y][I-ex-ey]) / 4
    σyx = (σ[y, x][I] + σ[y, x][I-ex] + σ[y, x][I-ey] + σ[y, x][I-ex-ey]) / 4
    σxz = (σ[x, z][I] + σ[x, z][I-ex] + σ[x, z][I-ez] + σ[x, z][I-ex-ez]) / 4
    σzx = (σ[z, x][I] + σ[z, x][I-ex] + σ[z, x][I-ez] + σ[z, x][I-ex-ez]) / 4
    σyz = (σ[y, z][I] + σ[y, z][I-ey] + σ[y, z][I-ez] + σ[y, z][I-ey-ez]) / 4
    σzy = (σ[z, y][I] + σ[z, y][I-ey] + σ[z, y][I-ez] + σ[z, y][I-ey-ez]) / 4
    SMatrix{3,3,typeof(σxx),9}(σxx, σyx, σzx, σxy, σyy, σzy, σxz, σyz, σzz)
end

"Interpolate collocated tensor to staggered tensor."
@inline function pol_tensor_stag(σ, x, i, j)
    ei, ej = e(i), e(j)
    if i == j
        σ[x][i, j]
    else
        (σ[x][i, j] + σ[x+ei][i, j] + σ[x+ej][i, j] + σ[x+ei+ej][i, j]) / 4
    end
end

@kernel function pol_tensor_stag!(σ_stag, σ_coll)
    x = @index(Global, Cartesian)
    @unroll for j in directions()
        @unroll for i in directions()
            σ_stag[x, i, j] = pol_tensor_stag(σ_coll, x, i, j)
        end
    end
end

function δ_collocated(n::Int, u, i, j, I)
    ei, ej = e(i), e(j)
    if i == j
        δ(n, u[i], j, I)
    else
        (
            δ(n, u[i], j, I) +
            δ(n, u[i], j, I - ej) +
            δ(n, u[i], j, I - ei) +
            δ(n, u[i], j, I - ei - ej)
        ) / 4
    end
end

∇_collocated(u, I) = ∇_collocated(1, u, I)

function ∇_collocated(n::Int, u, I)
    x, y, z = X(), Y(), Z()
    SMatrix{3,3,eltype(u),9}(
        δ_collocated(n, u, x, x, I),
        δ_collocated(n, u, y, x, I),
        δ_collocated(n, u, z, x, I),
        δ_collocated(n, u, x, y, I),
        δ_collocated(n, u, y, y, I),
        δ_collocated(n, u, z, y, I),
        δ_collocated(n, u, x, z, I),
        δ_collocated(n, u, y, z, I),
        δ_collocated(n, u, z, z, I),
    )
end

@inline idtensor() = SMatrix{3,3,Bool,9}(1, 0, 0, 0, 1, 0, 0, 0, 1)

@inline unittensor(i, j) =
    SVector(ntuple(k -> i == k, 3)) * SVector(ntuple(k -> j == k, 3))'

@kernel function velocitygradient!(∇u, u)
    I = @index(Global, Cartesian)
    x, y, z = X(), Y(), Z()
    ∇u[x, x][I] = δ(u[x], x, I)
    ∇u[y, x][I] = δ(u[x], y, I)
    ∇u[z, x][I] = δ(u[x], z, I)
    ∇u[x, y][I] = δ(u[y], x, I)
    ∇u[y, y][I] = δ(u[y], y, I)
    ∇u[z, y][I] = δ(u[y], z, I)
    ∇u[x, z][I] = δ(u[z], x, I)
    ∇u[y, z][I] = δ(u[z], y, I)
    ∇u[z, z][I] = δ(u[z], z, I)
end

"Compute ``u v^T`` in the collocated points."
@kernel function tensorproduct_coll!(uv, u, v)
    x = @index(Global, Cartesian)
    uvec = SVector(pol(u, x, 1, 1), pol(u, x, 2, 2), pol(u, x, 3, 3))
    vvec = SVector(pol(v, x, 1, 1), pol(v, x, 2, 2), pol(v, x, 3, 3))
    uv[x] = uvec * vvec'
end

"Compute ``u v^T`` in the staggered points."
@kernel function tensorproduct_stag!(uv, u, v)
    x = @index(Global, Cartesian)
    u11, v11 = pol(u, x, 1, 1), pol(v, x, 1, 1)
    u21, v21 = pol(u, x, 2, 1), pol(v, x, 2, 1)
    u31, v31 = pol(u, x, 3, 1), pol(v, x, 3, 1)
    u12, v12 = pol(u, x, 1, 2), pol(v, x, 1, 2)
    u22, v22 = pol(u, x, 2, 2), pol(v, x, 2, 2)
    u32, v32 = pol(u, x, 3, 2), pol(v, x, 3, 2)
    u13, v13 = pol(u, x, 1, 3), pol(v, x, 1, 3)
    u23, v23 = pol(u, x, 2, 3), pol(v, x, 2, 3)
    u33, v33 = pol(u, x, 3, 3), pol(v, x, 3, 3)
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

@kernel function dissipation!(diss, u, visc)
    x = @index(Global, Cartesian)
    ∇u = ∇_collocated(u, x)
    S = (∇u + ∇u') / 2
    diss[x] = 2 * visc * dot(S, S)
end

@kernel function tensordissipation!(diss, σ, ∇u)
    x = @index(Global, Cartesian)
    G = ∇u[x]
    S = (G + G') / 2
    diss[x] = -dot(σ[x], S)
end

@inline function dissipation(σ, u, i, j, I)
    ei, ej = e(i), e(j)
    if i == j
        σ[i, j][I] * δ(u[i], j, I)
    else
        # (
        #     σ[i, j][I] * strain(u, i, j, I) +
        #     σ[i, j][I-ei] * strain(u, i, j, I - ei) +
        #     σ[i, j][I-ej] * strain(u, i, j, I - ej) +
        #     σ[i, j][I-ei-ej] * strain(u, i, j, I - ei - ej)
        # ) / 4
        (
            σ[i, j][I] * δ(u[i], j, I) +
            σ[i, j][I-ei] * δ(u[i], j, I - ei) +
            σ[i, j][I-ej] * δ(u[i], j, I - ej) +
            σ[i, j][I-ei-ej] * δ(u[i], j, I - ei - ej)
        ) / 4
    end
end

@kernel function tensordissipation_staggered!(diss, σ, u)
    I = @index(Global, Cartesian)
    x, y, z = X(), Y(), Z()
    diss[I] =
        dissipation(σ, u, x, x, I) +
        dissipation(σ, u, y, x, I) +
        dissipation(σ, u, z, x, I) +
        dissipation(σ, u, x, y, I) +
        dissipation(σ, u, y, y, I) +
        dissipation(σ, u, z, y, I) +
        dissipation(σ, u, x, z, I) +
        dissipation(σ, u, y, z, I) +
        dissipation(σ, u, z, z, I)
end

@inline function invariants(∇u)
    S = (∇u + ∇u') / 2
    R = (∇u - ∇u') / 2
    tr(S * S), tr(R * R), tr(S * S * S), tr(S * R * R), tr(S * S * R * R)
end

"Compute deviatoric part of a tensor."
deviator(σ) = σ - tr(σ) / 3 * oneunit(σ)

"Compute Pope's tensor basis [popeTurbulentFlows2000](@cite)."
tensorbasis

@inline function tensorbasis(∇u)
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

@inline ninvariant() = 5
@inline ntensorbasis() = 10

"Fill `V` with invariants."
@kernel function fill_invariants!(V, ∇u)
    x = @index(Global, Cartesian)
    v = invariants(∇u[x])
    @unroll for i = 1:ninvariant()
        V[i, x] = v[i]
    end
end

"Fill `B` with basis tensors."
@kernel function fill_tensorbasis!(B, ∇u)
    x = @index(Global, Cartesian)
    b = tensorbasis(∇u[x])
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

"Expand tensor basis `B` and fill `τ[x] = c[i, x] * B[i, x]` (sum over `i`)."
@kernel function expand_tensorbasis!(τ, c, ∇u)
    x = @index(Global, Cartesian)
    b = tensorbasis(∇u[x])
    τx = zero(eltype(τ))
    # @unroll for i = 1:ntensorbasis()
    #     τx += c[i, x] * b[i]
    # end
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
    τ[x] = τx
end

@inline strain(u, i, j, x) = (δ(u[i], j, x) + δ(u[j], i, x)) / 2

@kernel function strain!(S, u)
    x = @index(Global, Cartesian)
    @unroll for i = 1:3
        @unroll for j = 1:3
            S[x, i, j] = strain(u, i, j, x)
        end
    end
end

@kernel function compute_qr!(q, r, ∇u)
    x = @index(Global, Cartesian)
    G = ∇u[x]
    q[x] = -tr(G * G) / 2
    r[x] = -tr(G * G * G) / 3
end

@kernel function compute_q!(q, ∇u)
    x = @index(Global, Cartesian)
    G = ∇u[x]
    q[x] = -tr(G * G) / 2
end

"""
Divergence of staggered tensor field ``σ``.
Subtract result from existing force field ``f``.
The operation is ``f_i \\leftarrow f_i - ∂_j σ_{i j}``.
"""
@kernel function tensordivergence!(f, σ)
    I = @index(Global, Cartesian)
    x, y, z = X(), Y(), Z()
    f[x][I] -= δ(σ[x, x], x, I) + δ(σ[x, y], y, I) + δ(σ[x, z], z, I)
    f[y][I] -= δ(σ[y, x], x, I) + δ(σ[y, y], y, I) + δ(σ[y, z], z, I)
    f[z][I] -= δ(σ[z, x], x, I) + δ(σ[z, y], y, I) + δ(σ[z, z], z, I)
end

"""
Divergence of collocated tensor field ``\\sigma``.
First interpolate to staggered points.
Subtract result from existing force field ``f``.
The operation is ``f_i \\leftarrow f_i - ∂_j σ_{i j}``.
"""
@inline function tensordivergence_collocated(
    σ,
    ii::Direction{i},
    jj::Direction{j},
    I,
) where {i,j}
    ei, ej = e(ii), e(jj)
    if i == j
        σa = σ[I][i, j]
        σb = σ[I+ei][i, j]
    else
        σa = (σ[I][i, j] + σ[I-ej][i, j] + σ[I+ei][i, j] + σ[I+ei-ej][i, j]) / 4
        σb = (σ[I][i, j] + σ[I+ei+ej][i, j] + σ[I+ei][i, j] + σ[I+ej][i, j]) / 4
    end
    -(σb - σa) / dx(σ.grid)
end

function symmetrize!(σsymm, σ)
    x, y, z = X(), Y(), Z()
    σsymm[x, x].data .= σ[x, x].data
    σsymm[y, y].data .= σ[y, y].data
    σsymm[z, z].data .= σ[z, z].data
    σsymm[x, y].data .= (σ[x, y].data .+ σ[y, x].data) ./ 2
    σsymm[x, z].data .= (σ[x, z].data .+ σ[z, x].data) ./ 2
    σsymm[y, z].data .= (σ[y, z].data .+ σ[z, y].data) ./ 2
    σsymm[y, x].data .= σsymm[x, y].data
    σsymm[z, x].data .= σsymm[x, z].data
    σsymm[z, y].data .= σsymm[y, z].data
    σsymm
end
