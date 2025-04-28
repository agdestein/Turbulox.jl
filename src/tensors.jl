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
@inline function pol_tensor_collocated(σ, x)
    e1, e2, e3 = e(1), e(2), e(3)
    σ11 = σ[x, 1, 1]
    σ22 = σ[x, 2, 2]
    σ33 = σ[x, 3, 3]
    σ12 = (σ[x, 1, 2] + σ[x-e1, 1, 2] + σ[x-e2, 1, 2] + σ[x-e1-e2, 1, 2]) / 4
    σ21 = (σ[x, 2, 1] + σ[x-e1, 2, 1] + σ[x-e2, 2, 1] + σ[x-e1-e2, 2, 1]) / 4
    σ13 = (σ[x, 1, 3] + σ[x-e1, 1, 3] + σ[x-e3, 1, 3] + σ[x-e1-e3, 1, 3]) / 4
    σ31 = (σ[x, 3, 1] + σ[x-e1, 3, 1] + σ[x-e3, 3, 1] + σ[x-e1-e3, 3, 1]) / 4
    σ23 = (σ[x, 2, 3] + σ[x-e2, 2, 3] + σ[x-e3, 2, 3] + σ[x-e2-e3, 2, 3]) / 4
    σ32 = (σ[x, 3, 2] + σ[x-e2, 3, 2] + σ[x-e3, 3, 2] + σ[x-e2-e3, 3, 2]) / 4
    SMatrix{3,3,eltype(σ),9}(σ11, σ21, σ31, σ12, σ22, σ32, σ13, σ23, σ33)
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
    @unroll for j = 1:3
        @unroll for i = 1:3
            σ_stag[x, i, j] = pol_tensor_stag(σ_coll, x, i, j)
        end
    end
end

@inline function δ_collocated(u, x, i, j)
    ei, ej = e(i), e(j)
    if i == j
        δ(u, x, i, j)
    else
        (
            δ(u, x, i, j) +
            δ(u, x - ej, i, j) +
            δ(u, x - ei, i, j) +
            δ(u, x - ei - ej, i, j)
        ) / 4
    end
end

@inline ∇_collocated(u, x::CartesianIndex{3}) = SMatrix{3,3,eltype(u),9}(
    δ_collocated(u, x, 1, 1),
    δ_collocated(u, x, 2, 1),
    δ_collocated(u, x, 3, 1),
    δ_collocated(u, x, 1, 2),
    δ_collocated(u, x, 2, 2),
    δ_collocated(u, x, 3, 2),
    δ_collocated(u, x, 1, 3),
    δ_collocated(u, x, 2, 3),
    δ_collocated(u, x, 3, 3),
)
@inline idtensor() = SMatrix{3,3,Bool,9}(1, 0, 0, 0, 1, 0, 0, 0, 1)

@inline unittensor(i, j) =
    SVector(ntuple(k -> i == k, 3)) * SVector(ntuple(k -> j == k, 3))'

@kernel function velocitygradient!(∇u, u)
    x = @index(Global, Cartesian)
    @unroll for i in 1:3
        @unroll for j in 1:3
            ∇u[x, i, j] = δ(u, x, i, j)
        end
    end
end

@kernel function velocitygradient_coll!(∇u, u)
    x = @index(Global, Cartesian)
    ∇u[x] = ∇_collocated(u, x)
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

@kernel function tensordissipation_staggered!(diss, σ, u)
    x = @index(Global, Cartesian)
    d = zero(eltype(diss))
    @unroll for i = 1:3
        @unroll for j = 1:3
            ei, ej = e(i), e(j)
            if i == j
                d -= σ[x, i, j] * δ(u, x, i, j)
            else
                d -=
                    (
                        σ[x, i, j] * strain(u, x, i, j) +
                        σ[x-ei, i, j] * strain(u, x - ei, i, j) +
                        σ[x-ej, i, j] * strain(u, x - ej, i, j) +
                        σ[x-ei-ej, i, j] * strain(u, x - ei - ej, i, j)
                    ) / 4
            end
        end
    end
    diss[x] = d
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

@inline strain(u, x, i, j) = (δ(u, x, i, j) + δ(u, x, j, i)) / 2

@kernel function strain!(S, u)
    x = @index(Global, Cartesian)
    @unroll for i = 1:3
        @unroll for j = 1:3
            S[x, i, j] = strain(u, x, i, j)
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
    x = @index(Global, Cartesian)
    @unroll for i in 1:3
        div = f[x, i]
        @unroll for j in 1:3
            ei, ej = e(i), e(j)
            div -= (σ[x+(i==j)*ej, i, j] - σ[x-(i!=j)*ej, i, j]) / dx(σ.grid)
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
@kernel function tensordivergence_collocated!(f, σ)
    x = @index(Global, Cartesian)
    @unroll for i in 1:3
        div = f[x, i] # add closure to existing force
        @unroll for j in 1:3
            ei, ej = e(i), e(j)
            if i == j
                σa = σ[x][i, j]
                σb = σ[x+ei][i, j]
            else
                σa =
                    (
                        σ[x][i, j] +
                        σ[x-ej][i, j] +
                        σ[x+ei][i, j] +
                        σ[x+ei-ej][i, j]
                    ) / 4
                σb =
                    (
                        σ[x][i, j] +
                        σ[x+ei+ej][i, j] +
                        σ[x+ei][i, j] +
                        σ[x+ej][i, j]
                    ) / 4
            end
            div -= (σb - σa) / dx(f.grid)
        end
        f[x, i] = div
    end
end

"Divergence first, then interpolate"
@kernel function tensordivergence_collocated_2!(f, σ)
    x = @index(Global, Cartesian)
    @unroll for i in 1:3
        div = f[x, i] # add closure to existing force
        @unroll for j in 1:3
            ei, ej = e(i), e(j)
            if i == j
                div -= (σ[x+ej][i, j] - σ[x][i, j]) / dx(f.grid)
            else
                div -=
                    (
                        (σ[x][i, j] - σ[x-ej][i, j]) / dx(f.grid) +
                        (σ[x+ej][i, j] - σ[x][i, j]) / dx(f.grid) +
                        (σ[x+ei][i, j] - σ[x+ei-ej][i, j]) / dx(f.grid) +
                        (σ[x+ei+ej][i, j] - σ[x+ei][i, j]) / dx(f.grid)
                    ) / 4
            end
        end
        f[x, i] = div
    end
end
