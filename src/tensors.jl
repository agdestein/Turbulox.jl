@inline function ∂x(u, I::CartesianIndex{d}, α, β, n) where {d}
    per = periodicindex(n)
    e = unitindices(Val(d))
    eα, eβ = e[α], e[β]
    if α == β
        n * (u[I, α] - u[I-eβ|>per, α])
    else
        n * (
            (u[I+eβ|>per, α] - u[I, α]) +
            (u[I-eα+eβ|>per, α] - u[I-eα|>per, α]) +
            (u[I, α] - u[I-eβ|>per, α]) +
            (u[I-eα|>per, α] - u[I-eα-eβ|>per, α])
        ) / 4
    end
end

@inline ∇(u, I::CartesianIndex{2}, n) = SMatrix{2,2,eltype(u),4}(
    ∂x(u, I, 1, 1, n),
    ∂x(u, I, 2, 1, n),
    ∂x(u, I, 1, 2, n),
    ∂x(u, I, 2, 2, n),
)
@inline ∇(u, I::CartesianIndex{3}, n) = SMatrix{3,3,eltype(u),9}(
    ∂x(u, I, 1, 1, n),
    ∂x(u, I, 2, 1, n),
    ∂x(u, I, 3, 1, n),
    ∂x(u, I, 1, 2, n),
    ∂x(u, I, 2, 2, n),
    ∂x(u, I, 3, 2, n),
    ∂x(u, I, 1, 3, n),
    ∂x(u, I, 2, 3, n),
    ∂x(u, I, 3, 3, n),
)

@inline idtensor(::CartesianIndex{2}) = SMatrix{2,2,Bool,4}(1, 0, 0, 1)
@inline idtensor(::CartesianIndex{3}) = SMatrix{3,3,Bool,9}(1, 0, 0, 0, 1, 0, 0, 0, 1)

@inline unittensor(::CartesianIndex{2}, α, β) =
    SVector(α == 1, α == 2) * SVector(β == 1, β == 2)'
@inline unittensor(::CartesianIndex{3}, α, β) =
    SVector(α == 1, α == 2, α == 3) * SVector(β == 1, β == 2, β == 3)'

@kernel function velocitygradient!(setup, ∇u, u)
    (; D, n) = setup
    I = @index(Global, Cartesian)
    ∇u[I] = ∇(u, I, n)
end

@kernel function tensorproduct!(setup, uv, u, v)
    (; D, n) = setup
    per = periodicindex(n)
    if getval(D) == 2
        i, j = @index(Global, Cartesian)
        uvec = SVector(
            (u[i, j, 1] + u[i-1|>per, j, 1]) / 2,
            (u[i, j, 2] + u[i, j-1|>per, 2]) / 2,
        )
        vvec = SVector(
            (v[i, j, 1] + v[i-1|>per, j, 1]) / 2,
            (v[i, j, 2] + v[i, j-1|>per, 2]) / 2,
        )
        uv[i, j] = uvec * vvec'
    else
        i, j, k = @index(Global, Cartesian)
        uvec = SVector(
            (u[i, j, k, 1] + u[i-1|>per, j, k, 1]) / 2,
            (u[i, j, k, 2] + u[i, j-1|>per, k, 2]) / 2,
            (u[i, j, k, 3] + u[i, j, k-1|>per, 3]) / 2,
        )
        vvec = SVector(
            (v[i, j, k, 1] + v[i-1|>per, j, k, 1]) / 2,
            (v[i, j, k, 2] + v[i, j-1|>per, k, 2]) / 2,
            (v[i, j, k, 3] + v[i, j, k-1|>per, 3]) / 2,
        )
        uv[i, j, k] = uvec * vvec'
    end
end

@kernel function tensorbasis!(setup, B, V, ∇u)
    I = @index(Global, Cartesian)
    S = (∇u[I] + ∇u[I]') / 2
    R = (∇u[I] - ∇u[I]') / 2
    if getval(D) == 2
        B[I, 1] = idtensor(I)
        B[I, 2] = S
        B[I, 3] = S * R - R * S
        V[I, 1] = dot(S, S)
        V[I, 2] = dot(R, R)
    else
        B[I, 1] = idtensor(I)
        B[I, 2] = S
        B[I, 3] = S * R - R * S
        B[I, 4] = S * S
        B[I, 5] = R * R
        B[I, 6] = S * S * R - R * S * S
        B[I, 7] = S * R * R + R * R * S
        B[I, 8] = R * S * R * R - R * R * S * R
        B[I, 9] = S * R * S * S - S * S * R * S
        B[I, 10] = S * S * R * R + R * R * S * S
        B[I, 11] = R * S * S * R * R - R * R * S * S * R
        V[I, 1] = tr(S * S)
        V[I, 2] = tr(R * R)
        V[I, 3] = tr(S * S * S)
        V[I, 4] = tr(S * R * R)
        V[I, 5] = tr(S * S * R * R)
    end
end
