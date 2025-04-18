"Allocate empty scalar field."
scalarfield(g::Grid) =
    KernelAbstractions.zeros(g.backend, typeof(g.L), ntuple(Returns(g.n), dim(g)))

"Allocate empty vector field."
vectorfield(g::Grid) = KernelAbstractions.zeros(
    g.backend,
    typeof(g.L),
    ntuple(Returns(g.n), dim(g))...,
    dim(g),
)

"Allocate empty tensor field (collocated)."
function collocated_tensorfield(g::Grid)
    (; L, backend) = g
    d = dim(g)
    d2 = d * d
    T = typeof(L)
    KernelAbstractions.zeros(backend, SMatrix{d,d,T,d2}, ntuple(Returns(g.n), d))
end

"Allocate empty tensor field (staggered)."
function tensorfield(g::Grid)
    (; L, n, backend) = g
    d = dim(g)
    T = typeof(L)
    KernelAbstractions.zeros(backend, T, ntuple(Returns(n), d)..., d, d)
end

"Allocate empty tensor field (symmetric, staggered)."
function symmetric_tensorfield(g::Grid)
    (; L, n, backend) = g
    d = dim(g)
    T = typeof(L)
    ndof = d == 2 ? 3 : 6
    KernelAbstractions.zeros(backend, T, ntuple(Returns(n), d)..., ndof)
end

function create_spectrum(; grid, kp, rng = Random.default_rng())
    (; L, backend) = grid
    T = typeof(L)
    d = dim(grid)
    τ = T(2π)

    # Maximum wavenumber (remove ghost volumes)
    K = ntuple(Returns(div(grid.n, 2)), d)

    # Wavenumber vectors
    kk = ntuple(
        i -> reshape(
            0:(K[i]-1),
            ntuple(Returns(1), i - 1)...,
            :,
            ntuple(Returns(1), d - i)...,
        ),
        d,
    )

    # Wavevector magnitude
    k = KernelAbstractions.zeros(backend, T, K)
    for kk in kk
        @. k += kk^2
    end
    k .= sqrt.(k)

    # Shared magnitude
    A = T(8τ / 3) / kp^5

    # Velocity magnitude
    a = @. complex(1) * sqrt(A * k^4 * exp(-τ * (k / kp)^2))
    a .*= grid.n^d

    # Apply random phase shift
    ξ = ntuple(i -> rand!(rng, KernelAbstractions.allocate(backend, T, K)), d)
    for i = 1:d
        a = cat(a, reverse(a; dims = i); dims = i)
        ξ = ntuple(d) do j
            s = i == j ? -1 : 1
            ξβ = ξ[j]
            cat(ξβ, reverse(s .* ξβ; dims = i); dims = i)
        end
    end
    ξ = sum(ξ)
    a = @. exp(im * τ * ξ) * a

    KK = 2 .* K
    kkkk = ntuple(
        i -> reshape(
            0:(KK[i]-1),
            ntuple(Returns(1), i - 1)...,
            :,
            ntuple(Returns(1), d - i)...,
        ),
        d,
    )
    knorm = KernelAbstractions.zeros(backend, T, KK)
    for i = 1:d
        @. knorm += kkkk[i]^2
    end
    knorm .= sqrt.(knorm)

    # Create random unit vector for each wavenumber
    if d == 2
        θ = rand!(rng, similar(knorm))
        e = (cospi.(2 .* θ), sinpi.(2 .* θ))
    elseif d == 3
        θ = rand!(rng, similar(knorm))
        ϕ = rand!(rng, similar(knorm))
        e = (sinpi.(θ) .* cospi.(2 .* ϕ), sinpi.(θ) .* sinpi.(2 .* ϕ), cospi.(θ))
    end

    # Remove non-divergence free part: (I - k k^T / k^2) e
    ke = sum(i -> e[i] .* kkkk[i], 1:d)
    for i = 1:d
        e0 = e[i][1:1] # CUDA doesn't like e[i][1]
        @. e[i] -= kkkk[i] * ke / knorm^2
        # Restore k=0 component, which is divergence free anyways
        e[i][1:1] .= e0
    end

    # Normalize
    enorm = sqrt.(sum(i -> e[i] .^ 2, 1:d))
    for i = 1:d
        e[i] ./= enorm
    end

    # Split velocity magnitude a into velocity components a*eα
    uhat = ntuple(d) do i
        eα = e[i]
        a .* eα
    end
    stack(uhat)
end

function randomfield(grid, solver!; A = 1, kp = 10, rng = Random.default_rng())
    d = dim(grid)

    # Create random velocity field
    uhat = create_spectrum(; grid, kp, rng)
    u = ifft(uhat, 1:d)
    u = @. A * real(u)

    # Make velocity field divergence free on staggered grid
    # (it is already divergence free on the "spectral grid")
    p = scalarfield(grid)
    project!(u, p, solver!, grid)
    u
end
