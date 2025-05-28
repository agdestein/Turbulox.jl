"Allocate empty tensor field (collocated)."
function collocated_tensorfield(g::Grid)
    (; L, backend) = g
    T = typeof(L)
    data = KernelAbstractions.zeros(backend, SMatrix{3,3,T,9}, g.n, g.n, g.n)
    ScalarField(g, data)
end

"Allocate empty tensor field (symmetric, staggered)."
symmetric_tensorfield(g::Grid) =
    KernelAbstractions.zeros(g.backend, typeof(g.L), g.n, g.n, g.n, 6)

function create_spectrum(; grid, kp, rng = Random.default_rng())
    (; L, backend) = grid
    T = typeof(L)
    d = 3
    τ = T(2π)

    # Maximum wavenumber (remove ghost volumes)
    K = ntuple(Returns(div(grid.n, 2)), 3)

    # Wavenumber vectors
    kk = reshape(0:(K[1]-1), :), reshape(0:(K[2]-1), 1, :), reshape(0:(K[3]-1), 1, 1, :)

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
    a .*= grid.n^3

    # Apply random phase shift
    ξ = ntuple(i -> rand!(rng, KernelAbstractions.allocate(backend, T, K)), 3)
    for i = 1:3
        a = cat(a, reverse(a; dims = i); dims = i)
        ξ = ntuple(3) do j
            s = i == j ? -1 : 1
            ξβ = ξ[j]
            cat(ξβ, reverse(s .* ξβ; dims = i); dims = i)
        end
    end
    ξ = sum(ξ)
    a = @. exp(im * τ * ξ) * a

    KK = 2 .* K
    kkkk =
        reshape(0:(KK[1]-1), :), reshape(0:(KK[2]-1), 1, :), reshape(0:(KK[3]-1), 1, 1, :)
    knorm = KernelAbstractions.zeros(backend, T, KK)
    for i = 1:3
        @. knorm += kkkk[i]^2
    end
    knorm .= sqrt.(knorm)

    # Create random unit vector for each wavenumber
    θ = rand!(rng, similar(knorm))
    ϕ = rand!(rng, similar(knorm))
    e = sinpi.(θ) .* cospi.(2 .* ϕ), sinpi.(θ) .* sinpi.(2 .* ϕ), cospi.(θ)

    # Remove non-divergence free part: (I - k k^T / k^2) e
    ke = sum(i -> e[i] .* kkkk[i], 1:3)
    for i = 1:3
        e0 = e[i][1:1] # CUDA doesn't like e[i][1]
        @. e[i] -= kkkk[i] * ke / knorm^2
        # Restore k=0 component, which is divergence free anyways
        e[i][1:1] .= e0
    end

    # Normalize
    enorm = sqrt.(sum(i -> e[i] .^ 2, 1:3))
    for i = 1:3
        e[i] ./= enorm
    end

    # Split velocity magnitude a into velocity components a*eα
    uhat = ntuple(3) do i
        eα = e[i]
        a .* eα
    end
    stack(uhat)
end

function randomfield(grid, poisson; A = 1, kp = 10, rng = Random.default_rng())
    # Create random velocity field
    uhat = create_spectrum(; grid, kp, rng)
    u = ifft(uhat, 1:3)
    u = @. A * real(u)
    u = VectorField(grid, u)

    # Make velocity field divergence free on staggered grid
    # (it is already divergence free on the "spectral grid")
    p = ScalarField(grid)
    project!(u, p, poisson)
    u
end
