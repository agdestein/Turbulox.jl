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

    # Maximum wavenumber
    K = div(grid.n, 2)

    # Wavenumber vectors
    kx = reshape(0:(K-1), :)
    ky = reshape(0:(K-1), 1, :)
    kz = reshape(0:(K-1), 1, 1, :)

    # Wavevector magnitude
    k = KernelAbstractions.zeros(backend, T, K, K, K)
    @. k = sqrt(kx^2 + ky^2 + kz^2)

    # Shared magnitude
    A = T(8τ / 3) / kp^5

    # Velocity magnitude
    a = @. complex(1) * sqrt(A * k^4 * exp(-τ * (k / kp)^2)) * grid.n^3

    # Apply random phase shift
    ξ = ntuple(i -> rand!(rng, KernelAbstractions.allocate(backend, T, K, K, K)), 3)
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

    KK = 2 * K
    kkkk = reshape(0:(KK-1), :), reshape(0:(KK-1), 1, :), reshape(0:(KK-1), 1, 1, :)
    knorm = KernelAbstractions.zeros(backend, T, KK, KK, KK)
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

# energyprofile(k2, (; amplitude, kpeak)) =
#     amplitude * k2^2 / kpeak^5 * exp(-2 * k2 / kpeak^2)

energyprofile(kpeak) = k -> k^4 *  exp(-2 * (k / kpeak)^2)


"""
Make random velocity field with prescribed energy spectrum profile.

Note: The profile takes the scalar squared wavenumber norm as input,
define it as `profile(k2)`.
"""
function randomfield_simple(profile, grid, poisson; rng = Random.default_rng(), params)
    # Create random field and make it divergence free
    u = VectorField(grid)
    randn!(rng, u.data)
    p = ScalarField(grid)
    project!(u, p, poisson)

    # Plan transform
    plan = plan_rfft(u.data, 1:3)

    # Fourier coefficients of velocity field
    uhat = plan * u.data

    # Adjust the amplitude to match energy profile
    @kernel function normalize!(uhat, profile, params, n)
        I = @index(Global, Cartesian)
        kx, ky, kz = I[1] - 1, I[2] - 1, I[3] - 1
        ux, uy, uz = uhat[I, 1], uhat[I, 2], uhat[I, 3]
        k2 = kx^2 + ky^2 + kz^2
        E0 = profile(k2, params)
        E = (abs2(ux) + abs2(uy) + abs2(uz)) / 2
        uhat[I, 1] *= sqrt(E0 / E) * n^3
        uhat[I, 2] *= sqrt(E0 / E) * n^3
        uhat[I, 3] *= sqrt(E0 / E) * n^3
    end
    ndrange = div(grid.n, 2) + 1, grid.n, grid.n
    apply!(normalize!, grid, uhat, profile, params, grid.n; ndrange)

    # Note: Ideally, we should maybe not do this for each k = (kx, ky, kz),
    # but rather do it for each shell k = 0, 1, 2, 3, ...,
    # such that the energy contained in the shell k is exactly the profile.

    # Set constant mode to zero (broadcast since uhat may be a GPU array)
    uhat[1:1] .= 0

    # Inverse transform
    ldiv!(u.data, plan, uhat)

    # Normally, adjusting the amplitude of the 3D vector uhat(k)
    # does not remove divergence freeness, since the divergence becomes
    # i k dot uhat(k) in  Fourier space, which stays zero if uhat is scaled.
    # But since our divergence is discrete, defined through staggered finite
    # differences, this is no longer exactly the case. So we project again
    # to correct for this (minor?) error.
    project!(u, p, poisson)

    # The velocity now has
    # the correct spectrum,
    # random phase shifts,
    # random orientations, 
    # and is also divergence free.
    u
end

"""
Make random velocity field with prescribed energy spectrum profile.

Note: The profile takes the scalar wavenumber norm as input,
define it as `profile(k)`.
"""
function randomfield_shell(profile, grid, poisson; totalenergy = 1, rng = Random.default_rng())
    # Compute energy (also remove FFT coefficient n^3)
    @kernel function energy!(E, uhat, n)
        I = @index(Global, Cartesian)
        ux, uy, uz = uhat[I, 1], uhat[I, 2], uhat[I, 3]
        E[I] = (abs2(ux) + abs2(uy) + abs2(uz)) / 2 / (n^3)^2
    end

    # Mask for active wavenumbers: kleft ≤ k < kleft + 1
    # Do everything squared to avoid floats
    @kernel function mask!(mask, kleft, n)
        I = @index(Global, Cartesian)
        # Like fftfreq, but with integer instead of Float64
        kx = I[1] - 1 # RFFT x-wavenumber
        ky = wavenumber(I[2], n) # FFT wavenumber
        kz = wavenumber(I[3], n) # FFT wavenumber
        mask[I] = kleft^2 ≤ kx^2 + ky^2 + kz^2 < (kleft + 1)^2
    end

    # Adjust the amplitude to match energy profile
    @kernel function normalize!(uhat, mask, factor)
        I = @index(Global, Cartesian)
        ux, uy, uz = uhat[I, 1], uhat[I, 2], uhat[I, 3]
        if mask[I]
            uhat[I, 1] = factor * ux
            uhat[I, 2] = factor * uy
            uhat[I, 3] = factor * uz
        end
    end

    # Create random field and make it divergence free
    u = VectorField(grid)
    randn!(rng, u.data)
    p = ScalarField(grid)
    project!(u, p, poisson)

    # RFFT exploits conjugate symmetry, so we only need half the modes
    kmax = div(grid.n, 2)
    ndrange = kmax + 1, grid.n, grid.n

    # Allocate arrays
    E = similar(p.data, ndrange...)
    Emask = similar(E)
    mask = similar(E, Bool)

    # Plan transform
    plan = plan_rfft(u.data, 1:3)

    # Fourier coefficients of velocity field
    uhat = plan * u.data

    # Compute energy
    apply!(energy!, grid, E, uhat, grid.n; ndrange)

    # Maximum partially resolved wavenumber (sqrt(dim) * kmax)
    kdiag =  floor(Int, sqrt(3) * div(grid.n, 2))

    # Adjust energy in each partially resolved shell [k, k+1)
    for k in 0:kdiag
        apply!(mask!, grid, mask, k, grid.n; ndrange) # Shell mask
        @. Emask = mask * E 
        Eshell = sum(Emask) + sum(view(Emask, 2:kmax, :, :)) # Current energy in shell
        E0 = profile(k) # Desired energy in shell
        factor = sqrt(E0 / Eshell) # E = u^2 / 2
        apply!(normalize!, grid, uhat, mask, factor; ndrange)
    end

    # Set constant mode to zero (broadcast since uhat may be a GPU array)
    uhat[1:1] .= 0

    # Inverse transform
    ldiv!(u.data, plan, uhat)

    # Normally, adjusting the amplitude of the 3D vector uhat(k)
    # does not remove divergence freeness, since the divergence becomes
    # i k dot uhat(k) in  Fourier space, which stays zero if uhat is scaled.
    # But since our divergence is discrete, defined through staggered finite
    # differences, this is no longer exactly the case. So we project again
    # to correct for this (minor?) error.
    project!(u, p, poisson)

    # Adjust total energy
    E = sum(abs2, u.data) / 2 / grid.n^3
    u.data .*= sqrt(totalenergy / E)

    # The velocity now has
    # the correct spectrum,
    # random phase shifts,
    # random orientations, 
    # and is also divergence free.
    u
end
