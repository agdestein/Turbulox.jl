function spectral_stuff(setup; npoint = 100)
    (; grid, backend, visc) = setup
    T = typeof(visc)
    d = dim(grid)

    N = ntuple(Returns(grid.n), d)
    K = ntuple(Returns(div(grid.n, 2)), d)

    k = zeros(T, K)
    if d == 2
        kx = reshape(0:K[1]-1, :)
        ky = reshape(0:K[2]-1, 1, :)
        @. k = sqrt(kx^2 + ky^2)
    elseif d == 3
        kx = reshape(0:K[1]-1, :)
        ky = reshape(0:K[2]-1, 1, :)
        kz = reshape(0:K[3]-1, 1, 1, :)
        @. k = sqrt(kx^2 + ky^2 + kz^2)
    end
    k = reshape(k, :)

    kmax = minimum(K) - 1
    isort = sortperm(k)
    ksort = k[isort]

    IntArray = typeof(adapt(backend, fill(0, 0)))
    inds = IntArray[]

    # Output query points (evenly log-spaced, but only integer wavenumbers)
    κ = logrange(T(1), T(kmax), npoint)
    κ = sort(unique(round.(Int, κ)))
    npoint = length(κ)

    for i = 1:npoint
        tol = T(0.01)
        jstart = findfirst(≥(κ[i] - tol), ksort)
        jstop = findfirst(≥(κ[i] + 1 - tol), ksort)
        isnothing(jstop) && (jstop = length(ksort) + 1)
        jstop -= 1
        push!(inds, adapt(IntArray, isort[jstart:jstop]))
    end

    u = KernelAbstractions.allocate(backend, T, N..., d)
    uhat = KernelAbstractions.allocate(
        backend,
        Complex{T},
        ntuple(i -> i == 1 ? div(grid.n, 2) + 1 : grid.n, d)...,
        d,
    )
    ehat = KernelAbstractions.allocate(backend, T, K..., d)
    plan = plan_rfft(u, 1:d)

    (; inds, κ, K, uhat, ehat, plan)
end

function spectrum(u, setup; npoint = 100, stuff = spectral_stuff(setup; npoint))
    (; grid) = setup
    (; n) = grid
    T = eltype(u)
    d = dim(grid)
    N = ntuple(Returns(n), d)
    (; inds, κ, K, uhat, ehat, plan) = stuff
    mul!(uhat, plan, u)
    fill!(ehat, 0)
    for i = 1:d
        uhathalf = view(uhat, ntuple(j -> 1:K[j], d)..., i)
        @. ehat += abs2(uhathalf) / 2 / (n^d)^2
    end
    s = map(i -> sum(view(ehat, i)), inds)
    (; s, κ)
end
