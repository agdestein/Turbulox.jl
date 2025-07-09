function spectral_stuff(grid; npoint = 100)
    (; L, backend) = grid
    T = typeof(L)

    N = ntuple(Returns(grid.n), 3)
    K = ntuple(Returns(div(grid.n, 2)), 3)

    k = zeros(T, K)
    kx = reshape(0:(K[1]-1), :)
    ky = reshape(0:(K[2]-1), 1, :)
    kz = reshape(0:(K[3]-1), 1, 1, :)
    @. k = sqrt(kx^2 + ky^2 + kz^2)
    k = reshape(k, :)

    kmax = minimum(K) - 1
    isort = sortperm(k)
    ksort = k[isort]

    IntArray = typeof(adapt(backend, fill(0, 0)))
    inds = IntArray[]

    # Output query points (evenly log-spaced, but only integer wavenumbers)
    kuse = logrange(T(1), T(kmax), npoint)
    kuse = sort(unique(round.(Int, kuse)))
    npoint = length(kuse)

    for i = 1:npoint
        tol = T(0.01)
        jstart = findfirst(≥(kuse[i] - tol), ksort)
        jstop = findfirst(≥(kuse[i] + 1 - tol), ksort)
        isnothing(jstop) && (jstop = length(ksort) + 1)
        jstop -= 1
        push!(inds, adapt(IntArray, isort[jstart:jstop]))
    end

    u_i = KernelAbstractions.allocate(backend, T, N...)
    uhat_i =
        KernelAbstractions.allocate(backend, Complex{T}, div(grid.n, 2) + 1, grid.n, grid.n)
    ehat = KernelAbstractions.allocate(backend, T, K)
    plan = plan_rfft(u_i)

    (; inds, k = kuse, K, uhat_i, ehat, plan)
end

function spectrum(u; npoint = 100, stuff = spectral_stuff(u.grid; npoint))
    (; grid) = u
    (; n) = grid
    T = eltype(u)
    N = n, n, n
    (; inds, k, K, uhat_i, ehat, plan) = stuff
    fill!(ehat, 0)
    for i = 1:3
        # mul!(uhat_i, plan, selectdim(u.data, 4, i))
        mul!(uhat_i, plan, view(u.data,:,:,:,i))
        uhathalf = view(uhat_i, ntuple(j -> 1:K[j], 3)...)
        @. ehat += abs2(uhathalf) / 2 / (n^3)^2
    end
    s = map(i -> sum(view(ehat, i)), inds)
    (; k, s)
end
