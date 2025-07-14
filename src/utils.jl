"""
FFT wavenumber (similar to `fftfreq`, but returns integers).

## Example

```@doctest
julia> Turbulox.wavenumber.(1:8, 8)
8-element Vector{Int64}:
  0
  1
  2
  3
 -4
 -3
 -2
 -1
```
"""
wavenumber(i, n) = i-1-ifelse(i <= (n+1) >> 1, 0, n)

"""
Precompute wavenumber shell indices and
create cache arrays for spectrum computation.

If `npoint` is nothing, the spectrum is computed for all resolved shells.
If `npoint = 100`, the spectrum is only computed for 100 evenly log-spaced wavenumbers.
"""
function spectral_stuff(grid; npoint = nothing)
    (; L, backend) = grid
    T = typeof(L)

    n = grid.n
    kmax = div(n, 2)

    kx = 0:kmax # For RFFT, the x-wavenumbers are 0:kmax
    ky = reshape(wavenumber.(1:n, grid.n), 1, :) # Normal FFT wavenumbers
    kz = reshape(wavenumber.(1:n, grid.n), 1, 1, :) # Normal FFT wavenumbers
    kk = @. kx^2 + ky^2 + kz^2
    kk = reshape(kk, :)

    isort = sortperm(kk) # Permutation for sorting the wavenumbers
    kksort = kk[isort]

    # Output query points (evenly log-spaced, but only integer wavenumbers)
    if isnothing(npoint)
        kuse = 1:kmax
    else
        kuse = logrange(T(1), T(kmax), npoint)
        kuse = sort(unique(round.(Int, kuse)))
    end

    # Since the wavenumbers are sorted, we just need to find the start and stop of each shell.
    # The linear indices for that shell is then given by the permutation in that range.
    inds = map(kuse) do k
        jstart = findfirst(≥(k^2), kksort)
        jstop = findfirst(≥((k + 1)^2), kksort)
        isnothing(jstop) && (jstop = length(kksort) + 1) # findfirst may return nothing
        jstop -= 1
        isort[jstart:jstop] # Linear indices of the i-th shell
    end

    # We need to adapt the shells for RFFT.
    # Consider the following example:
    #
    # julia> n, kmax = 8, 4;
    # julia> u = randn(n, n, n);
    # julia> f = fft(u); r = rfft(u);
    # julia> sum(abs2, f)
    # 275142.33506202063
    # julia> sum(abs2, r) + sum(abs2, view(r, 2:kmax, :, :))
    # 275142.3350620207
    #
    # To compute the energy of the FFT, we need an additional term for RFFT.
    # The second term sums over all the x-indices except for 1 and kmax + 1.
    # We thus need to add indices to account for the conjugate symmetry in RFFT.
    # For an RFFT array r of size (kmax + 1, n, n), we have the linear index relation
    # r[i] == r[x, y, z]
    # if
    # i == x + (y - 1) * (kmax + 1) + (z - 1) * (kmax + 1) * n.
    # We therefore need to exclude the indices:
    # (x == 1), i.e. (i % (kmax + 1) == 1), and
    # (x == kmax + 1), i.e. (i % (kmax + 1) == 0).
    # We only keep i if (i % (kmax + 1) > 1).
    conjinds = map(i -> filter(j -> j % (kmax + 1) > 1, i), inds)
    inds = map(vcat, inds, conjinds) # Include conjugate indices

    # Put indices on GPU
    inds = map(adapt(backend), inds)

    # Temporary arrays for spectrum computation
    u = KernelAbstractions.allocate(backend, T, n, n, n)
    uhat = KernelAbstractions.allocate(backend, Complex{T}, kmax + 1, n, n)
    ehat = similar(uhat, T)
    plan = plan_rfft(u)

    (; shells = inds, k = kuse, uhat, ehat, plan)
end

function spectrum(u; npoint = nothing, stuff = spectral_stuff(u.grid; npoint))
    (; grid) = u
    (; shells, k, uhat, ehat, plan) = stuff
    fill!(ehat, 0)
    for i = 1:3
        mul!(uhat, plan, view(u.data,:,:,:,i)) # RFFT of ui
        @. ehat += abs2(uhat) / 2 / (grid.n^3)^2
    end
    s = map(i -> sum(view(ehat, i)), shells) # Total energy in each shell
    (; k, s)
end
