"""
Apply `kernel!` on `setup.grid, args...` over the entire domain.
The `args` are typically input and output fields.
The kernel should be of the form
```julia
using KernelAbstractions
@kernel function kernel!(grid, args...)
    # content
end
```
"""
function apply!(kernel!, setup, args...)
    (; grid, backend, workgroupsize) = setup
    ndrange = ntuple(Returns(grid.n), dim(grid))
    kernel!(backend, workgroupsize)(grid, args...; ndrange)
    nothing
end

# Vector field gradient δu[α] / δx[β].
@inline δ(g::Grid{2}, u, X, α, β) =
    if α == β
        g.n * (u[X, α] - u[X-e(g, β)|>g, α])
    else
        g.n * (u[X+e(g, β)|>g, α] - u[X, α])
    end
@inline function δ(g::Grid{4}, u, X, α, β)
    if α == β
        δ1 = g.n * (u[X, α] - u[X-e(g, β)|>g, α])
        δ3 = g.n * (u[X+e(g, β)|>g, α] - u[X-2e(g, β)|>g, α])
    else
        δ1 = g.n * (u[X+e(g, β)|>g, α] - u[X, α])
        δ3 = g.n * (u[X+2e(g, β)|>g, α] - u[X-e(g, β)|>g, α])
    end
    9 * δ1 / 8 - δ3 / 8
end

# Scalar field gradient δp / δx[β].
@inline δ(g::Grid{2}, p, X, β) = g.n * (p[X] - p[X-e(g, β)|>g])
@inline function δ(g::Grid{4}, p, X, β)
    δ1 = g.n * (p[X] - p[X-e(g, β)|>g])
    δ3 = g.n * (p[X+e(g, β)|>g] - p[X-2e(g, β)|>g])
    9 * δ1 / 8 - δ3 / 8
end

"""
Compute divergence of vector field `u`.
Put the result in `div`.
"""
divergence!

@kernel function divergence!(setup, div, u)
    (; grid) = setup
    I = @index(Global, Cartesian)
    divI = zero(eltype(div))
    @unroll for β in 1:dim(grid)
        divI += δx(g, u, I, β)
    end
    div[I] = divI
end

"Approximate the convective force ``\\partial_j (u_i u_j)``."
function convterm end
@inline function convterm(g::Grid{2}, x, i, j)
    (; n) = g
    ei, ej = e(g, i), e(g, β)
    uij_a = (u[x-ej|>g, i] + u[x, i]) / 2
    uij_b = (u[x, i] + u[x+ej|>g, i]) / 2
    uji_a = (u[x-ej|>g, β] + u[x-ej+ei|>g, β]) / 2
    uji_b = (u[x, β] + u[x+ei|>g, β]) / 2
    ui_uj_a = uij_a * uji_a
    ui_uj_b = uij_b * uji_b
    n * (ui_uj_b - ui_uj_a)
end
@inline function convterm(g::Grid{4}, x, i, j)
    (; n) = g
    ei, ej = e(g, i), e(g, β)
    uij_a = (u[x-ej|>g, i] + u[x, i]) / 2
    uij_b = (u[x, i] + u[x+ej|>g, i]) / 2
    uji_a = (u[x-ej|>g, β] + u[x-ej+ei|>g, β]) / 2
    uji_b = (u[x, β] + u[x+ei|>g, β]) / 2
    ui_uj_a = uij_a * uji_a
    ui_uj_b = uij_b * uji_b
    n * (ui_uj_b - ui_uj_a)
end

"""
Compute convection-diffusion force from velocity `u`.
Add the force field to `f`.
"""
convectiondiffusion!

@kernel function convectiondiffusion!(g::Grid, visc, f, u)
    T = typeof(visc)
    dims = 1:dim(g)
    x = @index(Global, Cartesian)
    @unroll for i in dims
        fxi = f[x, i]
        @unroll for β in dims
            ei, ej = e(g, i), e(g, β)
            uij_a = (u[x-ej|>g, i] + u[x, i]) / 2
            uij_b = (u[x, i] + u[x+ej|>g, i]) / 2
            uji_a = (u[x-ej|>g, β] + u[x-ej+ei|>g, β]) / 2
            uji_b = (u[x, β] + u[x+ei|>g, β]) / 2
            ui_uj_a = uij_a * uji_a
            ui_uj_b = uij_b * uji_b
            δui_δxj_a = n * (u[x, i] - u[x-ej|>g, i])
            δui_δxj_b = n * (u[x+ej|>g, i] - u[x, i])
            fxi += n * (visc * (δui_δxj_b - δui_δxj_a) - (ui_uj_b - ui_uj_a))
        end
        f[x, i] = fxi
    end
end

"Create spectral Poisson solver from setup."
function poissonsolver(setup)
    (; backend, grid, visc) = setup
    T = typeof(visc)
    d = dim(grid)
    n = grid.n

    @assert order isa Order2 "Todo: find ahat for Order4"

    # Since we use rfft, the first dimension is halved
    kmax = ntuple(i -> i == 1 ? div(n, 2) + 1 : n, d)

    # Fourier transform of the discrete Laplacian
    ahat = ntuple(d) do i
        k = 0:kmax[i]-1
        ahat = KernelAbstractions.allocate(backend, T, kmax[i])
        @. ahat = 4 * sinpi(k / n)^2 * n^2
        ahat
    end

    # Placeholders for intermediate results
    phat = KernelAbstractions.allocate(backend, Complex{T}, kmax)
    p = KernelAbstractions.allocate(backend, T, ntuple(Returns(n), d))
    plan = plan_rfft(p)

    function solver!(p)
        # Fourier transform of right hand side
        mul!(phat, plan, p)

        # Solve for coefficients in Fourier space
        if getval(D) == 2
            ax = reshape(ahat[1], :)
            ay = reshape(ahat[2], 1, :)
            @. phat = -phat / (ax + ay)
        else
            ax = reshape(ahat[1], :)
            ay = reshape(ahat[2], 1, :)
            az = reshape(ahat[3], 1, 1, :)
            @. phat = -phat / (ax + ay + az)
        end

        # Pressure is determined up to constant. We set this to 0 (instead of
        # phat[1] / 0 = Inf)
        # Note use of singleton range 1:1 instead of scalar index 1
        # (otherwise CUDA gets annoyed)
        phat[1:1] .= 0

        # Inverse Fourier transform
        ldiv!(p, plan, phat)

        p
    end
end

"Subtract pressure gradient."
pressuregradient!

@kernel function pressuregradient!(grid, u, p)
    x = @index(Global, Cartesian)
    @unroll for i = 1:dim(grid)
        u[x, i] -= δ(grid, p, x, i)
    end
end

"Project velocity field onto divergence-free space."
function project!(u, p, poissonsolve!, setup)
    (; D) = setup

    # Divergence of tentative velocity field
    apply!(divergence!, setup, p, u)

    # Solve the Poisson equation
    poissonsolve!(p)

    # Apply pressure correction term
    apply!(pressuregradient!, setup, u, p)

    u
end

@kernel function vorticity!(setup, ω, u)
    (; D, n) = setup
    per = periodicindex(n)
    i, j = @index(Global, NTuple)
    dudy = n * (u[i, j+1|>per, 1] - u[i, j, 1])
    dvdx = n * (u[i+1|>per, j, 2] - u[i, j, 2])
    ω[i, j] = -dudy + dvdx
end
