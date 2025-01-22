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

# Vector field gradient δu[i] / δx[j].
@inline δ1(g::Grid, u, x, i, j) =
    g.n * (u[x+(i!=j)*e(g, j)|>g, i] - u[x-(i==j)*e(g, j)|>g, i])
@inline δ3(g::Grid, u, x, i, j) =
    g.n * (u[x+(1+(i!=j))*e(g, j)|>g, i] - u[x-(1+(i==j))*e(g, j)|>g, i]) / 3
@inline δ(g::Grid{2}, u, x, i, j) = δ1(g, u, x, i, j)
@inline δ(g::Grid{4}, u, x, i, j) = 9 * δ1(g, u, x, i, j) / 8 - δ3(g, u, x, i, j) / 8

# Scalar field gradient δp / δx[j].
@inline δ1(g::Grid, p, x, j) = g.n * (p[x+e(g, j)|>g] - p[x])
@inline δ3(g::Grid, p, x, j) = g.n * (p[x+2*e(g, j)|>g] - p[x-1e(g, j)|>g]) / 3
@inline δ(g::Grid{2}, p, x, j) = δ1(g, p, x, j)
@inline δ(g::Grid{4}, p, x, j) = 9 * δ1(g, p, x, j) / 8 - δ3(g, p, x, j) / 8

# Interpolate u[i] in direction j. Land in canonical position at x.
@inline pol1(g::Grid, u, x, i, j) =
    (u[x-(i==j)*e(g, j)|>g, i] + u[x+(i!=j)*e(g, j)|>g, i]) / 2
@inline pol3(g::Grid, u, x, i, j) =
    (u[x-(1+(i==j))*e(g, j)|>g, i] + u[x+(2-(i==j))*e(g, j)|>g, i]) / 2
@inline pol(g::Grid{2}, u, x, i, j) = pol1(g, u, x, i, j)
@inline pol(g::Grid{4}, u, x, i, j) = 9 * pol1(g, u, x, i, j) / 8 - pol3(g, u, x, i, j) / 8

"""
Compute divergence of vector field `u`.
Put the result in `div`.
"""
divergence!

@kernel function divergence!(g::Grid, div, u)
    x = @index(Global, Cartesian)
    divx = zero(eltype(div))
    @unroll for j = 1:dim(g)
        divx += δ(g, u, x, j, j)
    end
    div[x] = divx
end

"Approximate the convective force ``\\partial_j (u_i u_j)``."
function convterm end
@inline function convterm(g::Grid{2}, u, x, i, j)
    (; n) = g
    ei, ej = e(g, i), e(g, j)

    # a: left in xj
    # b: right in xj
    ui_xj_a = pol(g, u, x - (i != j) * ej |> g, i, j)
    ui_xj_b = pol(g, u, x + (i == j) * ej |> g, i, j)
    uj_xi_a = pol(g, u, x - (i != j) * ej |> g, j, i)
    uj_xi_b = pol(g, u, x + (i == j) * ej |> g, j, i)
    ui_uj_a = ui_xj_a * uj_xi_a
    ui_uj_b = ui_xj_b * uj_xi_b

    n * (ui_uj_b - ui_uj_a)
end

@inline function convterm(g::Grid{4}, u, x, i, j)
    ei, ej = e(g, i), e(g, j)

    # (a)a: (twice) left in xj
    # (b)b: (twice) right in xj
    # 1: second order
    # 3: fourth order

    # ui interpolated in direction xj
    ui_xj_1_a = pol1(g, u, x - (i != j) * ej |> g, i, j)
    ui_xj_1_b = pol1(g, u, x + (i == j) * ej |> g, i, j)
    ui_xj_3_aa = pol3(g, u, x - (1 + (i != j)) * ej |> g, i, j)
    ui_xj_3_bb = pol3(g, u, x + (1 + (i == j)) * ej |> g, i, j)

    # uj interpolated in direction xi
    uj_xi_aa = pol(g, u, x - (1 + (i != j)) * ej |> g, j, i)
    uj_xi_a = pol(g, u, x - (i != j) * ej |> g, j, i)
    uj_xi_b = pol(g, u, x + (i == j) * ej |> g, j, i)
    uj_xi_bb = pol(g, u, x + (1 + (i == j)) * ej |> g, j, i)

    # Tensor product -- see  Morinishi 1998 eq. (101)
    ui_uj_aa = ui_xj_3_aa * uj_xi_aa
    ui_uj_a = ui_xj_1_a * uj_xi_a
    ui_uj_b = ui_xj_1_b * uj_xi_b
    ui_uj_bb = ui_xj_3_bb * uj_xi_bb

    # Divergence of tensor: Lands at canonical position of ui in volume x
    # see  Morinishi 1998 eq. (101)
    g.n * (9 * (ui_uj_b - ui_uj_a) / 8 - (ui_uj_bb - ui_uj_aa) / 3 / 8)
end

diffusionterm(g::Grid{2}, u, x, i, j) =
    g.n^2 * (u[x-e(g, j)|>g, i] - 2 * u[x, i] + u[x+e(g, j)|>g, i])

function diffusionterm(g::Grid{4}, u, x, i, j)
    stencil = (1, -54, 783, -1460, 783, -54, 1) ./ 576 .* g.n^2 .|> eltype(u)
    diff = zero(eltype(u))
    @unroll for k = 1:7
        diff += stencil[k] * u[x+(k-4)*e(g, j)|>g, i]
    end
    diff
end

"""
Compute convection-diffusion force from velocity `u`.
Add the force field to `f`.
"""
convectiondiffusion!

@kernel function convectiondiffusion!(g::Grid, f, u, visc)
    T = typeof(visc)
    dims = 1:dim(g)
    x = @index(Global, Cartesian)
    @unroll for i in dims
        fxi = f[x, i]
        @unroll for j in dims
            fxi -= convterm(g, u, x, i, j)
            fxi += visc * diffusionterm(g, u, x, i, j)
        end
        f[x, i] = fxi
    end
end

laplace_stencil(g::Grid{2}) = [1, -2, 1] * g.n^2
laplace_stencil(g::Grid{4}) = [1, -54, 783, -1460, 783, -54, 1] / 576 * g.n^2

"Create spectral Poisson solver from setup."
function poissonsolver(setup)
    (; backend, grid, visc) = setup
    T = typeof(visc)
    d = dim(grid)
    n = grid.n

    # Since we use rfft, the first dimension is halved
    kmax = ntuple(i -> i == 1 ? div(n, 2) + 1 : n, d)

    # Discrete Laplacian stencil -- pad with zeros to full size
    a_cpu = laplace_stencil(grid)
    a_cpu = vcat(a_cpu, zeros(T, n - length(a_cpu)))
    a = adapt(backend, a_cpu)

    # Fourier transform of the discrete Laplacian
    ahat = fft(a)

    # Since we use rfft, the first dimension is halved
    ahat = map(k -> abs.(ahat[1:k]), kmax)

    # ahat = ntuple(d) do i
    #     k = 0:(kmax[i]-1)
    #     ahat = KernelAbstractions.allocate(backend, T, kmax[i])
    #     @. ahat = 4 * n^2 * sinpi(k / n)^2
    #     ahat
    # end

    # Placeholders for intermediate results
    phat = KernelAbstractions.allocate(backend, Complex{T}, kmax)
    p = KernelAbstractions.allocate(backend, T, ntuple(Returns(n), d))
    plan = plan_rfft(p)

    function solver!(p)
        # Fourier transform of right hand side
        mul!(phat, plan, p)

        # Solve for coefficients in Fourier space
        if dim(grid) == 2
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
    # Divergence of tentative velocity field
    apply!(divergence!, setup, p, u)

    # Solve the Poisson equation
    poissonsolve!(p)

    # Apply pressure correction term
    apply!(pressuregradient!, setup, u, p)

    u
end

@kernel function vorticity!(g::Grid, ω, u)
    x = @index(Global, Cartesian)
    δu1_δx2 = δ(g, u, x, 1, 2)
    δu2_δx1 = δ(g, u, x, 2, 1)
    ω[x] = -δu1_δx2 + δu2_δx1
end
