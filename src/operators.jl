# Weights for linear combination of finite difference and interpolation stencils
@inline w(::Grid{1}) = (1,)
@inline w(::Grid{2}) = 9 // 8, -1 // 8
@inline w(::Grid{3}) = 150 // 128, -25 // 128, 3 // 128
@inline w(::Grid{4}) = 1225 // 1024, -245 // 1024, 49 // 1024, -5 // 1024
@inline w(::Grid{5}) =
    19845 // 16384, -2205 // 8192, 567 // 8192, -405 // 32768, 35 // 32768

# "Compute finite difference δ_i u(x) of the order given by grid."
# "Compute second order finite difference ``\\delta^{(2n+1) h}_i u(x)`` of width ``(2 n + 1) h``."
@inline δ(u, i, x) = δ(u.grid, u, i, x)
@inline δ(n::Int, u, i, x) = δ(n, u.position[i], u, i, x)
@inline δ(n::Int, ::Stag, u, i, x) = (u[x+(n-1)*e(i)] - u[x-n*e(i)]) / (2n - 1) / dx(u.grid)
@inline δ(n::Int, ::Coll, u, i, x) = (u[x+n*e(i)] - u[x-(n-1)*e(i)]) / (2n - 1) / dx(u.grid)
# sum(ntuple(n -> w(u.grid)[n] * δ(n, u, i, x), u.grid.ho))
@inline δ(::Grid{1}, u, i, x) = δ(1, u, i, x)
@inline δ(::Grid{2}, u, i, x) = w(u.grid)[1] * δ(1, u, i, x) + w(u.grid)[2] * δ(2, u, i, x)
@inline δ(::Grid{3}, u, i, x) =
    w(u.grid)[1] * δ(1, u, i, x) +
    w(u.grid)[2] * δ(2, u, i, x) +
    w(u.grid)[3] * δ(3, u, i, x)
@inline δ(::Grid{4}, u, i, x) =
    w(u.grid)[1] * δ(1, u, i, x) +
    w(u.grid)[2] * δ(2, u, i, x) +
    w(u.grid)[3] * δ(3, u, i, x) +
    w(u.grid)[4] * δ(4, u, i, x)
@inline δ(::Grid{5}, u, i, x) =
    w(u.grid)[1] * δ(1, u, i, x) +
    w(u.grid)[2] * δ(2, u, i, x) +
    w(u.grid)[3] * δ(3, u, i, x) +
    w(u.grid)[4] * δ(4, u, i, x) +
    w(u.grid)[5] * δ(5, u, i, x)

# Interpolate u[i] in direction j. Land in canonical position at x.
@inline interpolate(u, i, x) = interpolate(u.grid, u, i, x)
@inline interpolate(n::Int, u, i, x) = interpolate(n, u.position[i], u, i, x)
@inline interpolate(n::Int, ::Stag, u, i, x) = (u[x-n*e(i)] + u[x+(n-1)*e(i)]) / 2
@inline interpolate(n::Int, ::Coll, u, i, x) = (u[x-(n-1)*e(i)] + u[x+n*e(i)]) / 2
@inline interpolate(::Grid{1}, u, i, x) = interpolate(1, u, i, x)
@inline interpolate(g::Grid{2}, u, i, x) =
    w(g)[1] * interpolate(1, u, i, x) + w(g)[2] * interpolate(2, u, i, x)
@inline interpolate(g::Grid{3}, u, i, x) =
    w(g)[1] * interpolate(1, u, i, x) +
    w(g)[2] * interpolate(2, u, i, x) +
    w(g)[3] * interpolate(3, u, i, x)
@inline interpolate(g::Grid{4}, u, i, x) =
    w(g)[1] * interpolate(1, u, i, x) +
    w(g)[2] * interpolate(2, u, i, x) +
    w(g)[3] * interpolate(3, u, i, x) +
    w(g)[4] * interpolate(4, u, i, x)
@inline interpolate(g::Grid{5}, u, i, x) =
    w(g)[1] * interpolate(1, u, i, x) +
    w(g)[2] * interpolate(2, u, i, x) +
    w(g)[3] * interpolate(3, u, i, x) +
    w(g)[4] * interpolate(4, u, i, x) +
    w(g)[5] * interpolate(5, u, i, x)

"""
Compute divergence of vector field `u`.
Put the result in `div`.
"""
divergence!

@kernel function divergence!(div, u)
    I = @index(Global, Cartesian)
    x, y, z = X(), Y(), Z()
    div[I] = δ(u[x], x, I) + δ(u[y], y, I) + δ(u[z], z, I)
end

"Get convection-diffusion stress tensor component `i,j`."
function stress(u, visc, i, j, x)
    # Non-linear stress
    ui_xj = interpolate(u[i], j, x)
    uj_xi = interpolate(u[j], i, x)
    ui_uj = ui_xj * uj_xi

    # Strain-rate
    δj_ui = δ(u[i], j, x)
    δi_uj = δ(u[j], i, x)
    sij = δj_ui + δi_uj

    # Resulting stress
    ui_uj - visc * sij
end

stresstensor(u, visc) = LazyTensorField(u.grid, stress, u, visc)

@kernel function stresstensor!(r, u, visc)
    I = @index(Global, Cartesian)
    x, y, z = X(), Y(), Z()
    r[x, x][I] = stress(u, visc, x, x, I)
    r[y, y][I] = stress(u, visc, y, y, I)
    r[z, z][I] = stress(u, visc, z, z, I)
    r[x, y][I] = r[y, x][I] = stress(u, visc, x, y, I)
    r[x, z][I] = r[z, x][I] = stress(u, visc, x, z, I)
    r[y, z][I] = r[z, y][I] = stress(u, visc, y, z, I)
end

@kernel function stresstensor_symm!(r, u, visc)
    I = @index(Global, Cartesian)
    x, y, z = X(), Y(), Z()
    r[I, 1] = stress(u, visc, x, x, I)
    r[I, 2] = stress(u, visc, y, y, I)
    r[I, 3] = stress(u, visc, z, z, I)
    r[I, 4] = stress(u, visc, x, y, I)
    r[I, 5] = stress(u, visc, x, z, I)
    r[I, 6] = stress(u, visc, y, z, I)
end

"Approximate the convective force ``\\partial_j (u_i u_j)``."
function conv(u, i, j, x)
    ei, ej = e(i), e(j)
    wg = w(u.grid)
    c = zero(eltype(u.data))
    @unroll for n = 1:getval(u.grid.ho)
        # ui interpolated in direction xj with grid size nh (n = 1, 3, 5, 7, 9)
        ui_nxj_a = interpolate(n, u[i], j, x - (n - 1 + (i != j)) * ej)
        ui_nxj_b = interpolate(n, u[i], j, x + (n - 1 + (i == j)) * ej)

        # uj interpolated in direction xi with order
        uj_xi_na = interpolate(u[j], i, x - (n - 1 + (i != j)) * ej)
        uj_xi_nb = interpolate(u[j], i, x + (n - 1 + (i == j)) * ej)

        # Tensor product -- see  Morinishi 1998 eq. (112)
        ui_uj_na = ui_nxj_a * uj_xi_na
        ui_uj_nb = ui_nxj_b * uj_xi_nb

        # Divergence of tensor: Lands at canonical position of ui in volume x
        # coefficient computed in script
        c += wg[n] * (ui_uj_nb - ui_uj_na) / (2n - 1) / dx(u.grid)
    end
    c
end

"Laplacian."
function lap(u, i, j, x)
    g = u.grid
    o = order(g)
    stencil = map(eltype(u.data), laplace_stencil(g))
    diff = zero(eltype(u.data))
    @unroll for k = 1:(2o-1)
        diff += stencil[k] * u[i][x+(k-o)*e(j)]
    end
    diff
end

@inline diffusion(u, visc, i, j, x) = visc * lap(u, i, j, x)
@inline convdiff(u, visc, i, j, x) = -conv(u, i, j, x) + visc * lap(u, i, j, x)

"""
Compute ``v_i(I) = \\sum_j f(args..., i, j, I)``.
"""
@kernel function tensorapply!(f, v, args...)
    I = @index(Global, Cartesian)
    x, y, z = X(), Y(), Z()
    v[x][I] = f(args..., x, x, I) + f(args..., x, y, I) + f(args..., x, z, I)
    v[y][I] = f(args..., y, x, I) + f(args..., y, y, I) + f(args..., y, z, I)
    v[z][I] = f(args..., z, x, I) + f(args..., z, y, I) + f(args..., z, z, I)
end

"""
Compute ``v_i(I) = v_i(I) + \\sum_j f(args..., i, j, I)``.
"""
@kernel function tensorapply_add!(f, v, args...)
    I = @index(Global, Cartesian)
    x, y, z = X(), Y(), Z()
    v[x][I] += f(args..., x, x, I) + f(args..., x, y, I) + f(args..., x, z, I)
    v[y][I] += f(args..., y, x, I) + f(args..., y, y, I) + f(args..., y, z, I)
    v[z][I] += f(args..., z, x, I) + f(args..., z, y, I) + f(args..., z, z, I)
end

@kernel function materialize_kernel!(v, u)
    I = @index(Global, Cartesian)
    v[I] = u[I]
end

materialize!(v::ScalarField, u::LazyScalarField) = apply!(materialize_kernel!, v.grid, v, u)
# materialize!(v::VectorField, u::LazyVectorField)
# materialize!(v::TensorField, u::LazyTensorField)

laplace_stencil(g::Grid{1}) = (1, -2, 1) ./ dx(g)^2
laplace_stencil(g::Grid{2}) = (1, -54, 783, -1460, 783, -54, 1) .// 576 ./ dx(g)^2
laplace_stencil(g::Grid{3}) =
    (81, -2250, 56125, -603000, 5627250, -10156412, 5627250, -603000, 56125, -2250, 81) .//
    1920^2 ./ dx(g)^2
laplace_stencil(g::Grid{4}) =
    (
        25 // 51380224,
        -7 // 524288,
        15953 // 78643200,
        -2513 // 786432,
        291865 // 9437184,
        -112105 // 524288,
        1702323 // 1048576,
        -4154746429 // 1445068800,
        1702323 // 1048576,
        -112105 // 524288,
        291865 // 9437184,
        -2513 // 786432,
        15953 // 78643200,
        -7 // 524288,
        25 // 51380224,
    ) ./ dx(g)^2
laplace_stencil(g::Grid{5}) =
    (
        1225 // 86973087744,
        -225 // 536870912,
        336897 // 52613349376,
        -14129 // 201326592,
        5341581 // 6710886400,
        -946071 // 134217728,
        36949451 // 805306368,
        -16857981 // 67108864,
        905696703 // 536870912,
        -78593583110603 // 26635508121600,
        905696703 // 536870912,
        -16857981 // 67108864,
        36949451 // 805306368,
        -946071 // 134217728,
        5341581 // 6710886400,
        -14129 // 201326592,
        336897 // 52613349376,
        -225 // 536870912,
        1225 // 86973087744,
    ) ./ dx(g)^2

"Merge stencil periodically if the stencil is longer than the grid size `n`."
mergestencil(s, n) =
    if length(s) > n
        a, b = s[1:n], s[(n+1):end]
        b = mergestencil(b, n)
        a[1:length(b)] .+= b
        a
    else
        s
    end

"Create spectral Poisson solver from grid."
function poissonsolver(grid)
    (; L, n, backend) = grid
    T = typeof(L)

    # Since we use rfft, the first dimension is halved
    kmax = div(n, 2) + 1, n, n

    # Discrete Laplacian stencil -- pad with zeros to full size
    a_cpu = laplace_stencil(grid) |> collect .|> T
    if length(a_cpu) > n
        # This is only necessary for convergence plot with small grid
        @warn "Laplacian stencil is longer than grid size. Merging."
        a_cpu = mergestencil(a_cpu, n)
    else
        a_cpu = vcat(a_cpu, zeros(T, n - length(a_cpu)))
    end
    a = adapt(backend, a_cpu)

    # Fourier transform of the discrete Laplacian
    ahat = fft(a)

    # Since we use rfft, the first dimension is halved
    ahat = map(k -> abs.(ahat[1:k]), kmax)

    # ahat = ntuple(3) do i
    #     k = 0:(kmax[i]-1)
    #     ahat = KernelAbstractions.allocate(backend, T, kmax[i])
    #     @. ahat = 4 / dx(grid)^2 * sinpi(k / n)^2
    #     ahat
    # end

    # Placeholders for intermediate results
    phat = KernelAbstractions.allocate(backend, Complex{T}, kmax)
    p = KernelAbstractions.allocate(backend, T, n, n, n)
    plan = plan_rfft(p)

    (; plan, phat, ahat)
end

function poissonsolve!(p, cache)
    (; plan, phat, ahat) = cache

    # Fourier transform of right hand side
    mul!(phat, plan, p.data)

    # Solve for coefficients in Fourier space
    ax = reshape(ahat[1], :)
    ay = reshape(ahat[2], 1, :)
    az = reshape(ahat[3], 1, 1, :)
    @. phat = -phat / (ax + ay + az)

    # Pressure is determined up to constant. We set this to 0 (instead of
    # phat[1] / 0 = Inf)
    # Note use of singleton range 1:1 instead of scalar index 1
    # (otherwise CUDA gets annoyed)
    phat[1:1] .= 0

    # Inverse Fourier transform
    ldiv!(p.data, plan, phat)

    p
end

"Subtract pressure gradient."
pressuregradient!

@kernel function pressuregradient!(u, p)
    I = @index(Global, Cartesian)
    x, y, z = X(), Y(), Z()
    u[x][I] -= δ(p, x, I)
    u[y][I] -= δ(p, y, I)
    u[z][I] -= δ(p, z, I)
end

"Project velocity field onto divergence-free space."
function project!(u, p, cache)
    # Divergence of tentative velocity field
    apply!(divergence!, u.grid, p, u)

    # Solve the Poisson equation
    poissonsolve!(p, cache)

    # Apply pressure correction term
    apply!(pressuregradient!, u.grid, u, p)

    u
end

"""
Get the following dimensional scale numbers [popeTurbulentFlows2000](@cite):

- Velocity ``u_\\text{avg} = \\langle u_i u_i \\rangle^{1/2}``
- Dissipation rate ``\\epsilon = 2 \\nu \\langle S_{ij} S_{ij} \\rangle``
- Kolmolgorov length scale ``\\eta = (\\frac{\\nu^3}{\\epsilon})^{1/4}``
- Taylor length scale ``\\lambda = (\\frac{5 \\nu}{\\epsilon})^{1/2} u_\\text{avg}``
- Taylor-scale Reynolds number ``Re_\\lambda = \\frac{\\lambda u_\\text{avg}}{\\sqrt{3} \\nu}``
- Integral length scale ``L = \\frac{3 \\pi}{2 u_\\text{avg}^2} \\int_0^\\infty \\frac{E(k)}{k} \\, \\mathrm{d} k``
- Large-eddy turnover time ``\\tau = \\frac{L}{u_\\text{avg}}``
"""
function get_scale_numbers(u, visc)
    (; grid) = u
    (; n) = grid
    T = eltype(u)
    uavg = sqrt(sum(abs2, u.data) / grid.n^3)
    TKE = uavg^2 / 2
    diss = ScalarField(grid)
    apply!(dissipation!, grid, diss, u, visc)
    D = sum(diss.data) / length(diss)
    eta = (visc^3 / D)^T(1 / 4)
    λ = sqrt(visc / D) * uavg
    L = uavg^3 / D
    # L = if dointegral
    #     K = div(n, 2)
    #     uhat = fft(u.data, 1:3)
    #     uhat = uhat[ntuple(i->1:K, 3)..., :]
    #     e = abs2.(uhat) ./ (2 * (n^3)^2)
    #     kx = reshape(0:(K-1), :)
    #     ky = reshape(0:(K-1), 1, :)
    #     kz = reshape(0:(K-1), 1, 1, :)
    #     @. e = e / sqrt(kx^2 + ky^2 + kz^2)
    #     e = sum(e; dims = 4)
    #     # Remove k=(0,...,0) component
    #     # Note use of singleton range 1:1 instead of scalar index 1
    #     # (otherwise CUDA gets annoyed)
    #     e[1:1] .= 0
    #     T(3π) / 2 / uavg^2 * sum(e)
    # else
    #     nothing
    # end
    t_int = L / uavg
    t_tay = λ / uavg
    t_kol = visc / D |> sqrt
    Re_int = L * uavg / visc
    Re_tay = λ * uavg / visc
    Re_kol = eta * uavg / visc
    (; uavg, D, L, λ, eta, t_int, t_tay, t_kol, Re_int, Re_tay, Re_kol)
end

@kernel function collocate_velocity!(ucoll, u)
    I = @index(Global, Cartesian)
    x, y, z = X(), Y(), Z()
    ux = interpolate(u[x], x, I)
    uy = interpolate(u[y], y, I)
    uz = interpolate(u[z], z, I)
    ucoll[I] = SVector(ux, uy, uz)
end

@kernel function velocitynorm!(unorm, u)
    I = @index(Global, Cartesian)
    x, y, z = X(), Y(), Z()
    ux = interpolate(u[x], x, I)
    uy = interpolate(u[y], y, I)
    uz = interpolate(u[z], z, I)
    unorm[I] = sqrt(ux^2 + uy^2 + uz^2)
end
