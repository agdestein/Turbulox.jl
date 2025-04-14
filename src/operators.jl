# Weights for linear combination of finite difference and interpolation stencils

# Order 4
const w4_1 = 9 // 8
const w4_3 = -1 // 8

# Order 6
const w6_1 = 150 // 128
const w6_3 = -25 // 128
const w6_5 = 3 // 128

# Order 8
const w8_1 = 1225 // 1024
const w8_3 = -245 // 1024
const w8_5 = 49 // 1024
const w8_7 = -5 // 1024

# Order 10
const w10_1 = 19845 // 16384
const w10_3 = -2205 // 8192
const w10_5 = 567 // 8192
const w10_7 = -405 // 32768
const w10_9 = 35 // 32768

# Vector field gradient δu[i] / δx[j].
@inline δ1(g::Grid, u, x, i, j) =
    (u[x+(i!=j)*e(g, j)|>g, i] - u[x-(i==j)*e(g, j)|>g, i]) / dx(g)
@inline δ3(g::Grid, u, x, i, j) =
    (u[x+(1+(i!=j))*e(g, j)|>g, i] - u[x-(1+(i==j))*e(g, j)|>g, i]) / 3 / dx(g)
@inline δ5(g::Grid, u, x, i, j) =
    (u[x+(2+(i!=j))*e(g, j)|>g, i] - u[x-(2+(i==j))*e(g, j)|>g, i]) / 5 / dx(g)
@inline δ7(g::Grid, u, x, i, j) =
    (u[x+(3+(i!=j))*e(g, j)|>g, i] - u[x-(3+(i==j))*e(g, j)|>g, i]) / 7 / dx(g)
@inline δ9(g::Grid, u, x, i, j) =
    (u[x+(4+(i!=j))*e(g, j)|>g, i] - u[x-(4+(i==j))*e(g, j)|>g, i]) / 9 / dx(g)
@inline δ(g::Grid{2}, u, x, i, j) = δ1(g, u, x, i, j)
@inline δ(g::Grid{4}, u, x, i, j) = w4_1 * δ1(g, u, x, i, j) + w4_3 * δ3(g, u, x, i, j)
@inline δ(g::Grid{6}, u, x, i, j) =
    w6_1 * δ1(g, u, x, i, j) + w6_3 * δ3(g, u, x, i, j) + w6_5 * δ5(g, u, x, i, j)
@inline δ(g::Grid{8}, u, x, i, j) =
    w8_1 * δ1(g, u, x, i, j) +
    w8_3 * δ3(g, u, x, i, j) +
    w8_5 * δ5(g, u, x, i, j) +
    w8_7 * δ7(g, u, x, i, j)
@inline δ(g::Grid{10}, u, x, i, j) =
    w10_1 * δ1(g, u, x, i, j) +
    w10_3 * δ3(g, u, x, i, j) +
    w10_5 * δ5(g, u, x, i, j) +
    w10_7 * δ7(g, u, x, i, j) +
    w10_9 * δ9(g, u, x, i, j)

# Scalar field gradient δp / δx[j].
@inline δ1(g::Grid, p, x, j) = (p[x+e(g, j)|>g] - p[x]) / dx(g)
@inline δ3(g::Grid, p, x, j) = (p[x+2*e(g, j)|>g] - p[x-1e(g, j)|>g]) / 3 / dx(g)
@inline δ5(g::Grid, p, x, j) = (p[x+3*e(g, j)|>g] - p[x-2e(g, j)|>g]) / 5 / dx(g)
@inline δ7(g::Grid, p, x, j) = (p[x+4*e(g, j)|>g] - p[x-3e(g, j)|>g]) / 7 / dx(g)
@inline δ9(g::Grid, p, x, j) = (p[x+5*e(g, j)|>g] - p[x-4e(g, j)|>g]) / 9 / dx(g)
@inline δ(g::Grid{2}, p, x, j) = δ1(g, p, x, j)
@inline δ(g::Grid{4}, p, x, j) = w4_1 * δ1(g, p, x, j) + w4_3 * δ3(g, p, x, j)
@inline δ(g::Grid{6}, p, x, j) =
    w6_1 * δ1(g, p, x, j) + w6_3 * δ3(g, p, x, j) + w6_5 * δ5(g, p, x, j)
@inline δ(g::Grid{8}, p, x, j) =
    w8_1 * δ1(g, p, x, j) +
    w8_3 * δ3(g, p, x, j) +
    w8_5 * δ5(g, p, x, j) +
    w8_7 * δ7(g, p, x, j)
@inline δ(g::Grid{10}, p, x, j) =
    w10_1 * δ1(g, p, x, j) +
    w10_3 * δ3(g, p, x, j) +
    w10_5 * δ5(g, p, x, j) +
    w10_7 * δ7(g, p, x, j) +
    w10_9 * δ9(g, p, x, j)

# Interpolate u[i] in direction j. Land in canonical position at x.
@inline pol1(g::Grid, u, x, i, j) =
    (u[x-(i==j)*e(g, j)|>g, i] + u[x+(i!=j)*e(g, j)|>g, i]) / 2
@inline pol3(g::Grid, u, x, i, j) =
    (u[x-(1+(i==j))*e(g, j)|>g, i] + u[x+(1+(i!=j))*e(g, j)|>g, i]) / 2
@inline pol5(g::Grid, u, x, i, j) =
    (u[x-(2+(i==j))*e(g, j)|>g, i] + u[x+(2+(i!=j))*e(g, j)|>g, i]) / 2
@inline pol7(g::Grid, u, x, i, j) =
    (u[x-(3+(i==j))*e(g, j)|>g, i] + u[x+(3+(i!=j))*e(g, j)|>g, i]) / 2
@inline pol9(g::Grid, u, x, i, j) =
    (u[x-(4+(i==j))*e(g, j)|>g, i] + u[x+(4+(i!=j))*e(g, j)|>g, i]) / 2
@inline pol(g::Grid{2}, u, x, i, j) = pol1(g, u, x, i, j)
@inline pol(g::Grid{4}, u, x, i, j) =
    w4_1 * pol1(g, u, x, i, j) + w4_3 * pol3(g, u, x, i, j)
@inline pol(g::Grid{6}, u, x, i, j) =
    w6_1 * pol1(g, u, x, i, j) + w6_3 * pol3(g, u, x, i, j) + w6_5 * pol5(g, u, x, i, j)
@inline pol(g::Grid{8}, u, x, i, j) =
    w8_1 * pol1(g, u, x, i, j) +
    w8_3 * pol3(g, u, x, i, j) +
    w8_5 * pol5(g, u, x, i, j) +
    w8_7 * pol7(g, u, x, i, j)
@inline pol(g::Grid{10}, u, x, i, j) =
    w10_1 * pol1(g, u, x, i, j) +
    w10_3 * pol3(g, u, x, i, j) +
    w10_5 * pol5(g, u, x, i, j) +
    w10_7 * pol7(g, u, x, i, j) +
    w10_9 * pol9(g, u, x, i, j)

"""
Compute divergence of vector field `u`.
Put the result in `div`.
"""
divergence!

@kernel function divergence!(grid, div, u)
    x = @index(Global, Cartesian)
    divx = zero(eltype(div))
    @unroll for j = 1:dim(grid)
        divx += δ(grid, u, x, j, j)
    end
    div[x] = divx
end


"Get convection-diffusion stress tensor component `i,j`."
@inline function stress(g::Grid, u, x, i, j, visc)
    # Non-linear stress
    ui_xj = pol(g, u, x, i, j)
    uj_xi = pol(g, u, x, j, i)
    ui_uj = ui_xj * uj_xi

    # Strain-rate
    δj_ui = δ(g, u, x, i, j)
    δi_uj = δ(g, u, x, j, i)
    sij = (δj_ui + δi_uj) / 2

    # Resulting stress
    ui_uj - 2 * visc * sij
end

@kernel function stresstensor!(g::Grid, r, u, visc)
    x = @index(Global, Cartesian)
    r[x, 1, 1] = stress(g, u, x, 1, 1, visc)
    r[x, 2, 2] = stress(g, u, x, 2, 2, visc)
    r[x, 3, 3] = stress(g, u, x, 3, 3, visc)
    r[x, 1, 2] = r[x, 2, 1] = stress(g, u, x, 1, 2, visc)
    r[x, 1, 3] = r[x, 3, 1] = stress(g, u, x, 1, 3, visc)
    r[x, 2, 3] = r[x, 3, 2] = stress(g, u, x, 2, 3, visc)
end

@kernel function stresstensor_symm!(g::Grid, r, u, visc)
    x = @index(Global, Cartesian)
    r[x, 1] = stress(g, u, x, 1, 1, visc)
    r[x, 2] = stress(g, u, x, 2, 2, visc)
    r[x, 3] = stress(g, u, x, 3, 3, visc)
    r[x, 4] = stress(g, u, x, 1, 2, visc)
    r[x, 5] = stress(g, u, x, 1, 3, visc)
    r[x, 6] = stress(g, u, x, 2, 3, visc)
end

"Approximate the convective force ``\\partial_j (u_i u_j)``."
function convterm end

@inline function convterm(g::Grid{2}, u, x, i, j)
    ei, ej = e(g, i), e(g, j)

    # a: left in xj
    # b: right in xj
    ui_xj_a = pol(g, u, x - (i != j) * ej |> g, i, j)
    ui_xj_b = pol(g, u, x + (i == j) * ej |> g, i, j)
    uj_xi_a = pol(g, u, x - (i != j) * ej |> g, j, i)
    uj_xi_b = pol(g, u, x + (i == j) * ej |> g, j, i)
    ui_uj_a = ui_xj_a * uj_xi_a
    ui_uj_b = ui_xj_b * uj_xi_b

    (ui_uj_b - ui_uj_a) / dx(g)
end

@inline function convterm(g::Grid{4}, u, x, i, j)
    ei, ej = e(g, i), e(g, j)

    # (n)a: (n/2 times) left in xj
    # (n)b: (n/2 times) right in xj
    # 1: grid size h
    # 3: grid size 3h

    # ui interpolated in direction xj
    ui_3xj_3a = pol3(g, u, x - (1 + (i != j)) * ej |> g, i, j)
    ui_1xj_1a = pol1(g, u, x - (i != j) * ej |> g, i, j)
    ui_1xj_1b = pol1(g, u, x + (i == j) * ej |> g, i, j)
    ui_3xj_3b = pol3(g, u, x + (1 + (i == j)) * ej |> g, i, j)

    # uj interpolated in direction xi
    uj_xi_3a = pol(g, u, x - (1 + (i != j)) * ej |> g, j, i)
    uj_xi_1a = pol(g, u, x - (i != j) * ej |> g, j, i)
    uj_xi_1b = pol(g, u, x + (i == j) * ej |> g, j, i)
    uj_xi_3b = pol(g, u, x + (1 + (i == j)) * ej |> g, j, i)

    # Tensor product -- see  Morinishi 1998 eq. (101)
    ui_uj_3a = ui_3xj_3a * uj_xi_3a
    ui_uj_1a = ui_1xj_1a * uj_xi_1a
    ui_uj_1b = ui_1xj_1b * uj_xi_1b
    ui_uj_3b = ui_3xj_3b * uj_xi_3b

    # Divergence of tensor: Lands at canonical position of ui in volume x
    # see  Morinishi 1998 eq. (101)
    (w4_1 * (ui_uj_1b - ui_uj_1a) + w4_3 * (ui_uj_3b - ui_uj_3a) / 3) / dx(g)
end

@inline function convterm(g::Grid{6}, u, x, i, j)
    ei, ej = e(g, i), e(g, j)

    # (n)a: (n/2 times) left in xj
    # (n)b: (n/2 times) right in xj
    # 1: grid size h
    # 3: grid size 3h
    # 5: grid size 5h

    # ui interpolated in direction xj
    ui_1xj_1a = pol1(g, u, x - (i != j) * ej |> g, i, j)
    ui_1xj_1b = pol1(g, u, x + (i == j) * ej |> g, i, j)
    ui_3xj_3a = pol3(g, u, x - (1 + (i != j)) * ej |> g, i, j)
    ui_3xj_3b = pol3(g, u, x + (1 + (i == j)) * ej |> g, i, j)
    ui_5xj_5a = pol5(g, u, x - (2 + (i != j)) * ej |> g, i, j)
    ui_5xj_5b = pol5(g, u, x + (2 + (i == j)) * ej |> g, i, j)

    # uj interpolated in direction xi
    uj_xi_5a = pol(g, u, x - (2 + (i != j)) * ej |> g, j, i)
    uj_xi_3a = pol(g, u, x - (1 + (i != j)) * ej |> g, j, i)
    uj_xi_1a = pol(g, u, x - (i != j) * ej |> g, j, i)
    uj_xi_1b = pol(g, u, x + (i == j) * ej |> g, j, i)
    uj_xi_3b = pol(g, u, x + (1 + (i == j)) * ej |> g, j, i)
    uj_xi_5b = pol(g, u, x + (2 + (i == j)) * ej |> g, j, i)

    # Tensor product -- see  Morinishi 1998 eq. (112)
    ui_uj_5a = ui_5xj_5a * uj_xi_5a
    ui_uj_3a = ui_3xj_3a * uj_xi_3a
    ui_uj_1a = ui_1xj_1a * uj_xi_1a
    ui_uj_1b = ui_1xj_1b * uj_xi_1b
    ui_uj_3b = ui_3xj_3b * uj_xi_3b
    ui_uj_5b = ui_5xj_5b * uj_xi_5b

    # Divergence of tensor: Lands at canonical position of ui in volume x
    # see  Morinishi 1998 eq. (112)
    (
        w6_1 * (ui_uj_1b - ui_uj_1a) +
        w6_3 * (ui_uj_3b - ui_uj_3a) / 3 +
        w6_5 * (ui_uj_5b - ui_uj_5a) / 5
    ) / dx(g)
end

@inline function convterm(g::Grid{8}, u, x, i, j)
    ei, ej = e(g, i), e(g, j)

    # (n)a: (n/2 times) left in xj
    # (n)b: (n/2 times) right in xj
    # 1: grid size h
    # 3: grid size 3h
    # 5: grid size 5h
    # 7: grid size 7h

    # ui interpolated in direction xj
    ui_7xj_7a = pol7(g, u, x - (3 + (i != j)) * ej |> g, i, j)
    ui_5xj_5a = pol5(g, u, x - (2 + (i != j)) * ej |> g, i, j)
    ui_3xj_3a = pol3(g, u, x - (1 + (i != j)) * ej |> g, i, j)
    ui_1xj_1a = pol1(g, u, x - (i != j) * ej |> g, i, j)
    ui_1xj_1b = pol1(g, u, x + (i == j) * ej |> g, i, j)
    ui_3xj_3b = pol3(g, u, x + (1 + (i == j)) * ej |> g, i, j)
    ui_5xj_5b = pol5(g, u, x + (2 + (i == j)) * ej |> g, i, j)
    ui_7xj_7b = pol7(g, u, x + (3 + (i == j)) * ej |> g, i, j)

    # uj interpolated in direction xi
    uj_xi_7a = pol(g, u, x - (3 + (i != j)) * ej |> g, j, i)
    uj_xi_5a = pol(g, u, x - (2 + (i != j)) * ej |> g, j, i)
    uj_xi_3a = pol(g, u, x - (1 + (i != j)) * ej |> g, j, i)
    uj_xi_1a = pol(g, u, x - (i != j) * ej |> g, j, i)
    uj_xi_1b = pol(g, u, x + (i == j) * ej |> g, j, i)
    uj_xi_3b = pol(g, u, x + (1 + (i == j)) * ej |> g, j, i)
    uj_xi_5b = pol(g, u, x + (2 + (i == j)) * ej |> g, j, i)
    uj_xi_7b = pol(g, u, x + (3 + (i == j)) * ej |> g, j, i)

    # Tensor product -- see  Morinishi 1998 eq. (112)
    ui_uj_7a = ui_7xj_7a * uj_xi_7a
    ui_uj_5a = ui_5xj_5a * uj_xi_5a
    ui_uj_3a = ui_3xj_3a * uj_xi_3a
    ui_uj_1a = ui_1xj_1a * uj_xi_1a
    ui_uj_1b = ui_1xj_1b * uj_xi_1b
    ui_uj_3b = ui_3xj_3b * uj_xi_3b
    ui_uj_5b = ui_5xj_5b * uj_xi_5b
    ui_uj_7b = ui_7xj_7b * uj_xi_7b

    # Divergence of tensor: Lands at canonical position of ui in volume x
    # coefficient computed in script
    (
        w8_1 * (ui_uj_1b - ui_uj_1a) +
        w8_3 * (ui_uj_3b - ui_uj_3a) / 3 +
        w8_5 * (ui_uj_5b - ui_uj_5a) / 5 +
        w8_7 * (ui_uj_7b - ui_uj_7a) / 7
    ) / dx(g)
end

@inline function convterm(g::Grid{10}, u, x, i, j)
    ei, ej = e(g, i), e(g, j)

    # (n)a: (n/2 times) left in xj
    # (n)b: (n/2 times) right in xj
    # 1: grid size h
    # 3: grid size 3h
    # 5: grid size 5h
    # 7: grid size 7h
    # 9: grid size 9h

    # ui interpolated in direction xj with grid size ah (a = 1, 3, 5, 7, 9)
    ui_9xj_9a = pol9(g, u, x - (4 + (i != j)) * ej |> g, i, j)
    ui_7xj_7a = pol7(g, u, x - (3 + (i != j)) * ej |> g, i, j)
    ui_5xj_5a = pol5(g, u, x - (2 + (i != j)) * ej |> g, i, j)
    ui_3xj_3a = pol3(g, u, x - (1 + (i != j)) * ej |> g, i, j)
    ui_1xj_1a = pol1(g, u, x - (i != j) * ej |> g, i, j)
    ui_1xj_1b = pol1(g, u, x + (i == j) * ej |> g, i, j)
    ui_3xj_3b = pol3(g, u, x + (1 + (i == j)) * ej |> g, i, j)
    ui_5xj_5b = pol5(g, u, x + (2 + (i == j)) * ej |> g, i, j)
    ui_7xj_7b = pol7(g, u, x + (3 + (i == j)) * ej |> g, i, j)
    ui_9xj_9b = pol9(g, u, x + (4 + (i == j)) * ej |> g, i, j)

    # uj interpolated in direction xi with order 10
    uj_xi_9a = pol(g, u, x - (4 + (i != j)) * ej |> g, j, i)
    uj_xi_7a = pol(g, u, x - (3 + (i != j)) * ej |> g, j, i)
    uj_xi_5a = pol(g, u, x - (2 + (i != j)) * ej |> g, j, i)
    uj_xi_3a = pol(g, u, x - (1 + (i != j)) * ej |> g, j, i)
    uj_xi_1a = pol(g, u, x - (i != j) * ej |> g, j, i)
    uj_xi_1b = pol(g, u, x + (i == j) * ej |> g, j, i)
    uj_xi_3b = pol(g, u, x + (1 + (i == j)) * ej |> g, j, i)
    uj_xi_5b = pol(g, u, x + (2 + (i == j)) * ej |> g, j, i)
    uj_xi_7b = pol(g, u, x + (3 + (i == j)) * ej |> g, j, i)
    uj_xi_9b = pol(g, u, x + (4 + (i == j)) * ej |> g, j, i)

    # Tensor product -- see  Morinishi 1998 eq. (112)
    ui_uj_9a = ui_9xj_9a * uj_xi_9a
    ui_uj_7a = ui_7xj_7a * uj_xi_7a
    ui_uj_5a = ui_5xj_5a * uj_xi_5a
    ui_uj_3a = ui_3xj_3a * uj_xi_3a
    ui_uj_1a = ui_1xj_1a * uj_xi_1a
    ui_uj_1b = ui_1xj_1b * uj_xi_1b
    ui_uj_3b = ui_3xj_3b * uj_xi_3b
    ui_uj_5b = ui_5xj_5b * uj_xi_5b
    ui_uj_7b = ui_7xj_7b * uj_xi_7b
    ui_uj_9b = ui_9xj_9b * uj_xi_9b

    # Divergence of tensor: Lands at canonical position of ui in volume x
    # coefficient computed in script
    (
        w10_1 * (ui_uj_1b - ui_uj_1a) +
        w10_3 * (ui_uj_3b - ui_uj_3a) / 3 +
        w10_5 * (ui_uj_5b - ui_uj_5a) / 5 +
        w10_7 * (ui_uj_7b - ui_uj_7a) / 7 +
        w10_9 * (ui_uj_9b - ui_uj_9a) / 9
    ) / dx(g)
end

@inline function diffusionterm(g::Grid, u, x, i, j)
    o = order(g)
    stencil = laplace_stencil(g) .|> eltype(u)
    diff = zero(eltype(u))
    @unroll for k = 1:2o-1
        diff += stencil[k] * u[x+(k-o)*e(g, j)|>g, i]
    end
    diff
end

"""
Compute convection-diffusion force from velocity `u`.
Add the force field to `f`.
"""
convectiondiffusion!

@kernel function convectiondiffusion!(grid, f, u, visc)
    T = eltype(u)
    dims = 1:dim(grid)
    x = @index(Global, Cartesian)
    @unroll for i in dims
        fxi = f[x, i]
        @unroll for j in dims
            fxi -= convterm(grid, u, x, i, j)
            fxi += visc * diffusionterm(grid, u, x, i, j)
        end
        f[x, i] = fxi
    end
end

@kernel function convection!(grid, f, u)
    T = eltype(u)
    dims = 1:dim(grid)
    x = @index(Global, Cartesian)
    @unroll for i in dims
        fxi = f[x, i]
        @unroll for j in dims
            fxi -= convterm(grid, u, x, i, j)
        end
        f[x, i] = fxi
    end
end

@kernel function diffusion!(grid, f, u, visc)
    T = eltype(u)
    dims = 1:dim(grid)
    x = @index(Global, Cartesian)
    @unroll for i in dims
        fxi = f[x, i]
        @unroll for j in dims
            fxi += visc * diffusionterm(grid, u, x, i, j)
        end
        f[x, i] = fxi
    end
end

laplace_stencil(g::Grid{2}) = (1, -2, 1) ./ dx(g)^2
laplace_stencil(g::Grid{4}) = (1, -54, 783, -1460, 783, -54, 1) .// 576 ./ dx(g)^2
laplace_stencil(g::Grid{6}) =
    (81, -2250, 56125, -603000, 5627250, -10156412, 5627250, -603000, 56125, -2250, 81) .//
    1920^2 ./ dx(g)^2
laplace_stencil(g::Grid{8}) =
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
laplace_stencil(g::Grid{10}) =
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
        a, b = s[1:n], s[n+1:end]
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
    d = dim(grid)

    # Since we use rfft, the first dimension is halved
    kmax = ntuple(i -> i == 1 ? div(n, 2) + 1 : n, d)

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

    # ahat = ntuple(d) do i
    #     k = 0:(kmax[i]-1)
    #     ahat = KernelAbstractions.allocate(backend, T, kmax[i])
    #     @. ahat = 4 / dx(grid)^2 * sinpi(k / n)^2
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
function project!(u, p, poissonsolve!, grid)
    # Divergence of tentative velocity field
    apply!(divergence!, grid, p, u)

    # Solve the Poisson equation
    poissonsolve!(p)

    # Apply pressure correction term
    apply!(pressuregradient!, grid, u, p)

    u
end

@kernel function vorticity!(g::Grid, ω, u)
    x = @index(Global, Cartesian)
    δu1_δx2 = δ(g, u, x, 1, 2)
    δu2_δx1 = δ(g, u, x, 2, 1)
    ω[x] = -δu1_δx2 + δu2_δx1
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
function get_scale_numbers(u, grid, visc)
    (; n) = grid
    d = dim(grid)
    T = eltype(u)
    uavg = sqrt(sum(abs2, u) / length(u))
    ϵ_field = scalarfield(grid)
    apply!(dissipation!, grid, ϵ_field, u, visc)
    ϵ = sum(ϵ_field) / length(ϵ_field)
    eta = (visc^3 / ϵ)^T(1 / 4)
    λ = sqrt(5 * visc / ϵ) * uavg
    L = let
        K = div(n, 2)
        uhat = fft(u, 1:d)
        uhat = uhat[ntuple(i -> 1:K, d)..., :]
        e = abs2.(uhat) ./ (2 * (n^d)^2)
        if d == 2
            kx = reshape(0:K-1, :)
            ky = reshape(0:K-1, 1, :)
            @. e = e / sqrt(kx^2 + ky^2)
        else
            kx = reshape(0:K-1, :)
            ky = reshape(0:K-1, 1, :)
            kz = reshape(0:K-1, 1, 1, :)
            @. e = e / sqrt(kx^2 + ky^2 + kz^2)
        end
        e = sum(e; dims = d + 1)
        # Remove k=(0,...,0) component
        # Note use of singleton range 1:1 instead of scalar index 1
        # (otherwise CUDA gets annoyed)
        e[1:1] .= 0
        T(3π) / 2 / uavg^2 * sum(e)
    end
    t_int = L / uavg
    t_tay = λ / uavg
    t_kol = visc / ϵ |> sqrt
    Re_int = L * uavg / visc
    Re_tay = λ * uavg / sqrt(T(3)) / visc
    Re_kol = eta * uavg / sqrt(T(3)) / visc
    (; uavg, ϵ, L, λ, eta, t_int, t_tay, t_kol, Re_int, Re_tay, Re_kol)
end
