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
@inline δ5(g::Grid, u, x, i, j) =
    g.n * (u[x+(2+(i!=j))*e(g, j)|>g, i] - u[x-(2+(i==j))*e(g, j)|>g, i]) / 5
@inline δ7(g::Grid, u, x, i, j) =
    g.n * (u[x+(3+(i!=j))*e(g, j)|>g, i] - u[x-(3+(i==j))*e(g, j)|>g, i]) / 7
@inline δ9(g::Grid, u, x, i, j) =
    g.n * (u[x+(4+(i!=j))*e(g, j)|>g, i] - u[x-(4+(i==j))*e(g, j)|>g, i]) / 9
@inline δ(g::Grid{2}, u, x, i, j) = δ1(g, u, x, i, j)
@inline δ(g::Grid{4}, u, x, i, j) = 9 // 8 * δ1(g, u, x, i, j) - 1 // 8 * δ3(g, u, x, i, j)
@inline δ(g::Grid{6}, u, x, i, j) =
    150 // 128 * δ1(g, u, x, i, j) - 25 // 128 * δ3(g, u, x, i, j) +
    3 // 128 * δ5(g, u, x, i, j)
@inline δ(g::Grid{8}, u, x, i, j) =
    1225 // 1024 * δ1(g, u, x, i, j) +
    -245 // 1024 * δ3(g, u, x, i, j) +
    49 // 1024 * δ5(g, u, x, i, j) +
    -5 // 1024 * δ7(g, u, x, i, j)
@inline δ(g::Grid{10}, u, x, i, j) =
    19845 // 16384 * δ1(g, u, x, i, j) +
    -2205 // 8192 * δ3(g, u, x, i, j) +
    567 // 8192 * δ5(g, u, x, i, j) +
    -405 // 32768 * δ7(g, u, x, i, j) +
    35 // 32768 * δ9(g, u, x, i, j)

# Scalar field gradient δp / δx[j].
@inline δ1(g::Grid, p, x, j) = g.n * (p[x+e(g, j)|>g] - p[x])
@inline δ3(g::Grid, p, x, j) = g.n * (p[x+2*e(g, j)|>g] - p[x-1e(g, j)|>g]) / 3
@inline δ5(g::Grid, p, x, j) = g.n * (p[x+3*e(g, j)|>g] - p[x-2e(g, j)|>g]) / 5
@inline δ7(g::Grid, p, x, j) = g.n * (p[x+4*e(g, j)|>g] - p[x-3e(g, j)|>g]) / 7
@inline δ9(g::Grid, p, x, j) = g.n * (p[x+5*e(g, j)|>g] - p[x-4e(g, j)|>g]) / 9
@inline δ(g::Grid{2}, p, x, j) = δ1(g, p, x, j)
@inline δ(g::Grid{4}, p, x, j) = 9 // 8 * δ1(g, p, x, j) - 1 // 8 * δ3(g, p, x, j)
@inline δ(g::Grid{6}, p, x, j) =
    150 // 128 * δ1(g, p, x, j) - 25 // 128 * δ3(g, p, x, j) + 3 // 128 * δ5(g, p, x, j)
@inline δ(g::Grid{8}, p, x, j) =
    1225 // 1024 * δ1(g, p, x, j) +
    -245 // 1024 * δ3(g, p, x, j) +
    49 // 1024 * δ5(g, p, x, j) +
    -5 // 1024 * δ7(g, p, x, j)
@inline δ(g::Grid{10}, p, x, j) =
    19845 // 16384 * δ1(g, p, x, j) +
    -2205 // 8192 * δ3(g, p, x, j) +
    567 // 8192 * δ5(g, p, x, j) +
    -405 // 32768 * δ7(g, p, x, j) +
    35 // 32768 * δ9(g, p, x, j)

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
    9 // 8 * pol1(g, u, x, i, j) - 1 // 8 * pol3(g, u, x, i, j)
@inline pol(g::Grid{6}, u, x, i, j) =
    150 // 128 * pol1(g, u, x, i, j) - 25 // 128 * pol3(g, u, x, i, j) +
    3 // 128 * pol5(g, u, x, i, j)
@inline pol(g::Grid{8}, u, x, i, j) =
    1225 // 1024 * pol1(g, u, x, i, j) +
    -245 // 1024 * pol3(g, u, x, i, j) +
    49 // 1024 * pol5(g, u, x, i, j) +
    -5 // 1024 * pol7(g, u, x, i, j)
@inline pol(g::Grid{10}, u, x, i, j) =
    19845 // 16384 * pol1(g, u, x, i, j) +
    -2205 // 8192 * pol3(g, u, x, i, j) +
    567 // 8192 * pol5(g, u, x, i, j) +
    -405 // 32768 * pol7(g, u, x, i, j) +
    35 // 32768 * pol9(g, u, x, i, j)

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
    g.n * (9 // 8 * (ui_uj_b - ui_uj_a) - 1 // 8 * (ui_uj_bb - ui_uj_aa) / 3)
end

@inline function convterm(g::Grid{6}, u, x, i, j)
    ei, ej = e(g, i), e(g, j)

    # (a)(a)a: (thrice) (twice) left in xj
    # (b)(b)b: (thrice) (twice) right in xj
    # 1: second order
    # 3: fourth order
    # 5: sixth order

    # ui interpolated in direction xj
    ui_xj_1_a = pol1(g, u, x - (i != j) * ej |> g, i, j)
    ui_xj_1_b = pol1(g, u, x + (i == j) * ej |> g, i, j)
    ui_xj_3_aa = pol3(g, u, x - (1 + (i != j)) * ej |> g, i, j)
    ui_xj_3_bb = pol3(g, u, x + (1 + (i == j)) * ej |> g, i, j)
    ui_xj_5_aaa = pol5(g, u, x - (2 + (i != j)) * ej |> g, i, j)
    ui_xj_5_bbb = pol5(g, u, x + (2 + (i == j)) * ej |> g, i, j)

    # uj interpolated in direction xi
    uj_xi_aaa = pol(g, u, x - (2 + (i != j)) * ej |> g, j, i)
    uj_xi_aa = pol(g, u, x - (1 + (i != j)) * ej |> g, j, i)
    uj_xi_a = pol(g, u, x - (i != j) * ej |> g, j, i)
    uj_xi_b = pol(g, u, x + (i == j) * ej |> g, j, i)
    uj_xi_bb = pol(g, u, x + (1 + (i == j)) * ej |> g, j, i)
    uj_xi_bbb = pol(g, u, x + (2 + (i == j)) * ej |> g, j, i)

    # Tensor product -- see  Morinishi 1998 eq. (112)
    ui_uj_aaa = ui_xj_5_aaa * uj_xi_aaa
    ui_uj_aa = ui_xj_3_aa * uj_xi_aa
    ui_uj_a = ui_xj_1_a * uj_xi_a
    ui_uj_b = ui_xj_1_b * uj_xi_b
    ui_uj_bb = ui_xj_3_bb * uj_xi_bb
    ui_uj_bbb = ui_xj_5_bbb * uj_xi_bbb

    # Divergence of tensor: Lands at canonical position of ui in volume x
    # see  Morinishi 1998 eq. (112)
    g.n * (
        150 // 128 * (ui_uj_b - ui_uj_a) - 25 // 128 * (ui_uj_bb - ui_uj_aa) / 3 +
        3 // 128 * (ui_uj_bbb - ui_uj_aaa) / 5
    )
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
    ui_xj_7_7a = pol7(g, u, x - (3 + (i != j)) * ej |> g, i, j)
    ui_xj_5_5a = pol5(g, u, x - (2 + (i != j)) * ej |> g, i, j)
    ui_xj_3_3a = pol3(g, u, x - (1 + (i != j)) * ej |> g, i, j)
    ui_xj_1_1a = pol1(g, u, x - (i != j) * ej |> g, i, j)
    ui_xj_1_1b = pol1(g, u, x + (i == j) * ej |> g, i, j)
    ui_xj_3_3b = pol3(g, u, x + (1 + (i == j)) * ej |> g, i, j)
    ui_xj_5_5b = pol5(g, u, x + (2 + (i == j)) * ej |> g, i, j)
    ui_xj_7_7b = pol7(g, u, x + (3 + (i == j)) * ej |> g, i, j)

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
    ui_uj_7a = ui_xj_7_7a * uj_xi_7a
    ui_uj_5a = ui_xj_5_5a * uj_xi_5a
    ui_uj_3a = ui_xj_3_3a * uj_xi_3a
    ui_uj_1a = ui_xj_1_1a * uj_xi_1a
    ui_uj_1b = ui_xj_1_1b * uj_xi_1b
    ui_uj_3b = ui_xj_3_3b * uj_xi_3b
    ui_uj_5b = ui_xj_5_5b * uj_xi_5b
    ui_uj_7b = ui_xj_7_7b * uj_xi_7b

    # Divergence of tensor: Lands at canonical position of ui in volume x
    # coefficient computed in script
    g.n * (
        1225 // 1024 * (ui_uj_1b - ui_uj_1a) +
        -245 // 1024 * (ui_uj_3b - ui_uj_3a) / 3 +
        49 // 1024 * (ui_uj_5b - ui_uj_5a) / 5 +
        -5 // 1024 * (ui_uj_7b - ui_uj_7a) / 7
    )
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

    # ui interpolated in direction xj
    ui_xj_9_9a = pol9(g, u, x - (4 + (i != j)) * ej |> g, i, j)
    ui_xj_7_7a = pol7(g, u, x - (3 + (i != j)) * ej |> g, i, j)
    ui_xj_5_5a = pol5(g, u, x - (2 + (i != j)) * ej |> g, i, j)
    ui_xj_3_3a = pol3(g, u, x - (1 + (i != j)) * ej |> g, i, j)
    ui_xj_1_1a = pol1(g, u, x - (i != j) * ej |> g, i, j)
    ui_xj_1_1b = pol1(g, u, x + (i == j) * ej |> g, i, j)
    ui_xj_3_3b = pol3(g, u, x + (1 + (i == j)) * ej |> g, i, j)
    ui_xj_5_5b = pol5(g, u, x + (2 + (i == j)) * ej |> g, i, j)
    ui_xj_7_7b = pol7(g, u, x + (3 + (i == j)) * ej |> g, i, j)
    ui_xj_9_9b = pol9(g, u, x + (4 + (i == j)) * ej |> g, i, j)

    # uj interpolated in direction xi
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
    ui_uj_9a = ui_xj_9_9a * uj_xi_9a
    ui_uj_7a = ui_xj_7_7a * uj_xi_7a
    ui_uj_5a = ui_xj_5_5a * uj_xi_5a
    ui_uj_3a = ui_xj_3_3a * uj_xi_3a
    ui_uj_1a = ui_xj_1_1a * uj_xi_1a
    ui_uj_1b = ui_xj_1_1b * uj_xi_1b
    ui_uj_3b = ui_xj_3_3b * uj_xi_3b
    ui_uj_5b = ui_xj_5_5b * uj_xi_5b
    ui_uj_7b = ui_xj_7_7b * uj_xi_7b
    ui_uj_9b = ui_xj_9_9b * uj_xi_9b

    # Divergence of tensor: Lands at canonical position of ui in volume x
    # coefficient computed in script
    g.n * (
        19845 // 16384 * (ui_uj_1b - ui_uj_1a) +
        -2205 // 8192 * (ui_uj_3b - ui_uj_3a) / 3 +
        567 // 8192 * (ui_uj_5b - ui_uj_5a) / 5 +
        -405 // 32768 * (ui_uj_7b - ui_uj_7a) / 7 +
        35 // 32768 * (ui_uj_9b - ui_uj_9a) / 9
    )
end

@inline diffusionterm(g::Grid{2}, u, x, i, j) =
    g.n^2 * (u[x-e(g, j)|>g, i] - 2 * u[x, i] + u[x+e(g, j)|>g, i])

@inline function diffusionterm(g::Grid{4}, u, x, i, j)
    stencil = (1, -54, 783, -1460, 783, -54, 1) ./ 576 .* g.n^2 .|> eltype(u)
    diff = zero(eltype(u))
    @unroll for k = 1:7
        diff += stencil[k] * u[x+(k-4)*e(g, j)|>g, i]
    end
    diff
end

@inline function diffusionterm(g::Grid{6}, u, x, i, j)
    stencil =
        (
            81,
            -2250,
            56125,
            -603000,
            5627250,
            -10156412,
            5627250,
            -603000,
            56125,
            -2250,
            81,
        ) ./ 1920^2 .* g.n^2 .|> eltype(u)
    diff = zero(eltype(u))
    @unroll for k = 1:length(stencil)
        diff += stencil[k] * u[x+(k-6)*e(g, j)|>g, i]
    end
    diff
end

@inline function diffusionterm(g::Grid{8}, u, x, i, j)
    stencil =
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
        ) .* g.n^2 .|> eltype(u)
    diff = zero(eltype(u))
    @unroll for k = 1:length(stencil)
        diff += stencil[k] * u[x+(k-8)*e(g, j)|>g, i]
    end
    diff
end

@inline function diffusionterm(g::Grid{10}, u, x, i, j)
    stencil =
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
        ) .* g.n^2 .|> eltype(u)
    diff = zero(eltype(u))
    @unroll for k = 1:length(stencil)
        diff += stencil[k] * u[x+(k-10)*e(g, j)|>g, i]
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
laplace_stencil(g::Grid{6}) =
    [81, -2250, 56125, -603000, 5627250, -10156412, 5627250, -603000, 56125, -2250, 81] /
    1920^2 * g.n^2
laplace_stencil(g::Grid{8}) =
    [
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
    ] * g.n^2
laplace_stencil(g::Grid{10}) =
    [
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
    ] * g.n^2

mergestencil(s, n) =
    if length(s) > n
        a, b = s[1:n], s[n+1:end]
        b = mergestencil(b, n)
        a[1:length(b)] .+= b
        a
    else
        s
    end

"Create spectral Poisson solver from setup."
function poissonsolver(setup)
    (; backend, grid, visc) = setup
    T = typeof(visc)
    d = dim(grid)
    n = grid.n

    # Since we use rfft, the first dimension is halved
    kmax = ntuple(i -> i == 1 ? div(n, 2) + 1 : n, d)

    # Discrete Laplacian stencil -- pad with zeros to full size
    a_cpu = laplace_stencil(grid) .|> T
    if length(a_cpu) > n
        # This is only necessary for convergence plot
        @warn "Laplacian stencil is longer than grid size. Merging."
        a_cpu = mergestencil(a_cpu, n)
    else
        a_cpu = vcat(a_cpu, zeros(T, n - length(a_cpu)))
    end
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

"""
Get the following dimensional scale numbers [Pope2000](@cite):

- Velocity ``u_\\text{avg} = \\langle u_i u_i \\rangle^{1/2}``
- Dissipation rate ``\\epsilon = 2 \\nu \\langle S_{ij} S_{ij} \\rangle``
- Kolmolgorov length scale ``\\eta = (\\frac{\\nu^3}{\\epsilon})^{1/4}``
- Taylor length scale ``\\lambda = (\\frac{5 \\nu}{\\epsilon})^{1/2} u_\\text{avg}``
- Taylor-scale Reynolds number ``Re_\\lambda = \\frac{\\lambda u_\\text{avg}}{\\sqrt{3} \\nu}``
- Integral length scale ``L = \\frac{3 \\pi}{2 u_\\text{avg}^2} \\int_0^\\infty \\frac{E(k)}{k} \\, \\mathrm{d} k``
- Large-eddy turnover time ``\\tau = \\frac{L}{u_\\text{avg}}``
"""
function get_scale_numbers(u, setup)
    (; grid, visc) = setup
    (; n) = grid
    d = dim(grid)
    T = eltype(u)
    uavg = sqrt(sum(abs2, u) / length(u))
    ϵ_field = scalarfield(setup)
    apply!(dissipation!, setup, ϵ_field, u, setup.visc)
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
    t_kol = eta / uavg
    Re_int = L * uavg / visc
    Re_tay = λ * uavg / sqrt(T(3)) / visc
    Re_kol = eta * uavg / sqrt(T(3)) / visc
    (; uavg, ϵ, L, λ, eta, t_int, t_tay, t_kol, Re_int, Re_tay, Re_kol)
end
