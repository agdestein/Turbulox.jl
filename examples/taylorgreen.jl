if false
    include("src/Turbulox.jl")
    using .Turbulox
end

using Turbulox
using Random
using CUDA
using LinearAlgebra
using WGLMakie
using CairoMakie

# T = Float32
T = Float64
f = T(1)

visc = 1 / 2000 |> T

# Analytical solution for 2D Taylor-Green vortex
sol = let
    a = 8 * π^2 * visc |> T
    (dim, x, y, t) ->
        (dim == 1 ? -sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y)) * exp(-a * t)
end

function rmse(; order, n, tmax)
    @info n
    grid = Turbulox.Grid(; order, dim = 2, n)
    setup = Turbulox.problem_setup(; grid, visc, backend = CUDABackend())
    solver! = Turbulox.poissonsolver(setup)
    u = Turbulox.vectorfield(setup)
    x = range(0f, 1f, grid.n + 1)[2:end]
    Δx = x[2] - x[1]
    xp = x .- Δx / 2
    x = range(0f, 1f, grid.n + 1)[2:end]
    u[:, :, 1] .= sol.(1, x, xp', 0f)
    u[:, :, 2] .= sol.(2, xp, x', 0f)
    # Turbulox.project!(u, Turbulox.scalarfield(setup), solver!, setup)
    cache = (;
        ustart = Turbulox.vectorfield(setup),
        du = Turbulox.vectorfield(setup),
        p = Turbulox.scalarfield(setup),
    )
    t = 0f
    while t < tmax
        Δt = T(0.4) * Turbulox.propose_timestep(u, setup)
        Δt = min(Δt, tmax - t)
        Turbulox.timestep!(Turbulox.default_right_hand_side!, u, cache, Δt, solver!, setup)
        t += Δt
        # @show t
    end
    uref = Turbulox.vectorfield(setup)
    uref[:, :, 1] .= sol.(1, x, xp', t)
    uref[:, :, 2] .= sol.(2, xp, x', t)
    norm(u .- uref) / norm(uref)
end

tmax = 2f
nn = [16, 32, 64, 128, 256, 512]
e2 = map(n -> rmse(; order = 2, n, tmax), nn)
e4 = map(n -> rmse(; order = 4, n, tmax), nn)
e6 = map(n -> rmse(; order = 6, n, tmax), nn)

using CairoMakie

CairoMakie.activate!()
WGLMakie.activate!()

fig = let
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        xlabel = "n",
        ylabel = "Relative error",
        xscale = log10,
        yscale = log10,
        xticks = nn,
    )
    scatterlines!(ax, nn, e2; label = "2nd order scheme")
    scatterlines!(ax, nn, e4; label = "4th order scheme")
    scatterlines!(ax, nn, e6; label = "6th order scheme")
    # lines!(ax, nn, 1e-1 ./ nn .^ 2)#; linestyle = :dash)
    # lines!(ax, nn, 6e-1 ./ nn .^ 4)#; linestyle = :dash)
    # lines!(ax, nn, 2e0 ./ nn .^ 6)#; linestyle = :dash)
    ylims!(ax, (1e-15, 1e-2))
    # Lower triangle
    nref = nn[3]
    eref = e6[3] / 5
    x = nref / sqrt(2), nref * sqrt(2)
    y = eref / sqrt(2)^6, eref * sqrt(2)^6
    a = Point2f(x[1], y[2])
    b = Point2f(x[2], y[1])
    c = Point2f(x[1], y[1])
    lines!(ax, [a, b, c, a]; color = :black)
    text!(ax, "6"; position = Point2f(0.9 * x[1], 0.6 * eref))
    text!(ax, "1"; position = Point2f(0.95 * nref, 0.2 * y[1]))
    # Upper triangle
    nref = nn[4]
    eref = e2[4] * 5
    x = nref / sqrt(2), nref * sqrt(2)
    y = eref / sqrt(2)^2, eref * sqrt(2)^2
    a = Point2f(x[1], y[2])
    b = Point2f(x[2], y[1])
    c = Point2f(x[2], y[2])
    lines!(ax, [a, b, c, a]; color = :black)
    text!(ax, "2"; position = Point2f(x[2] / 0.9, eref * 0.6))
    text!(ax, "1"; position = Point2f(nref / 0.95, y[1] / 0.2))
    axislegend(ax; position = :lb)
    fig
end

save("convergence_$T.png", fig)
