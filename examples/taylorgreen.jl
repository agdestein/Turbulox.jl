if false
    include("src/Turbulox.jl")
    using .Turbulox
end

using Turbulox
using Random
using CUDA
using LinearAlgebra
using GLMakie
using CairoMakie

visc = 1 / 2000

# Analytical solution for 2D Taylor-Green vortex
sol = let
    a = 8 * π^2 * visc
    (dim, x, y, t) ->
        (dim == 1 ? -sinpi(2x) * cospi(2y) : cospi(2x) * sinpi(2y)) * exp(-a * t)
end

function rmse(; order, n, tmax)
    grid = Turbulox.Grid(; order, dim = 2, n)
    setup = Turbulox.problem_setup(; grid, visc, backend = CUDABackend())
    solver! = Turbulox.poissonsolver(setup)
    u = Turbulox.vectorfield(setup)
    x = range(0.0, 1.0, grid.n + 1)[2:end]
    Δx = x[2] - x[1]
    xp = x .- Δx / 2
    x = range(0.0, 1.0, grid.n + 1)[2:end]
    u[:, :, 1] .= sol.(1, x, xp', 0.0)
    u[:, :, 2] .= sol.(2, xp, x', 0.0)
    # Turbulox.project!(u, Turbulox.scalarfield(setup), solver!, setup)
    cache = (;
        ustart = Turbulox.vectorfield(setup),
        du = Turbulox.vectorfield(setup),
        p = Turbulox.scalarfield(setup),
    )
    t = 0.0
    while t < tmax
        Δt = 0.4 * Turbulox.propose_timestep(u, setup)
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

nn = [8, 16, 32, 64, 128, 256, 512, 1024]
e2 = map(n -> rmse(; order = 2, n, tmax = 2.0), nn)
e4 = map(n -> rmse(; order = 4, n, tmax = 2.0), nn)

fig = let
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        xlabel = "n",
        ylabel = "RMSE",
        xscale = log10,
        yscale = log10,
        xticks = nn,
    )
    scatterlines!(ax, nn, e2; label = "2nd order scheme")
    scatterlines!(ax, nn, e4; label = "4nd order scheme")
    lines!(ax, nn, 1e-1 ./ nn .^ 2; linestyle = :dash, label = "2nd order ref")
    lines!(ax, nn, 1e-1 ./ nn .^ 4; linestyle = :dash, label = "4nd order ref")
    axislegend(ax)
    fig
end

save("convergence.pdf", fig)
