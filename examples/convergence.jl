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
using KernelAbstractions
using JLD2

backend = KernelAbstractions.CPU()
# backend = CUDABackend()

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

function bench(; order, n, tmax)
    @info order, n
    grid = Turbulox.Grid(; order, dim = 2, n)
    setup = Turbulox.problem_setup(; grid, visc, backend)
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
    let
        # Warm-up
        Δt = T(0.9) * Turbulox.propose_timestep(u, setup)
        Turbulox.timestep!(
            Turbulox.default_right_hand_side!,
            copy(u),
            cache,
            Δt,
            solver!,
            setup,
        )
    end
    timing = time()
    t = 0f
    while t < tmax
        Δt = T(0.1) * Turbulox.propose_timestep(u, setup)
        Δt = min(Δt, tmax - t)
        Turbulox.timestep!(Turbulox.default_right_hand_side!, u, cache, Δt, solver!, setup)
        t += Δt
        @show t
    end
    timing = time() - timing
    uref = Turbulox.vectorfield(setup)
    uref[:, :, 1] .= sol.(1, x, xp', t)
    uref[:, :, 2] .= sol.(2, xp, x', t)
    err = norm(u .- uref) / norm(uref)
    (; timing, err, order, n)
end

tmax = T(0.5)
b2 = map(n -> bench(; order = 2, n, tmax), [16, 32, 64, 128, 256, 512, 1024])
b4 = map(n -> bench(; order = 4, n, tmax), [16, 32, 64, 128, 256, 512])
b6 = map(n -> bench(; order = 6, n, tmax), [16, 32, 64, 128, 256])

jldsave("output/convergence_backend=$(backend)_threads=$(Threads.nthreads()).jld2"; b2, b4, b6)

using CairoMakie

CairoMakie.activate!()
WGLMakie.activate!()

# Plot timing vs error
fig = let
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        xlabel = "Time",
        ylabel = "Relative error",
        xscale = log10,
        yscale = log10,
    )
    scatterlines!(ax, map(b -> Point2f(b.timing, b.err), b2), label = "Order 2")
    scatterlines!(ax, map(b -> Point2f(b.timing, b.err), b4), label = "Order 4")
    scatterlines!(ax, map(b -> Point2f(b.timing, b.err), b6), label = "Order 6")
    axislegend(ax)
    save("output/timing_vs_error_backend=$(backend)_threads=$(Threads.nthreads()).pdf", fig)
    fig
end

# Plot grid size vs error
fig = let
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        xlabel = "n",
        ylabel = "Relative error",
        xscale = log10,
        yscale = log10,
        xticks = map(b -> b.n, b2),
    )
    scatterlines!(ax, map(b -> Point2f(b.n, b.err), b2); label = "Order 2")
    scatterlines!(ax, map(b -> Point2f(b.n, b.err), b4); label = "Order 4")
    scatterlines!(ax, map(b -> Point2f(b.n, b.err), b6); label = "Order 6")
    ylims!(ax, (3e-16, 1e-3))

    # Lower triangle
    nref = b6[3].n
    eref = b6[3].err / 5
    x = nref / sqrt(2), nref * sqrt(2)
    y = eref / sqrt(2)^6, eref * sqrt(2)^6
    a = Point2f(x[1], y[2])
    b = Point2f(x[2], y[1])
    c = Point2f(x[1], y[1])
    lines!(ax, [a, b, c, a]; color = :black)
    text!(ax, "6"; position = Point2f(0.9 * x[1], 0.6 * eref))
    text!(ax, "1"; position = Point2f(0.95 * nref, 0.2 * y[1]))

    # Upper triangle
    nref = b2[4].n
    eref = b2[4].err * 5
    x = nref / sqrt(2), nref * sqrt(2)
    y = eref / sqrt(2)^2, eref * sqrt(2)^2
    a = Point2f(x[1], y[2])
    b = Point2f(x[2], y[1])
    c = Point2f(x[2], y[2])
    lines!(ax, [a, b, c, a]; color = :black)
    text!(ax, "2"; position = Point2f(x[2] / 0.9, eref * 0.6))
    text!(ax, "1"; position = Point2f(nref / 0.95, y[1] / 0.2))

    axislegend(ax; position = :lb)
    save("output/convergence_backend=$(backend)_threads=$(Threads.nthreads()).pdf", fig)
    fig
end
