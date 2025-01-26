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
b2 = map(n -> bench(; order = 2, n, tmax), [4, 8, 16, 32, 64, 128, 256, 512, 1024])
b4 = map(n -> bench(; order = 4, n, tmax), [4, 8, 16, 32, 64, 128, 256, 512])
b6 = map(n -> bench(; order = 6, n, tmax), [4, 8, 16, 32, 64, 128, 256])
b8 = map(n -> bench(; order = 8, n, tmax), [4, 8, 16, 32, 64, 128])
b10 = map(n -> bench(; order = 10, n, tmax), [4, 8, 16, 32, 64])

# jldsave(
#     "output/convergence_backend=$(backend)_threads=$(Threads.nthreads()).jld2";
#     b2,
#     b4,
#     b6,
#     b8,
#     b10,
# )
# b2, b4, b6, b8, b10 = load(
#     "output/convergence_backend=$(backend)_threads=$(Threads.nthreads()).jld2",
#     "b2",
#     "b4",
#     "b6",
#     "b8",
#     "b10",
# )

using CairoMakie

CairoMakie.activate!()
GLMakie.activate!()

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
    for (b, marker, order) in [
        (b2, :circle, 2),
        (b4, :utriangle, 4),
        (b6, :rect, 6),
        (b8, :diamond, 8),
        (b10, :cross, 10),
    ]
        scatterlines!(ax, map(b -> Point2f(b.timing, b.err), b); marker, label = "Order $order")
    end
    axislegend(ax; position = :lb)
    save("output/timing_vs_error_backend=$(backend)_threads=$(Threads.nthreads()).pdf", fig)
    fig
end

# Lower triangle
function lowertriangle!(ax, xref, yref, order; width = 2)
    x = xref / sqrt(width), xref * sqrt(width)
    y = yref / sqrt(width)^order, yref * sqrt(width)^order
    a = Point2f(x[1], y[2])
    b = Point2f(x[2], y[1])
    c = Point2f(x[1], y[1])
    lines!(ax, [a, b, c, a]; color = :black)
    text!(ax, "$order"; position = Point2f(0.8 * x[1], 0.6 * yref))
    text!(ax, "1"; position = Point2f(0.95 * xref, 0.2 * y[1]))
end

# Upper triangle
function uppertriangle!(ax, xref, yref, order; width = 2)
    x = xref / sqrt(width), xref * sqrt(width)
    y = yref / sqrt(width)^order, yref * sqrt(width)^order
    a = Point2f(x[1], y[2])
    b = Point2f(x[2], y[1])
    c = Point2f(x[2], y[2])
    lines!(ax, [a, b, c, a]; color = :black)
    text!(ax, "$order"; position = Point2f(x[2] / 0.9, yref * 0.6))
    text!(ax, "1"; position = Point2f(xref / 0.95, y[1] * 20))
end

WGLMakie.activate!()

# Plot grid size vs error
fig = let
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        xlabel = "Resolution",
        ylabel = "Relative error",
        xscale = log10,
        yscale = log10,
        xticks = map(b -> b.n, b2),
    )
    for (b, marker, order) in [
        (b2, :circle, 2),
        (b4, :utriangle, 4),
        (b6, :rect, 6),
        (b8, :diamond, 8),
        (b10, :cross, 10),
    ]
        scatterlines!(ax, map(b -> Point2f(b.n, b.err), b); marker, label = "Order $order")
    end
    ylims!(ax, (3e-16, 2e-2))
    uppertriangle!(ax, b2[5].n / sqrt(2), 2 * b2[5].err * 5, 2; width = 4)
    lowertriangle!(ax, b10[2].n, b10[2].err / 10, 10)
    axislegend(ax; position = :lb)
    save("output/convergence_backend=$(backend)_threads=$(Threads.nthreads()).pdf", fig)
    fig
end
