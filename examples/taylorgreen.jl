if false
    include("../src/Turbulox.jl")
    using .Turbulox
end

using CUDA
using JLD2
using KernelAbstractions
using LinearAlgebra
using Random
using Turbulox
using GLMakie
using WriteVTK

set_theme!(theme_dark())

# T = Float32
T = Float64
f = T(1)

# backend = KernelAbstractions.CPU()
backend = CUDABackend()

visc = 1 / 2_000 |> T

# Initial conditions
Ux(x, y, z) = sinpi(2x) * cospi(2y) * sinpi(2z) / 2
Uy(x, y, z) = -cospi(2x) * sinpi(2y) * sinpi(2z) / 2
Uz(x, y, z) = zero(x)

grid = Turbulox.RGrid(; order = 2, dim = 3, n = 128, backend);
solver! = Turbulox.poissonsolver(grid);

u = Turbulox.vectorfield(grid);
cache = (;
    ustart = Turbulox.vectorfield(grid),
    du = Turbulox.vectorfield(grid),
    p = Turbulox.scalarfield(grid),
);

# Initialize
x = range(0f, 1f, grid.n + 1)[2:end];
Δx = x[2] - x[1];
x = x .- Δx / 2;
y = reshape(x, 1, :);
z = reshape(x, 1, 1, :);
Δx = x[2] - x[1];
xp = x
# xp = x .- Δx / 2;
yp = reshape(xp, 1, :);
zp = reshape(xp, 1, 1, :);
u[:, :, :, 1] .= Ux.(x, yp, zp);
u[:, :, :, 2] .= Uy.(xp, y, zp);
u[:, :, :, 3] .= Uz.(xp, yp, z);
Turbulox.project!(u, cache.p, solver!, grid);

"Compute ``z``-component of vorticity in the plane ``z=z``."
@kernel function vort_z(grid, ω, u, z)
    x, y = @index(Global, NTuple)
    X = CartesianIndex((x, y, z))
    δux_δy = Turbulox.δ(grid, u, X, 1, 2)
    δuy_δx = Turbulox.δ(grid, u, X, 2, 1)
    ω[x, y] = -δux_δy + δuy_δx
end

fig, ω_obs, ω = let
    ω_obs = Observable(Array(u[:, :, 1, 1]))
    colorrange = lift(ω_obs) do ω
        r = maximum(abs, ω)
        -r, r
    end
    # fig, ax, hm = image(
    fig, ax, hm = heatmap(
        ω_obs;
        # colorrange = (-7, 7),
        colormap = :seaborn_icefire_gradient,
        axis = (
            # title = "Vorticity ω_z",
            # xlabel = "x",
            # ylabel = "y",
            xticksvisible = false,
            xticklabelsvisible = false,
            yticksvisible = false,
            yticklabelsvisible = false,
            aspect = DataAspect(),
        ),
        figure = (; size = (600, 600)),
    )
    Colorbar(fig[1, 2], hm)
    fig, ω_obs, ω
end
fig

fig, ω_obs, ω = let
    ω = KernelAbstractions.zeros(backend, T, grid.n, grid.n)
    Turbulox.apply!(vort_z, grid, ω, u, grid.n ÷ 4)
    ω_obs = Observable(Array(ω))
    colorrange = lift(ω_obs) do ω
        r = maximum(abs, ω)
        -r, r
    end
    # fig, ax, hm = image(
    fig, ax, hm = heatmap(
        ω_obs;
        # colorrange = (-7, 7),
        colormap = :seaborn_icefire_gradient,
        axis = (
            # title = "Vorticity ω_z",
            # xlabel = "x",
            # ylabel = "y",
            xticksvisible = false,
            xticklabelsvisible = false,
            yticksvisible = false,
            yticklabelsvisible = false,
            aspect = DataAspect(),
        ),
        figure = (; size = (600, 600)),
    )
    Colorbar(fig[1, 2], hm)
    fig, ω_obs, ω
end
fig

v = copy(u);
copyto!(u, v);

# usmag = copy(u)

# stream = VideoStream(fig; visible = true);

closure! = Turbulox.clark_model(grid, T(1 / grid.n));
closure! = Turbulox.eddyviscosity_model(
    Turbulox.smagorinsky_viscosity!,
    grid,
    T(0.17),
    T(1 / grid.n),
);
closure! =
    Turbulox.eddyviscosity_model(Turbulox.vreman_viscosity!, grid, T(0.28), T(1 / grid.n));
closure! = Turbulox.eddyviscosity_model(
    Turbulox.verstappen_viscosity!,
    grid,
    T(0.345),
    T(1 / grid.n),
);
closure! =
    Turbulox.eddyviscosity_model(Turbulox.nicoud_viscosity!, grid, T(1.35), T(1 / grid.n));

du = zero(u);
closure!(du, u);

function rhs!(du, u, grid)
    fill!(du, 0)
    Turbulox.apply!(Turbulox.convectiondiffusion!, grid, du, u, visc)
    # closure!(du, u)
end

CUDA.@profile rhs!(du, u, grid)
@profview (rhs!(du, u, grid); KernelAbstractions.synchronize(backend))

let
    copyto!(u, v)
    t = 0f
    i = 0
    tmax = 20f
    n0 = norm(u)
    while t < tmax
        i += 1
        Δt = T(0.85) * Turbulox.propose_timestep(u, grid)
        Δt = min(Δt, tmax - t)
        # Δt = T(0.01)
        Turbulox.timestep!(rhs!, u, cache, Δt, solver!, grid)
        t += Δt
        @show t
        if i % 1 == 0
            nn = norm(u)
            nn > 2 * n0 && break
            Turbulox.apply!(vort_z, grid, ω, u, grid.n ÷ 4)
            # ω_obs[] = copyto!(ω_obs[], ω)
            ω_obs[] = copyto!(ω_obs[], u[:, :, 1, 1])
            sleep(0.01)
            # recordframe!(stream)
        end
    end
end

save("taylorgreen_order=$(Turbulox.order(grid))_n=$(grid.n).mp4", stream)

s = Turbulox.get_scale_numbers(u, grid, visc)

let
    stuff = Turbulox.spectral_stuff(grid)
    spec = Turbulox.spectrum(usmag, grid; stuff)
    fig = Figure()
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10)
    lines!(ax, Point2f.(spec.κ, spec.s))
    xslope = spec.κ[10:end]
    yslope = @. 1.58 * s.ϵ^(2 / 3) * (π * xslope)^(-5 / 3)
    lines!(ax, xslope, yslope; color = Cycled(2))
    vlines!(ax, 1 / s.L / 1)
    vlines!(ax, 1 / s.λ / 1)
    display(GLMakie.Screen(), fig)
end

ubar = Turbulox.vectorfield(grid);
∇u = Turbulox.collocated_tensorfield(grid);
q = Turbulox.scalarfield(grid);
r = Turbulox.scalarfield(grid);
Turbulox.gaussian!(ubar, u, T(1 / 50), grid);
Turbulox.apply!(Turbulox.velocitygradient_coll!, grid, ∇u, u);
# Turbulox.apply!(Turbulox.velocitygradient!, grid, ∇u, ubar);
Turbulox.apply!(Turbulox.compute_qr!, grid, q, r, ∇u);

q
r

let
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        xticksvisible = false,
        xticklabelsvisible = false,
        yticksvisible = false,
        yticklabelsvisible = false,
        aspect = DataAspect(),
    )
    slider = SliderGrid(
        fig[1, 2],
        (; label = "Coordinate", range = 1:grid.n),
        (; label = "Dimension", range = 1:3);
        tellheight = false,
        width = 250,
    )
    qq = q[:, :, 1] |> Array
    Q = map(getfield.(slider.sliders, :value)...) do i, d
        copyto!(qq, selectdim(q, d, i))
        qq
    end
    heatmap!(
        ax,
        Q;
        colorrange = (0.10, 5),
        # lowclip = :transparent,
    )
    fig
end

volume(q |> Array; clip_planes = [Plane3f(Vec3f(0, 1, 1), 50)])

contour(
    q |> Array;
    colormap = :inferno,
    figure = (; size = (800, 800)),
    colorrange = (0.1, 5),
    alpha = 0.5,
    levels = range(0.1, 5, 10),
    isorange = 0.4,
    # lowclip = :transparent,
)

volume(
    q |> Array;
    colormap = :inferno,
    # colormap = Reverse(:inferno),
    figure = (; size = (800, 800)),
    algorithm = :iso, # IsoValue
    # algorithm = :absorption, # Absorption
    # algorithm = :mip, # MaximumIntensityProjection
    # algorithm = :absorptionrgba, # AbsorptionRGBA
    # algorithm = :additive, # AdditiveRGBA
    # algorithm = :indexedabsorption, # IndexedAbsorptionRGBA
    isorange = 1.0,
    isovalue = 10,
    # absorption=50f0,
    colorrange = (0.1, 5),
    # colorrange = (-0.1, -5),
    # lowclip = :transparent,
)

Q[] = copyto!(Q[], view(q, :, 220, :))

s = Turbulox.get_scale_numbers(u, grid, visc)

let
    rstar = r |> Array |> vec
    qstar = q |> Array |> vec
    # rstar = s.t_kol^3 * rstar
    # qstar = s.t_kol^2 * qstar
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "R", ylabel = "Q", limits = (-7, 7, -7, 10))
    hb = hexbin!(
        ax,
        rstar,
        qstar;
        bins = 1000,
        # colormap = :plasma,
        colormap = Reverse(:Spectral),
        colorscale = log10,
    )
    # scatter!(r |> Array |> vec, q |> Array |> vec; alpha = 0.05)
    Colorbar(fig[1, 2], hb)
    save("output/qr.png", fig)
    # save("output/qrbar.png", fig)
    fig
end

vtk_grid("output/fields", xp, xp, xp) do vtk
    vtk["q"] = q |> Array
    vtk["r"] = r |> Array
    vtk["u"] = ((eachslice(u; dims = 4) .|> Array)...,)
end

u[1, :, :, 3] |> Array |> heatmap

# save_object("output/u.jld2", u |> Array)
# copyto!(u, load_object("output/u.jld2"));

s.uavg
s.L
s.λ
s.eta
s.ϵ
s.Re_int
s.Re_tay
s.Re_kol
s.t_int
s.t_tay
s.t_kol
