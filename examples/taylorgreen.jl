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
using WGLMakie
using WriteVTK

set_theme!(theme_dark())

T = Float32
# T = Float64
f = T(1)

backend = CUDABackend()

visc = 1 / 50_000 |> T

# Initial conditions
Ux(x, y, z) = sinpi(1x) * cospi(1y) * sinpi(1z) / 2
Uy(x, y, z) = -cospi(1x) * sinpi(1y) * sinpi(1z) / 2
Uz(x, y, z) = zero(x)

grid = Turbulox.Grid(; order = 4, dim = 3, n = 256);
setup = Turbulox.problem_setup(; grid, visc, backend);
solver! = Turbulox.poissonsolver(setup);

u = Turbulox.vectorfield(setup);
cache = (;
    ustart = Turbulox.vectorfield(setup),
    du = Turbulox.vectorfield(setup),
    p = Turbulox.scalarfield(setup),
);

# Initialize
x = range(0f, 1f, grid.n + 1)[2:end];
y = reshape(x, 1, :);
z = reshape(x, 1, 1, :);
Δx = x[2] - x[1];
xp = x .- Δx / 2;
yp = reshape(xp, 1, :);
zp = reshape(xp, 1, 1, :);
u[:, :, :, 1] .= Ux.(x, yp, zp);
u[:, :, :, 2] .= Uy.(xp, y, zp);
u[:, :, :, 3] .= Uz.(xp, yp, z);
Turbulox.project!(u, cache.p, solver!, setup);

"Compute ``z``-component of vorticity in the plane ``z=z``."
@kernel function vort_z(grid, ω, u, z)
    x, y = @index(Global, NTuple)
    X = CartesianIndex((x, y, z))
    δux_δy = Turbulox.δ(grid, u, X, 1, 2)
    δuy_δx = Turbulox.δ(grid, u, X, 2, 1)
    ω[x, y] = -δux_δy + δuy_δx
end

fig, ω_obs = let
    ω = KernelAbstractions.zeros(backend, T, grid.n, grid.n)
    Turbulox.apply!(vort_z, setup, ω, u, grid.n ÷ 4)
    ω_obs = Observable(Array(ω))
    colorrange = lift(ω_obs) do ω
        r = maximum(abs, ω)
        -r, r
    end
    fig, ax, hm = image(
        ω_obs;
        colorrange = (-5, 5),
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
    # Colorbar(fig[1, 2], hm)
    fig, ω_obs
end

v = copy(u);
# u = v;

stream = VideoStream(fig; visible = true);

t = 0f
i = 0
tmax = 50f
while t < tmax
    i += 1
    Δt = T(0.85) * Turbulox.propose_timestep(u, setup)
    Δt = min(Δt, tmax - t)
    Turbulox.timestep!(Turbulox.default_right_hand_side!, u, cache, Δt, solver!, setup)
    t += Δt
    @show t
    if i % 20 == 0
        Turbulox.apply!(vort_z, setup, ω, u, grid.n ÷ 4)
        ω_obs[] = copyto!(ω_obs[], ω)
        recordframe!(stream)
    end
end

save("taylorgreen_onevortex.mp4", stream)

ubar = Turbulox.vectorfield(setup);
∇u = Turbulox.collocated_tensorfield(setup);
q = Turbulox.scalarfield(setup);
r = Turbulox.scalarfield(setup);
Turbulox.gaussian!(ubar, u, T(1/50), setup);
Turbulox.apply!(Turbulox.velocitygradient_collocated!, setup, ∇u, u);
# Turbulox.apply!(Turbulox.velocitygradient!, setup, ∇u, ubar);
Turbulox.apply!(Turbulox.compute_qr!, setup, q, r, ∇u);

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
        colorrange = (0, 1),
        # lowclip = :transparent,
    )
    fig
end

Q[] = copyto!(Q[], view(q, :, 220, :))

s = Turbulox.get_scale_numbers(u, setup)

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
