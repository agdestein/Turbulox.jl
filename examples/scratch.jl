if false
    include("src/Turbulox.jl")
    using .Turbulox
end

T = Float64

using CUDA
using LinearAlgebra
using Random
using Turbulox
using WGLMakie

grid = Turbulox.Grid(; order = 8, dim = 3, n = 32);
setup = Turbulox.problem_setup(; grid, visc = 1 / 6000, backend = CUDABackend());
solver! = Turbulox.poissonsolver(setup);
cache = (;
    ustart = Turbulox.vectorfield(setup),
    du = Turbulox.vectorfield(setup),
    p = Turbulox.scalarfield(setup),
);

ubar = Turbulox.vectorfield(setup);
τ = Turbulox.collocated_tensorfield(setup);
σ = Turbulox.collocated_tensorfield(setup);
Turbulox.gaussian!(ubar, u, 0.1, setup);
Turbulox.apply!(Turbulox.tensorproduct!, setup, σ, u, u)
Turbulox.gaussian!(τ, σ, 0.1, setup);
Turbulox.apply!(Turbulox.tensorproduct!, setup, σ, ubar, ubar)
@. τ = τ - σ

CUDA.@allowscalar τ[1]
CUDA.@allowscalar uu[1]
CUDA.@allowscalar uubar[1]

Turbulox.filter_kernel!

solver!.ahat.contents[1] / grid.n^2

u = Turbulox.randomfield(setup, solver!);

du = zero(u);
Turbulox.apply!(Turbulox.convection!, setup, du, u);

dot(u, du) / grid.n^3

force = copy(u);
# closure! = Turbulox.smagorinsky_model(setup, 0.1, 0.02)
closure! = Turbulox.clark_model(setup, 0.02)
closure!(force, u);

ubar = zero(u);

Turbulox.gaussian!(ubar, u, 0.1, setup)

u

p = Turbulox.scalarfield(setup);
Turbulox.apply!(Turbulox.divergence!, setup, p, u)
p

Turbulox.project!(u, p, solver!, setup);
u

u[:, :, 1] |> Array |> heatmap

du = zero(u);
@benchmark begin
    fill!(du, 0)
    Turbulox.apply!(Turbulox.convectiondiffusion!, setup, du, u, setup.visc)
end

du

ω = Turbulox.scalarfield(setup)
ω_cpu = Array(ω)
vort = Observable(ω_cpu)
fig, ax, hm = image(vort; colormap = :viridis)
Colorbar(fig[1, 2], hm)

v = copy(u);
copyto!(u, v);

stuff = Turbulox.spectral_stuff(setup);
spec = Turbulox.spectrum(u, setup; stuff);
curve = Observable(Point2f.(spec.κ, spec.s));
lines(curve; axis = (; xscale = log10, yscale = log10))

# closure! = Turbulox.smagorinsky_model(setup, T(0.17), T(1) / grid.n)
closure! = Turbulox.clark_model(setup, 1 / grid.n)

function rhs!(du, u, setup)
    fill!(du, 0)
    Turbulox.apply!(Turbulox.convectiondiffusion!, setup, du, u, setup.visc)
    closure!(du, u)
end

for i = 1:20
    Δt = 0.4 * Turbulox.propose_timestep(u, setup)
    Turbulox.timestep!(rhs!, u, cache, Δt, solver!, setup)
    @show Δt
    if i % 1 == 0
        # Turbulox.apply!(Turbulox.vorticity!, setup, ω, u)
        # vort[] = copyto!(ω_cpu, ω)
        spec = Turbulox.spectrum(u, setup; stuff)
        curve[] = Point2f.(spec.κ, spec.s)
        sleep(0.01)
    end
end

s = Turbulox.get_scale_numbers(u, setup)

s.uavg
s.L
s.λ
s.eta
s.t_int
s.t_tay
s.t_kol
s.Re_int
s.Re_tay
s.Re_kol
s.ϵ

spec = Turbulox.spectrum(u, setup; stuff)

let
    fig = Figure()
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10)
    lines!(ax, Point2f.(spec.κ, spec.s))
    xslope = spec.κ[10:end]
    yslope = @. 1.58 * s.ϵ^(2 / 3) * (π * xslope)^(-5 / 3)
    lines!(ax, xslope, yslope; color = Cycled(2))
    # vlines!(ax, 1 / s.L / 1)
    # vlines!(ax, 1 / s.λ / 1)
    fig
end

Turbulox.create_spectrum(; setup, kp = 10)

# div = Turbulox.scalarfield(setup);
# Turbulox.apply!(Turbulox.divergence!, setup, div, u)
# div
#
# Turbulox.project!(u, cache.p, solver!, setup)
#
# Turbulox.apply!(Turbulox.divergence!, setup, div, u)
# div
#
# u = Turbulox.vectorfield(setup);
# f = Turbulox.vectorfield(setup);
# div = Turbulox.scalarfield(setup);
# p = Turbulox.scalarfield(setup);
# randn!(u);
#
# Turbulox.apply!(Turbulox.divergence!, setup, div, u)
# Turbulox.apply!(Turbulox.convectiondiffusion!, setup, f, u, setup.visc)
# Turbulox.project!(u, p, solver!, setup)
#
# Turbulox.apply!(Turbulox.divergence!, setup, div, u)
# div
#
# p
# div
# u
# f
#
# Turbulox.step_wray3!(Turbulox.default_right_hand_side!, u, cache, 1e-3, solver!, setup)
