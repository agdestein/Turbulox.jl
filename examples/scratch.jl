if false
    include("src/Turbulox.jl")
    using .Turbulox
end

using Turbulox
using Random
using CUDA
using GLMakie

grid = Turbulox.Grid(; order = 8, dim = 2, n = 128);
setup = Turbulox.problem_setup(;
    grid,
    visc = 1 / 6000,
    backend = CUDABackend(),
);
solver! = Turbulox.poissonsolver(setup);
cache = (;
    ustart = Turbulox.vectorfield(setup),
    du = Turbulox.vectorfield(setup),
    p = Turbulox.scalarfield(setup),
);
u = Turbulox.vectorfield(setup);
randn!(u);
u .*= 10;

solver!.ahat.contents[1] / grid.n^2

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

f = zero(u);
Turbulox.apply!(Turbulox.convectiondiffusion!, setup, f, u, setup.visc)
f

ω = Turbulox.scalarfield(setup)
ω_cpu = Array(ω)
vort = Observable(ω_cpu)
fig, ax, hm = image(vort; colormap = :viridis)
Colorbar(fig[1, 2], hm)

v = copy(u)
copyto!(u, v)

# closure! = Turbulox.smagorinsky_model(setup, 0.1, 0.01)
closure! = Turbulox.clark_model(setup, 0.01)
function rhs!(du, u, setup)
    fill!(du, 0)
    Turbulox.apply!(Turbulox.convectiondiffusion!, setup, du, u, setup.visc)
    # closure!(du, u)
end

for i = 1:5000
    Δt = 0.4 * Turbulox.propose_timestep(u, setup)
    Turbulox.timestep!(rhs!, u, cache, Δt, solver!, setup)
    @show Δt
    if i % 10 == 0
        Turbulox.apply!(Turbulox.vorticity!, setup, ω, u)
        vort[] = copyto!(ω_cpu, ω)
        sleep(0.01)
    end
end

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

