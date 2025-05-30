using Turbulox: ∇_collocated
if false
    include("../src/Turbulox.jl")
    using .Turbulox
end

using BenchmarkTools
using CUDA
using Turbulox

T = Float64
# backend = CPU()
backend = CUDABackend()
grid = Turbulox.Grid(; ho = Val(1), n = 256, L = 1.0, backend);
poisson = Turbulox.poissonsolver(grid);
cache = (;
    ustart = Turbulox.VectorField(grid),
    du = Turbulox.VectorField(grid),
    p = Turbulox.ScalarField(grid),
);

# Initial conditions
Ux(x, y, z) = sinpi(2x) * cospi(2y) * sinpi(2z) / 2
Uy(x, y, z) = -cospi(2x) * sinpi(2y) * sinpi(2z) / 2
Uz(x, y, z) = zero(x)
ustart = Turbulox.VectorField(grid);
let
    x = range(0 |> T, 1 |> T, grid.n + 1)[2:end];
    Δx = x[2] - x[1];
    x = x .- Δx / 2;
    y = reshape(x, 1, :);
    z = reshape(x, 1, 1, :);
    Δx = x[2] - x[1];
    xp = x
    # xp = x .- Δx / 2;
    yp = reshape(xp, 1, :);
    zp = reshape(xp, 1, 1, :);
    ustart.data[:, :, :, 1] .= Ux.(x, yp, zp);
    ustart.data[:, :, :, 2] .= Uy.(xp, y, zp);
    ustart.data[:, :, :, 3] .= Uz.(xp, yp, z);
    Turbulox.project!(ustart, cache.p, poisson);
end

@benchmark let
    u = ustart
    visc = eltype(u)(5e-5)
    apply!(tensorapply!, u.grid, convdiff, cache.du, u, visc)
    # apply!(tensorapply!, u.grid, diffusion, cache.du, u, visc)
    # apply!(tensorapply!, u.grid, conv, cache.du, u)
end

@benchmark let
    u = ustart
    visc = eltype(u)(5e-5)
    σ = stresstensor(u, visc)
    fill!(cache.du.data, 0)
    apply!(tensordivergence!, u.grid, cache.du, σ)
end

function rhs!(du, u)
    visc = eltype(du)(5e-5)
    # apply!(tensorapply!, u.grid, convdiff, du, u, visc)
    σ = stresstensor(u, visc)
    fill!(cache.du.data, 0)
    apply!(tensordivergence!, u.grid, cache.du, σ)
    du
end

u = VectorField(grid)

@time let
    copyto!(u.data, ustart.data)
    visc = 5e-5 |> T
    t = 0 |> T
    i = 0
    tmax = 1 |> T
    while t < tmax
        i += 1
        Δt = T(0.85) * Turbulox.propose_timestep(u, visc)
        Δt = min(Δt, tmax - t)
        Turbulox.timestep!(rhs!, u, cache, Δt, solver!)
        t += Δt
        @show t
    end
end

# 14.969949 seconds (6.48 M allocations: 320.616 MiB, 0.22% gc time, 8.21% compilation time)

# Tensor:
# 16.906574 seconds (375.98 k allocations: 13.007 MiB, 0.10% compilation time: 100% of which was recompilation)
