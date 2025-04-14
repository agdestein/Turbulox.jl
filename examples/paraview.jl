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

T = Float32
# T = Float64

# backend = KernelAbstractions.CPU()
backend = CUDABackend()

visc = 1 / 10_000 |> T

grid = Turbulox.Grid(; order = 2, dim = 3, n = 256);
setup = Turbulox.problem_setup(; grid, visc, backend);
solver! = Turbulox.poissonsolver(setup);

ustart = Turbulox.randomfield(setup, solver!; kp = 5);
cache = (;
    ustart = Turbulox.vectorfield(setup),
    du = Turbulox.vectorfield(setup),
    p = Turbulox.scalarfield(setup),
);

function rhs!(du, u, setup)
    fill!(du, 0)
    Turbulox.apply!(Turbulox.convectiondiffusion!, setup, du, u, setup.visc)
    # closure!(du, u)
end

x = range(0 |> T, 1 |> T, grid.n + 1)[2:end] .- T(1) / grid.n
outdir = joinpath(@__DIR__, "dns") |> mkpath
# rm(outdir; recursive = true)
# outdir |> mkpath

let
    u = copy(ustart)
    t = 0 |> T
    i = 0
    tmax = 5 |> T
    ∇u = Turbulox.collocated_tensorfield(setup)
    q = Turbulox.scalarfield(setup)
    q_cpu = q |> Array
    pvd = paraview_collection("$outdir/q")
    while t < tmax
        Δt = T(0.85) * Turbulox.propose_timestep(u, setup)
        Δt = min(Δt, tmax - t)
        if i > 0
            Turbulox.timestep!(rhs!, u, cache, Δt, solver!, setup)
            t += Δt
        end
        stepsleft = round(Int, (tmax - t) / Δt)
        @show t, stepsleft
        if i % 20 == 0
            Turbulox.apply!(Turbulox.velocitygradient_coll!, setup, ∇u, u)
            Turbulox.apply!(Turbulox.compute_q!, setup, q, ∇u)
            copyto!(q_cpu, q)
            vtk_grid("$outdir/q_$i", x, x, x) do vtk
                vtk["Q"] = q |> Array
                vtk["TimeValue"] = t
                pvd[t] = vtk
            end
        end
        i += 1
    end
    close(pvd)
end
