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

# Turbulox.FFTW.set_provider!("mkl")

T = Float32
# T = Float64

# backend = KernelAbstractions.CPU()
backend = CUDABackend()

visc = 1 / 1_000 |> T

grid = Grid(; ho = Val(2), n = 128, L = 1.0);
solver! = poissonsolver(grid);

ustart = randomfield(grid, solver!; kp = 5);
cache = (; ustart = VectorField(grid), du = VectorField(grid), p = ScalarField(grid));

function rhs!(du, u)
    fill!(du, 0)
    visc = 1 / 1_000 |> eltype(u)
    apply!(Turbulox.convectiondiffusion!, u.grid, du, u, visc)
    # closure!(du, u)
end

x = get_axis(grid, Coll())
outdir = joinpath(@__DIR__, "dns") |> mkpath
# rm(outdir; recursive = true)
# outdir |> mkpath

let
    u = VectorField(grid, copy(ustart.data))
    t = 0 |> T
    i = 0
    tmax = 1 |> T
    ∇u = collocated_tensorfield(grid)
    q = ScalarField(grid)
    Gmag = ScalarField(grid)
    q_cpu = q.data |> Array
    pvd = paraview_collection("$outdir/q")
    while t < tmax
        Δt = T(0.85) * propose_timestep(u, visc)
        Δt = min(Δt, tmax - t)
        if i > 0
            timestep!(rhs!, u, cache, Δt, solver!)
            t += Δt
        end
        stepsleft = round(Int, (tmax - t) / Δt)
        @show t, stepsleft
        if i % 3 == 0
            apply!(Turbulox.velocitygradient_coll!, grid, ∇u, u)
            apply!(Turbulox.compute_q!, grid, q, ∇u)
            Gmag.data .= sqrt.(sum.(abs2, ∇u.data))
            copyto!(q_cpu, q)
            vtk_grid("$outdir/q_$i", x, x, x) do vtk
                vtk["Q"] = q_cpu
                vtk["G"] = Gmag
                vtk["TimeValue"] = t
                pvd[t] = vtk
            end
        end
        i += 1
    end
    close(pvd)
end
