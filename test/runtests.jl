if false
    # These are implicitly loaded in `@testitem`s
    include("../src/Turbulox.jl")
    using .Turbulox
    using Test
end

using TestItemRunner

@run_package_tests

@testitem "Code quality (Aqua.jl)" begin
    using Aqua
    Aqua.test_all(Turbulox; project_extras = false)
end

@testitem "Code linting (JET.jl)" begin
    using JET
    JET.test_package(Turbulox; target_defined_modules = true)
end

@testitem "Consistency of weights" begin
    using Turbulox: w4_1, w4_3
    using Turbulox: w6_1, w6_3, w6_5
    using Turbulox: w8_1, w8_3, w8_5, w8_7
    using Turbulox: w10_1, w10_3, w10_5, w10_7, w10_9
    # Use equality, since weights are rational
    @test w4_1 + w4_3 == 1
    @test w6_1 + w6_3 + w6_5 == 1
    @test w8_1 + w8_3 + w8_5 + w8_7 == 1
    @test w10_1 + w10_3 + w10_5 + w10_7 + w10_9 == 1
end

@testitem "Consistency of Laplace stencils" begin
    using Turbulox: laplace_stencil
    for order in [2, 4, 6, 8, 10]
        g = Grid(; order, dim = 3, L = 1.0, n = 16)
        stencil = laplace_stencil(g)
        # Use equality, since weights are rational
        @test sum(stencil) ≈ 0 atol = 1e-12 # Constant functions
        @test sum(eachindex(stencil) .* stencil) ≈ 0 atol = 1e-12 # Linear functions
    end
end

@testitem "Pressure projection" begin
    for order in [2, 4, 6, 8, 10]
        grid = Grid(; order, dim = 3, L = 1.0, n = 16)
        solver! = poissonsolver(grid)
        u = randn(grid.n, grid.n, grid.n, 3)
        p = scalarfield(grid)
        project!(u, p, solver!, grid)
        div = scalarfield(grid)
        apply!(divergence!, grid, div, u)
        @test maximum(abs, div) < 1e-12
    end
end

@testitem "(Skew-)symmetry of operators" begin
    using LinearAlgebra
    for order in [2, 4, 6, 8, 10]
        grid = Grid(; order, dim = 3, L = 1.0, n = 16)
        solver! = poissonsolver(grid)
        u = randomfield(grid, solver!)

        # Check that the convection operator is skew-symmetric
        # for a divergence-free field
        u = randomfield(grid, solver!) # Divergence-free
        du = zero(u)
        apply!(convection!, grid, du, u)
        dE = dot(u, du) / grid.n^3
        @test abs(dE) < 1e-12

        # Test that it is not skew-symmetric for a non-divergence-free field
        # This is because we use the divergence-form
        # (see Morinishi et al. 1998)
        u = randn(grid.n, grid.n, grid.n, 3) # Non-divergence-free
        du = zero(u)
        apply!(convection!, grid, du, u)
        dE = dot(u, du) / grid.n^3
        @test abs(dE) > 1e-12

        # Check that the diffusion operator is dissipative
        u = randn(grid.n, grid.n, grid.n, 3) # Non-divergence-free
        du = zero(u)
        apply!(diffusion!, grid, du, u, 1e-3)
        dE = dot(u, du) / grid.n^3
        @test dE < 0
    end
end

@testitem "Eddy viscosity" begin
    using Turbulox: velocitygradient!
    using Turbulox:
        smagorinsky_viscosity!,
        wale_viscosity!,
        vreman_viscosity!,
        verstappen_viscosity!,
        nicoud_viscosity!
    grid = Grid(; order = 2, dim = 3, L = 1.0, n = 16)
    u = randn(grid.n, grid.n, grid.n, 3)
    ∇u = staggered_tensorfield(grid)
    visc = scalarfield(grid)
    apply!(velocitygradient!, grid, ∇u, u)
    Δ = 1 / grid.n
    @testset "Smagorinsky" begin
        apply!(smagorinsky_viscosity!, grid, visc, ∇u, 0.17, Δ)
        @test all(>(0), visc)
    end
    @testset "WALE" begin
        apply!(wale_viscosity!, grid, visc, ∇u, 0.569, Δ)
        @test all(>(0), visc)
    end
    @testset "Vreman" begin
        apply!(vreman_viscosity!, grid, visc, ∇u, 0.28, Δ)
        @test all(>(0), visc)
    end
    @testset "Verstappen" begin
        apply!(verstappen_viscosity!, grid, visc, ∇u, 0.345, Δ)
        @test all(>(0), visc)
    end
    @testset "Nicoud" begin
        apply!(nicoud_viscosity!, grid, visc, ∇u, 1.35, Δ)
        @test all(>(0), visc)
    end
end
