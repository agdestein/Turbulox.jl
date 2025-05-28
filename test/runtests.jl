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

# @testitem "Code linting (JET.jl)" begin
#     using JET
#     JET.test_package(Turbulox; target_defined_modules = true)
# end

@testitem "Consistency of weights" begin
    using Turbulox: w
    g(ho) = Grid(; ho, n = 8, L = 1.0)
    # Use equality, since weights are rational
    @test Val(1) |> g |> w |> sum == 1
    @test Val(2) |> g |> w |> sum == 1
    @test Val(3) |> g |> w |> sum == 1
    @test Val(4) |> g |> w |> sum == 1
    @test Val(5) |> g |> w |> sum == 1
end

@testitem "Consistency of Laplace stencils" begin
    using Turbulox: laplace_stencil
    for ho = 1:5
        g = Grid(; ho = Val(ho), L = 1.0, n = 16)
        stencil = laplace_stencil(g)
        # Use equality, since weights are rational
        @test sum(stencil) ≈ 0 atol = 1e-12 # Constant functions
        @test sum(eachindex(stencil) .* stencil) ≈ 0 atol = 1e-12 # Linear functions
    end
end

@testitem "Pressure projection" begin
    for ho = 1:5
        grid = Grid(; ho = Val(ho), L = 1.0, n = 16)
        poisson = poissonsolver(grid)
        u = VectorField(grid, randn(grid.n, grid.n, grid.n, 3))
        p = ScalarField(grid)
        project!(u, p, poisson)
        div = ScalarField(grid)
        apply!(divergence!, grid, div, u)
        @test maximum(abs, div) < 1e-12
    end
end

@testitem "(Skew-)symmetry of operators" begin
    using LinearAlgebra
    for ho = 1:5
        grid = Grid(; ho = Val(ho), L = 1.0, n = 16)
        solver! = poissonsolver(grid)
        # Check that the convection operator is skew-symmetric
        # for a divergence-free field
        u = randomfield(grid, solver!) # Divergence-free
        du = VectorField(grid)
        apply!(tensorapply!, grid, conv, du, u)
        dE = dot(u.data, du.data) / grid.n^3
        @test abs(dE) < 1e-12
        # Test that it is not skew-symmetric for a non-divergence-free field
        # This is because we use the divergence-form
        # (see Morinishi et al. 1998)
        u = VectorField(grid, randn(grid.n, grid.n, grid.n, 3)) # Non-divergence-free
        du = VectorField(grid)
        apply!(tensorapply!, grid, conv, du, u)
        dE = dot(u.data, du.data) / grid.n^3
        @test abs(dE) > 1e-12
        # Check that the diffusion operator is dissipative
        u = VectorField(grid, randn(grid.n, grid.n, grid.n, 3)) # Non-divergence-free
        du = VectorField(grid)
        apply!(tensorapply!, grid, diffusion, du, u, 1e-3)
        dE = dot(u.data, du.data) / grid.n^3
        @test dE < 0
    end
end

# Check that divergence of convection-diffusion stress is the same as direct
# implementation.
# Note: Direct implementation does div(nu grad u),
# while stress tensor is div(nu (grad u + grad u^T)).
# These are only the same for divergence-free fields.
@testitem "Stress tensor" begin
    using LinearAlgebra
    grid = Grid(; L = 1.0, n = 16)
    u = VectorField(grid, randn(grid.n, grid.n, grid.n, 3))
    # Stress with grad u and sym(grad u) are only the same for divergence-free fields
    p = ScalarField(grid)
    poisson = poissonsolver(grid)
    project!(u, p, poisson)
    f1 = VectorField(grid)
    f2 = VectorField(grid)
    visc = 1e-3
    apply!(tensorapply!, u.grid, convdiff, f1, u, visc)
    # Lazy stress tensor
    σ = stresstensor(u, visc)
    apply!(tensordivergence!, grid, f2, σ)
    @test norm(f1.data - f2.data) < 1e-10
end

# All eddy viscosities should be positive
@testitem "Eddy viscosity" begin
    using Turbulox: velocitygradient!
    using Turbulox:
        smagorinsky_viscosity!,
        wale_viscosity!,
        vreman_viscosity!,
        verstappen_viscosity!,
        nicoud_viscosity!
    grid = Grid(; L = 1.0, n = 16)
    u = VectorField(grid, randn(grid.n, grid.n, grid.n, 3))
    ∇u = TensorField(grid)
    visc = ScalarField(grid)
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
