using Turbulox
using Test
using Aqua
using JET

@testset "Turbulox.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(Turbulox; project_extras = false)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(Turbulox; target_defined_modules = true)
    end
    # Write your tests here.
end
