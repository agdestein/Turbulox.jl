"Simulate turbulence in a box."
module Turbulox

using Adapt
using FFTW # For pressure solver and spectrum computation
using KernelAbstractions # For CPU/GPU kernels
using KernelAbstractions.Extras.LoopInfo: @unroll
using LinearAlgebra
using Random # For reproducibility
using StaticArrays # For tensors

include("concept.jl")
include("initializers.jl")
include("operators.jl")
include("tensors.jl")
include("closures.jl")
include("time.jl")
include("filters.jl")
include("utils.jl")

export Stag, Coll, Grid, order, dim, dx, get_axis, problem_setup, apply!
export scalarfield, vectorfield, collocated_tensorfield, staggered_tensorfield, randomfield
export convectiondiffusion!
export pol_tensor_stag!, tensorproduct_stag!, tensorproduct_coll!, tensordivergence!
export tophat, gaussian, applyfilter!
export propose_timestep, timestep!
export spectral_stuff, spectrum

end
