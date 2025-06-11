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

export Stag, Coll, X, Y, Z, Direction
export Position, Center, FaceX, FaceY, FaceZ, EdgeX, EdgeY, EdgeZ, Corner
export vectorposition, tensorposition
export Grid, order, dx, get_axis, apply!
export ScalarField, VectorField, collocated_tensorfield, TensorField, randomfield
export LazyScalarField, LazyVectorField, LazyTensorField
export materialize!
export poissonsolver,
    poissonsolve!,
    pressuregradient!,
    project!,
    divergence!,
    convdiff,
    conv,
    lap,
    diffusion,
    tensorapply!,
    stresstensor,
    stresstensor!,
    stress,
    δ
export desymmetrize!,
    pol_tensor_stag!, tensorproduct_stag!, tensorproduct_coll!, tensordivergence!
export symmetrize!
export tophat, gaussian, applyfilter!
export volumefilter!, surfacefilter!, linefilter!
export propose_timestep, timestep!, right_hand_side!
export spectral_stuff, spectrum, get_scale_numbers
export eddyviscosity_model!,
    clark_model!,
    smagorinsky_viscosity!,
    wale_viscosity!,
    vreman_viscosity!,
    verstappen_viscosity!,
    nicoud_viscosity!

end
