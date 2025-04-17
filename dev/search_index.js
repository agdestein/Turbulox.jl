var documenterSearchIndex = {"docs":
[{"location":"references/#References","page":"References","title":"References","text":"","category":"section"},{"location":"references/","page":"References","title":"References","text":"R. A. Clark, J. H. Ferziger and W. C. Reynolds. Evaluation of subgrid-scale models using an accurately simulated turbulent flow. Journal of Fluid Mechanics 91, 1–16 (1979). Accessed on Jan 9, 2025.\n\n\n\nS. B. Pope. Turbulent flows (Cambridge University Press, Cambridge ; New York, 2000).\n\n\n\nF. Nicoud, H. B. Toda, O. Cabrit, S. Bose and J. Lee. Using singular values to build a subgrid-scale model for large eddy simulations. Physics of Fluids 23, 085106 (2011). Accessed on Jan 10, 2025.\n\n\n\nF. X. Trias, D. Folch, A. Gorobets and A. Oliva. Building proper invariants for eddy-viscosity subgrid-scale models. Physics of Fluids 27, 065103 (2015). Accessed on Jan 9, 2025.\n\n\n\nJ. Smagorinsky. General circulation experiments with the primitive equations: I. The basic experiment. Monthly Weather Review 91, 99–164 (1963). Accessed on Jan 4, 2025.\n\n\n\nR. Verstappen. When Does Eddy Viscosity Damp Subfilter Scales Sufficiently? Journal of Scientific Computing 49, 94–110 (2011). Accessed on Jan 4, 2025.\n\n\n\nA. W. Vreman. An eddy-viscosity subgrid-scale model for turbulent shear flow: Algebraic theory and applications. Physics of Fluids 16, 3670–3681 (2004). Accessed on Jan 4, 2025.\n\n\n\nF. Nicoud and F. Ducros. Subgrid-Scale Stress Modelling Based on the Square of the Velocity Gradient Tensor. Flow, Turbulence and Combustion 62, 183–200 (1999). Accessed on Jan 10, 2025.\n\n\n\n","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = Turbulox","category":"page"},{"location":"#Turbulox","page":"Home","title":"Turbulox","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for Turbulox.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [Turbulox]","category":"page"},{"location":"#Turbulox.Turbulox","page":"Home","title":"Turbulox.Turbulox","text":"Simulate turbulence in a box.\n\n\n\n\n\n","category":"module"},{"location":"#Turbulox.Coll","page":"Home","title":"Turbulox.Coll","text":"Collocated grid position.\n\n\n\n\n\n","category":"type"},{"location":"#Turbulox.Grid","page":"Home","title":"Turbulox.Grid","text":"Staggered grid of order o and dimension d.\n\n\n\n\n\n","category":"type"},{"location":"#Turbulox.Stag","page":"Home","title":"Turbulox.Stag","text":"Staggered grid position.\n\n\n\n\n\n","category":"type"},{"location":"#Turbulox.apply!-Tuple{Any, Grid, Vararg{Any}}","page":"Home","title":"Turbulox.apply!","text":"Apply kernel! on grid, args... over the entire domain. The args are typically input and output fields. The kernel should be of the form\n\nusing KernelAbstractions\n@kernel function kernel!(grid, args...)\n    # content\nend\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.applyfilter!-NTuple{6, Any}","page":"Home","title":"Turbulox.applyfilter!","text":"Filter scalar field.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.applyfilter!-Tuple{Any, Any, Any, Any, Any, Any, Stag, Stag}","page":"Home","title":"Turbulox.applyfilter!","text":"Filter staggered tensor field.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.applyfilter!-Tuple{Any, Any, Any, Any, Any, Any, Stag}","page":"Home","title":"Turbulox.applyfilter!","text":"Filter staggered vector field.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.clark_model-Tuple{Any, Any}","page":"Home","title":"Turbulox.clark_model","text":"Clark's gradient model [1].\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.collocated_tensorfield-Tuple{Grid}","page":"Home","title":"Turbulox.collocated_tensorfield","text":"Allocate empty tensor field (collocated).\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.convectiondiffusion!","page":"Home","title":"Turbulox.convectiondiffusion!","text":"Compute convection-diffusion force from velocity u. Add the force field to f.\n\n\n\n\n\n","category":"function"},{"location":"#Turbulox.convterm","page":"Home","title":"Turbulox.convterm","text":"Approximate the convective force partial_j (u_i u_j).\n\n\n\n\n\n","category":"function"},{"location":"#Turbulox.default_right_hand_side!-NTuple{4, Any}","page":"Home","title":"Turbulox.default_right_hand_side!","text":"Default right-hand side function (without projection).\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.desymmetrize!-Union{Tuple{o}, Tuple{Grid{o, 2}, Any, Any}} where o","page":"Home","title":"Turbulox.desymmetrize!","text":"Copy symmtric tensor to normal tensor\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.deviator-Tuple{Any}","page":"Home","title":"Turbulox.deviator","text":"Compute deviatoric part of a tensor.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.dim-Union{Tuple{Grid{o, d}}, Tuple{d}, Tuple{o}} where {o, d}","page":"Home","title":"Turbulox.dim","text":"Get physical dimension.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.divergence!","page":"Home","title":"Turbulox.divergence!","text":"Compute divergence of vector field u. Put the result in div.\n\n\n\n\n\n","category":"function"},{"location":"#Turbulox.dx-Tuple{Grid}","page":"Home","title":"Turbulox.dx","text":"Get grid spacing.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.e-Tuple{Any, Any}","page":"Home","title":"Turbulox.e","text":"Get unit index in dimension i.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.eddyviscosity_closure!-Tuple{Any}","page":"Home","title":"Turbulox.eddyviscosity_closure!","text":"Divergence of -2 * visc * S. Interpolate visc to staggered points first. Subtract result from existing force field f.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.eddyviscosity_model-NTuple{4, Any}","page":"Home","title":"Turbulox.eddyviscosity_model","text":"Eddy viscosity closure model.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.expand_tensorbasis!-Tuple{Any}","page":"Home","title":"Turbulox.expand_tensorbasis!","text":"Expand tensor basis B and fill τ[x] = c[i, x] * B[i, x] (sum over i).\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.fill_invariants!-Tuple{Any}","page":"Home","title":"Turbulox.fill_invariants!","text":"Fill V with invariants.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.fill_tensorbasis!-Tuple{Any}","page":"Home","title":"Turbulox.fill_tensorbasis!","text":"Fill B with basis tensors.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.gaussian-Tuple{Any, Any, Any}","page":"Home","title":"Turbulox.gaussian","text":"Gaussian filter kernel.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.get_axis-Tuple{Grid, Stag}","page":"Home","title":"Turbulox.get_axis","text":"Get grid points along axis.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.get_scale_numbers-Tuple{Any, Any, Any}","page":"Home","title":"Turbulox.get_scale_numbers","text":"Get the following dimensional scale numbers [2]:\n\nVelocity u_textavg = langle u_i u_i rangle^12\nDissipation rate epsilon = 2 nu langle S_ij S_ij rangle\nKolmolgorov length scale eta = (fracnu^3epsilon)^14\nTaylor length scale lambda = (frac5 nuepsilon)^12 u_textavg\nTaylor-scale Reynolds number Re_lambda = fraclambda u_textavgsqrt3 nu\nIntegral length scale L = frac3 pi2 u_textavg^2 int_0^infty fracE(k)k  mathrmd k\nLarge-eddy turnover time tau = fracLu_textavg\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.getval-Union{Tuple{Val{x}}, Tuple{x}} where x","page":"Home","title":"Turbulox.getval","text":"Get value from Val.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.mergestencil-Tuple{Any, Any}","page":"Home","title":"Turbulox.mergestencil","text":"Merge stencil periodically if the stencil is longer than the grid size n.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.nicoud_viscosity!-Tuple{Any}","page":"Home","title":"Turbulox.nicoud_viscosity!","text":"Compute Nicoud's eddy viscosity (sigma-model) [3, 4] Proposed value for C is 1.35.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.order-Union{Tuple{Grid{o, d}}, Tuple{d}, Tuple{o}} where {o, d}","page":"Home","title":"Turbulox.order","text":"Get order of grid.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.poissonsolver-Tuple{Any}","page":"Home","title":"Turbulox.poissonsolver","text":"Create spectral Poisson solver from grid.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.pol_tensor_collocated-Union{Tuple{o}, Tuple{Grid{o, 2}, Any, Any}} where o","page":"Home","title":"Turbulox.pol_tensor_collocated","text":"Interpolate staggered tensor to volume centers.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.pol_tensor_stag-Tuple{Grid, Vararg{Any, 4}}","page":"Home","title":"Turbulox.pol_tensor_stag","text":"Interpolate collocated tensor to staggered tensor.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.pressuregradient!","page":"Home","title":"Turbulox.pressuregradient!","text":"Subtract pressure gradient.\n\n\n\n\n\n","category":"function"},{"location":"#Turbulox.project!-NTuple{4, Any}","page":"Home","title":"Turbulox.project!","text":"Project velocity field onto divergence-free space.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.propose_timestep-Tuple{Any, Any, Any}","page":"Home","title":"Turbulox.propose_timestep","text":"Get proposed maximum time step for convection and diffusion terms.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.s3pqr_viscosity!-Tuple{Any}","page":"Home","title":"Turbulox.s3pqr_viscosity!","text":"Compute S3PQR eddy viscosity [4]\n\nProposed values for p (set valp = Val(p)):\n\n-5/2\n-1\n0\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.scalarfield-Tuple{Grid}","page":"Home","title":"Turbulox.scalarfield","text":"Allocate empty scalar field.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.smagorinsky_viscosity!-Tuple{Any}","page":"Home","title":"Turbulox.smagorinsky_viscosity!","text":"Compute Smagorinsky's original eddy viscosity [5]. Proposed value for C is 0.17.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.stress-Tuple{Grid, Vararg{Any, 5}}","page":"Home","title":"Turbulox.stress","text":"Get convection-diffusion stress tensor component i,j.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.surfacefilter_tensor!-NTuple{6, Any}","page":"Home","title":"Turbulox.surfacefilter_tensor!","text":"Surface-average staggered tensor field.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.symmetric_tensorfield-Tuple{Grid}","page":"Home","title":"Turbulox.symmetric_tensorfield","text":"Allocate empty tensor field (symmetric, staggered).\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.tensorbasis","page":"Home","title":"Turbulox.tensorbasis","text":"Compute Pope's tensor basis [2].\n\n\n\n\n\n","category":"function"},{"location":"#Turbulox.tensordivergence!-Tuple{Any}","page":"Home","title":"Turbulox.tensordivergence!","text":"Divergence of staggered tensor field σ. Subtract result from existing force field f. The operation is f_i leftarrow f_i - _j σ_i j.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.tensordivergence_collocated!-Tuple{Any}","page":"Home","title":"Turbulox.tensordivergence_collocated!","text":"Divergence of collocated tensor field sigma. First interpolate to staggered points. Subtract result from existing force field f. The operation is f_i leftarrow f_i - _j σ_i j.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.tensordivergence_collocated_2!-Tuple{Any}","page":"Home","title":"Turbulox.tensordivergence_collocated_2!","text":"Divergence first, then interpolate\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.tensorfield-Tuple{Grid}","page":"Home","title":"Turbulox.tensorfield","text":"Allocate empty tensor field (staggered).\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.tensorproduct_coll!-Tuple{Any}","page":"Home","title":"Turbulox.tensorproduct_coll!","text":"Compute u v^T in the collocated points.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.tensorproduct_stag!-Tuple{Any}","page":"Home","title":"Turbulox.tensorproduct_stag!","text":"Compute u v^T in the staggered points.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.timestep!-NTuple{6, Any}","page":"Home","title":"Turbulox.timestep!","text":"Perform time step using Wray's third-order scheme.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.vectorfield-Tuple{Grid}","page":"Home","title":"Turbulox.vectorfield","text":"Allocate empty vector field.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.verstappen_viscosity!-Tuple{Any}","page":"Home","title":"Turbulox.verstappen_viscosity!","text":"Compute Verstappen's eddy viscosity [4, 6]. Proposed values for C are 0.345 [6] or 0.527 [4].\"\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.volumefilter_tensor!-NTuple{5, Any}","page":"Home","title":"Turbulox.volumefilter_tensor!","text":"Volume-average staggered tensor field.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.vreman_viscosity!-Tuple{Any}","page":"Home","title":"Turbulox.vreman_viscosity!","text":"Compute Vreman's eddy viscosity [4, 7]. Proposed value for C is 0.28.\n\n\n\n\n\n","category":"method"},{"location":"#Turbulox.wale_viscosity!-Tuple{Any}","page":"Home","title":"Turbulox.wale_viscosity!","text":"Compute WALE [4, 8]. Proposed value for C is 0.569.\n\n\n\n\n\n","category":"method"}]
}
