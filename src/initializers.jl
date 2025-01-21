"Allocate empty scalar field."
scalarfield(setup) = KernelAbstractions.zeros(
    setup.backend,
    typeof(setup.visc),
    ntuple(Returns(setup.grid.n), dim(setup.grid)),
)

"Allocate empty vector field."
vectorfield(setup) = KernelAbstractions.zeros(
    setup.backend,
    typeof(setup.visc),
    ntuple(Returns(setup.grid.n), dim(setup.grid))...,
    dim(setup.grid),
)

"Allocate empty tensor field."
function tensorfield(setup)
    (; backend, grid, visc) = setup
    d = dim(grid)
    d2 = d * d
    T = typeof(visc)
    KernelAbstractions.zeros(backend, SMatrix{d,d,T,d2}, ntuple(Returns(grid.n), D))
end
