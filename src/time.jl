# Time stepping

"Default right-hand side function (without projection)."
right_hand_side!(du, u; viscosity) =
    apply!(tensorapply!, u.grid, convdiff, du, u, viscosity)

"Perform time step using Wray's third-order scheme."
function timestep!(f!, u, cache, Δt, poisson; params...)
    (; ustart, du, p) = cache
    T = eltype(u)

    # Low-storage Butcher tableau:
    # c1 | 0             ⋯   0
    # c2 | a1  0         ⋯   0
    # c3 | b1 a2  0      ⋯   0
    # c4 | b1 b2 a3  0   ⋯   0
    #  ⋮ | ⋮   ⋮  ⋮  ⋱   ⋱   ⋮
    # cn | b1 b2 b3  ⋯ an-1  0
    # ---+--------------------
    #    | b1 b2 b3  ⋯ bn-1 an
    #
    # Note the definition of (ai)i.
    # They are shifted to simplify the for-loop.
    # TODO: Make generic by passing a, b, c as inputs
    a = T(8 / 15), T(5 / 12), T(3 / 4)
    b = T(1 / 4), T(0)
    c = T(0), T(8 / 15), T(2 / 3)
    nstage = length(a)

    # Update current solution
    copyto!(ustart.data, u.data)

    for i = 1:nstage
        f!(du, u; params...)

        # Compute u = project(ustart + Δt * a[i] * du)
        i == 1 || copyto!(u.data, ustart.data) # Skip first iter
        axpy!(a[i] * Δt, du.data, u.data)
        project!(u, p, poisson)

        # Compute ustart = ustart + Δt * b[i] * du
        i == nstage || axpy!(b[i] * Δt, du.data, ustart.data) # Skip last iter
    end

    u
end

"Get proposed maximum time step for convection and diffusion terms."
function propose_timestep(u, visc)
    g = u.grid
    Δt_diff = dx(g)^2 / 3 / 2 / visc
    Δt_conv = minimum(u -> dx(g) / abs(u), u.data)
    Δt = min(Δt_diff, Δt_conv)
    Δt
end
