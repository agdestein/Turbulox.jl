# Time stepping

"Default right-hand side function (without projection)."
function default_right_hand_side!(du, u, setup)
    fill!(du, 0)
    apply!(convectiondiffusion!, setup, du, u, setup.visc)
    du
end

"Perform time step using Wray's third-order scheme."
function timestep!(f!, u, cache, Δt, solver!, setup)
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
    copyto!(ustart, u)

    for i = 1:nstage
        f!(du, u, setup)

        # Compute u = project(ustart + Δt * a[i] * du)
        copyto!(u, ustart)
        axpy!(a[i] * Δt, du, u)
        project!(u, p, solver!, setup)

        # Compute ustart = ustart + Δt * b[i] * du
        # Skip for last iter
        i == nstage || axpy!(b[i] * Δt, du, ustart)
    end

    u
end

"Get proposed maximum time step for convection and diffusion terms."
function propose_timestep(u, setup)
    (; grid, visc) = setup
    (; n) = grid
    Δt_diff = 1 / (2 * visc * n^2)
    Δt_conv = minimum(u -> 1 / abs(u) / n, u)
    Δt = min(Δt_diff, Δt_conv)
    Δt
end
