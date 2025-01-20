"Extend index periodically so that it stays within the domain."
function periodicindex end
periodicindex(n) = i -> periodicindex(i, n)
periodicindex(i::Integer, n) = mod1(i, n)
periodicindex(I::CartesianIndex, n) = CartesianIndex(mod1.(I.I, n))

"Get unit indices for given dimension."
function unitindices end
unitindices(::Val{2}) = CartesianIndex((1, 0)), CartesianIndex((0, 1))
unitindices(::Val{3}) =
    CartesianIndex((1, 0, 0)), CartesianIndex((0, 1, 0)), CartesianIndex((0, 0, 1))

"""
Apply `kernel!` on `setup, args...` over the entire domain.
The `args` are typically input and output fields.
The kernel should be of the form
```julia
using KernelAbstractions
@kernel function kernel!(setup, args...)
    # content
end
```
"""
function apply!(kernel!, setup, args...)
    (; D, n, backend, workgroupsize) = setup
    ndrange = ntuple(Returns(n), D)
    kernel!(backend, workgroupsize)(setup, args...; ndrange)
    nothing
end

"""
Compute divergence of vector field `u`.
Put the result in `div`.
"""
divergence!

@kernel function divergence!(setup, div, u)
    (; D, n) = setup
    per = periodicindex(n)
    e = unitindices(D)
    I = @index(Global, Cartesian)
    dx = zero(eltype(div))
    @unroll for α = 1:getval(D)
        dx += n * (u[I, α] - u[I-e[α]|>per, α])
    end
    div[I] = dx
end

"""
Compute convection-diffusion force from velocity `u`.
Add the force field to `f`.
"""
convectiondiffusion!

@kernel function convectiondiffusion!(setup, f, u)
    (; D, n, visc) = setup
    per = periodicindex(n)
    e = unitindices(D)
    T = typeof(visc)
    dims = 1:getval(D)
    I = @index(Global, Cartesian)
    @unroll for α in dims
        fIα = f[I, α]
        @unroll for β in dims
            eα, eβ = e[α], e[β]
            uαβ1 = (u[I-eβ|>per, α] + u[I, α]) / 2
            uαβ2 = (u[I, α] + u[I+eβ|>per, α]) / 2
            uβα1 = (u[I-eβ|>per, β] + u[I-eβ+eα|>per, β]) / 2
            uβα2 = (u[I, β] + u[I+eα|>per, β]) / 2
            uαuβ1 = uαβ1 * uβα1
            uαuβ2 = uαβ2 * uβα2
            ∂βuα1 = n * (u[I, α] - u[I-eβ|>per, α])
            ∂βuα2 = n * (u[I+eβ|>per, α] - u[I, α])
            fIα += n * (visc * (∂βuα2 - ∂βuα1) - (uαuβ2 - uαuβ1))
        end
        f[I, α] = fIα
    end
end

"Create spectral Poisson solver from setup."
function poissonsolver(setup)
    (; backend, D, n, visc) = setup
    T = typeof(visc)

    # Since we use rfft, the first dimension is halved
    kmax = ntuple(d -> d == 1 ? div(n, 2) + 1 : n, D)

    # Fourier transform of the discrete Laplacian
    ahat = ntuple(D) do d
        k = 0:kmax[d]-1
        ahat = KernelAbstractions.allocate(backend, T, kmax[d])
        @. ahat = 4 * sinpi(k / n)^2 * n^2
        ahat
    end

    # Placeholders for intermediate results
    phat = KernelAbstractions.allocate(backend, Complex{T}, kmax)
    p = KernelAbstractions.allocate(backend, T, ntuple(Returns(n), D))
    plan = plan_rfft(p)

    function solver!(p)
        # Fourier transform of right hand side
        mul!(phat, plan, p)

        # Solve for coefficients in Fourier space
        if getval(D) == 2
            ax = reshape(ahat[1], :)
            ay = reshape(ahat[2], 1, :)
            @. phat = -phat / (ax + ay)
        else
            ax = reshape(ahat[1], :)
            ay = reshape(ahat[2], 1, :)
            az = reshape(ahat[3], 1, 1, :)
            @. phat = -phat / (ax + ay + az)
        end

        # Pressure is determined up to constant. We set this to 0 (instead of
        # phat[1] / 0 = Inf)
        # Note use of singleton range 1:1 instead of scalar index 1
        # (otherwise CUDA gets annoyed)
        phat[1:1] .= 0

        # Inverse Fourier transform
        ldiv!(p, plan, phat)

        p
    end
end

"Subtract pressure gradient."
pressuregradient!

@kernel function pressuregradient!(setup, u, p)
    (; D, n) = setup
    per = periodicindex(n)
    e = unitindices(D)
    I = @index(Global, Cartesian)
    @unroll for α = 1:getval(D)
        u[I, α] -= n * (p[I+e[α]|>per] - p[I])
    end
end

"Project velocity field onto divergence-free space."
function project!(u, p, poissonsolve!, setup)
    (; D) = setup

    # Divergence of tentative velocity field
    apply!(divergence!, setup, p, u)

    # Solve the Poisson equation
    poissonsolve!(p)

    # Apply pressure correction term
    apply!(pressuregradient!, setup, u, p)

    u
end

@kernel function vorticity!(setup, ω, u)
    (; D, n) = setup
    per = periodicindex(n)
    i, j = @index(Global, NTuple)
    dudy = n * (u[i, j+1|>per, 1] - u[i, j, 1])
    dvdx = n * (u[i+1|>per, j, 2] - u[i, j, 2])
    ω[i, j] = -dudy + dvdx
end
