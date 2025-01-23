"Cell-centered viscosity from staggered strain tensor."
@kernel function smagorinsky_viscosity!(g::Grid, visc, S, C, Δ)
    x = @index(Global, Cartesian)
    s = if dim(g) == 2
        e1, e2 = e(g, 1), e(g, 2)
        s11 = S[x, 1, 1]
        s22 = S[x, 2, 2]
        s12 =
            (S[x, 1, 2] + S[x-e1|>g, 1, 2] + S[x-e2|>g, 1, 2] + S[x-e1-e2|>g, 1, 2]) / 4
        sqrt(2s11^2 + 2s22^2 + 4s12^2)
    else
        e1, e2, e3 = e(g, 1), e(g, 2), e(g, 3)
        s11 = S[x, 1, 1]
        s22 = S[x, 2, 2]
        s33 = S[x, 3, 3]
        s12 =
            (S[x, 1, 2] + S[x-e1|>g, 1, 2] + S[x-e2|>g, 1, 2] + S[x-e1-e2|>g, 1, 2]) / 4
        s13 =
            (S[x, 1, 3] + S[x-e1|>g, 1, 3] + S[x-e3|>g, 1, 3] + S[x-e1-e3|>g, 1, 3]) / 4
        s23 =
            (S[x, 2, 3] + S[x-e2|>g, 2, 3] + S[x-e3|>g, 2, 3] + S[x-e2-e3|>g, 2, 3]) / 4
        sqrt(2s11^2 + 2s22^2 + 2s33^2 + 4s12^2 + 4s13^2 + 4s23^2)
    end
    visc[x] = (C * Δ)^2 * s
end

"""
Divergence of `2 * visc * S`.
Interpolate visc to staggered points first.
Add result to existing force field `f`.
"""
@kernel function eddyviscosity_closure!(g::Grid, f, visc, S)
    x = @index(Global, Cartesian)
    dims = 1:dim(g)
    @unroll for i in dims
        div = f[x, i] # Add closure to existing force
        @unroll for j in dims
            ei, ej = e(g, i), e(g, j)
            nu_t = if i == j
                nu_a = visc[x]
                nu_b = visc[x+ei|>g]
                s_a = S[x, i, j]
                s_b = S[x+ei|>g, i, j]
            else
                nu_a = (visc[x] + visc[x-ej|>g] + visc[x+ei|>g] + visc[x+ei-ej|>g]) / 4
                nu_b = (visc[x] + visc[x+ei+ej|>g] + visc[x+ei|>g] + visc[x+ej|>g]) / 4
                s_b = S[x, i, j]
                s_a = S[x-ej|>g, i, j]
            end
            div += 2 * g.n * (s_b * nu_b - s_a * nu_a)
        end
        f[x, i] = div
    end
end

function smagorinsky_model(setup, C, Δ)
    S = staggered_tensorfield(setup)
    visc = scalarfield(setup)
    function closure!(force, u)
        apply!(strain!, setup, S, u)
        apply!(smagorinsky_viscosity!, setup, visc, S, C, Δ)
        apply!(eddyviscosity_closure!, setup, force, visc, S)
        force
    end
end

@kernel function clark_tensor!(grid, A, Δ)
    x = @index(Global, Cartesian)
    a = A[x]
    A[x] = Δ^2 / 12 * a * a'
end

function clark_model(setup, Δ)
    τ = collocated_tensorfield(setup)
    function closure!(force, u)
        apply!(velocitygradient_collocated!, setup, τ, u)
        apply!(clark_tensor!, setup, τ, Δ)
        apply!(tensordivergence_collocated!, setup, force, τ)
    end
end
