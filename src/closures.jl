@kernel function clark_tensor!(grid, G, Δ)
    x = @index(Global, Cartesian)
    g = G[x]
    G[x] = -Δ^2 / 12 * g * g'
end

"""
Clark's gradient model [clarkEvaluationSubgridscaleModels1979](@cite).
"""
function clark_model(setup, Δ)
    τ = collocated_tensorfield(setup)
    function closure!(force, u)
        apply!(velocitygradient_collocated!, setup, τ, u)
        apply!(clark_tensor!, setup, τ, Δ)
        apply!(tensordivergence_collocated!, setup, force, τ)
    end
end

"""
Divergence of `2 * visc * S`.
Interpolate `visc` to staggered points first.
Add result to existing force field `f`.
"""
@kernel function eddyviscosity_closure!(g::Grid, f, visc, ∇u)
    x = @index(Global, Cartesian)
    dims = 1:dim(g)
    @unroll for i in dims
        div = f[x, i] # Add closure to existing force
        @unroll for j in dims
            ei, ej = e(g, i), e(g, j)
            if i == j
                nu_a = visc[x]
                nu_b = visc[x+ei|>g]
                s_a = (∇u[x, i, j] + ∇u[x, j, i]) / 2
                s_b = (∇u[x+ei|>g, i, j] + ∇u[x+ei|>g, j, i]) / 2
            else
                nu_a = (visc[x] + visc[x-ej|>g] + visc[x+ei|>g] + visc[x+ei-ej|>g]) / 4
                nu_b = (visc[x] + visc[x+ei+ej|>g] + visc[x+ei|>g] + visc[x+ej|>g]) / 4
                s_a = (∇u[x-ej|>g, i, j] + ∇u[x-ej|>g, j, i]) / 2
                s_b = (∇u[x, i, j] + ∇u[x, j, i]) / 2
            end
            div += 2 * g.n * (s_b * nu_b - s_a * nu_a)
        end
        f[x, i] = div
    end
end

"Eddy viscosity closure model."
function eddyviscosity_model(viscosity!, setup, C, Δ)
    ∇u = staggered_tensorfield(setup)
    visc = scalarfield(setup)
    function closure!(force, u)
        apply!(velocitygradient!, setup, ∇u, u)
        apply!(viscosity!, setup, visc, ∇u, C, Δ)
        apply!(eddyviscosity_closure!, setup, force, visc, ∇u)
        force
    end
end

"""
Compute Smagorinsky's original eddy viscosity [smagorinskyGeneralCirculationExperiments1963](@cite).
Proposed value for `C` is 0.17.
"""
@kernel function smagorinsky_viscosity!(g::Grid, visc, ∇u, C, Δ)
    x = @index(Global, Cartesian)
    G = pol_tensor_collocated(g, ∇u, x)
    s = if dim(g) == 2
        s11, s22, s12 = G[1, 1], G[2, 2], (G[1, 2] + G[2, 1]) / 2
        sqrt(2s11^2 + 2s22^2 + 4s12^2)
    else
        s11, s22, s33 = G[1, 1], G[2, 2], G[3, 3]
        s12, s13, s23 =
            (G[1, 2] + G[2, 1]) / 2, (G[1, 3] + G[3, 1]) / 2, (G[2, 3] + G[3, 2]) / 2
        sqrt(2s11^2 + 2s22^2 + 2s33^2 + 4s12^2 + 4s13^2 + 4s23^2)
    end
    visc[x] = (C * Δ)^2 * s
end

"""
Compute Vreman's eddy viscosity [vremanEddyviscositySubgridscaleModel2004,triasBuildingProperInvariants2015](@cite).
Proposed value for `C` is 0.28.
"""
@kernel function vreman_viscosity!(g::Grid, visc, ∇u, C, Δ)
    x = @index(Global, Cartesian)
    G = pol_tensor_collocated(g, ∇u, x)
    S = (G + G') / 2
    Ω = (G - G') / 2
    QG = -tr(G * G) / 2
    QS = -tr(S * S) / 2
    QΩ = -tr(Ω * Ω) / 2
    V2 = 4 * (tr(S * S * Ω * Ω) − 2 * QS * QΩ)
    visc[x] = (C * Δ)^2 * sqrt((V2 + QG^2) / 2 / (QΩ - QS))
end

"""
Compute Verstappen's eddy viscosity [verstappenWhenDoesEddy2011,triasBuildingProperInvariants2015](@cite).
Proposed values for `C` are
0.345 [verstappenWhenDoesEddy2011](@cite) or
0.527 [triasBuildingProperInvariants2015](@cite)."
"""
@kernel function verstappen_viscosity!(g::Grid, visc, ∇u, C, Δ)
    x = @index(Global, Cartesian)
    G = pol_tensor_collocated(g, ∇u, x)
    S = (G + G') / 2
    QS = -tr(S * S) / 2
    RS = tr(S * S * S) / 3
    visc[x] = -(C * Δ)^2 * abs(RS) / QS
end

"""
Compute Nicoud's eddy viscosity (``\\sigma``-model) [nicoudUsingSingularValues2011,triasBuildingProperInvariants2015](@cite)
Proposed value for `C` is 1.35.
"""
@kernel function nicoud_viscosity!(g::Grid, visc, ∇u, C, Δ)
    x = @index(Global, Cartesian)
    G = pol_tensor_collocated(g, ∇u, x)
    # Note: λ3 ≤ λ2 ≤ λ1
    λ3, λ2, λ1 = eigvals(G * G')
    λ3 = max(λ3, 0) # Might be -eps due to numerical errors
    σ1, σ2, σ3 = sqrt(λ1), sqrt(λ2), sqrt(λ3)
    visc[x] = (C * Δ)^2 * σ3 * (σ1 - σ2) * (σ2 - σ3) / σ1^2
end

"""
Compute S3PQR eddy viscosity [triasBuildingProperInvariants2015](@cite)

Proposed values for ``p`` (set `valp = Val(p))`:

- -5/2
- -1
- 0
"""
@kernel function s3pqr_viscosity!(g::Grid, visc, ∇u, (C, valp), Δ)
    T = eltype(visc)
    p = getval(valp)
    x = @index(Global, Cartesian)
    G = pol_tensor_collocated(g, ∇u, x)
    S = (G + G') / 2
    Ω = (G - G') / 2
    QG = -tr(G * G) / 2
    QS = -tr(S * S) / 2
    QΩ = -tr(Ω * Ω) / 2
    RG = tr(G * G * G) / 3
    V2 = 4 * (tr(S * S * Ω * Ω) − 2 * QS * QΩ)
    PGG = 2 * (QΩ − QS)
    QGG = V2 + QG^2
    RGG = RG^2
    visc[x] = (C * Δ)^2 * PGG^p * QGG^-(p + 1) * RGG^((p+T(5)/2) / 3)
end
