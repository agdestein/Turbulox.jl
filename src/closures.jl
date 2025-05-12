@inline function clark_tensor(Δ, u, I)
    G = ∇_collocated(u, I)
    Δ^2 / 12 * G * G'
end

"""
Clark's gradient model [clarkEvaluationSubgridscaleModels1979](@cite).
"""
function clark_model!(force, Δ, u)
    (; grid) = force
    τ = LazyScalarField(grid, clark_tensor, Δ, u)
    apply!(tensorapply_add!, grid, tensordivergence_collocated, force, τ)
end

"""
Divergence of `-2 * visc * S`.
Interpolate `visc` to staggered points first.
Subtract result from existing force field `f`.
"""
@inline function eddyviscosity_closure(visc, ∇u, i, j, x)
    ei, ej = e(i), e(j)
    if i == j
        nu_a = visc[x]
        nu_b = visc[x+ei]
        s_a = (∇u[i, j][x] + ∇u[j, i][x]) / 2
        s_b = (∇u[i, j][x+ei] + ∇u[j, i][x+ei]) / 2
    else
        nu_a = (visc[x] + visc[x-ej] + visc[x+ei] + visc[x+ei-ej]) / 4
        nu_b = (visc[x] + visc[x+ei+ej] + visc[x+ei] + visc[x+ej]) / 4
        s_a = (∇u[i, j][x-ej] + ∇u[j, i][x-ej]) / 2
        s_b = (∇u[i, j][x] + ∇u[j, i][x]) / 2
    end
    2 * (s_b * nu_b - s_a * nu_a) / dx(visc.grid)
end

@kernel function eddyviscosity_tensor!(τ, visc, ∇u)
    x = @index(Global, Cartesian)
    G = ∇u[x]
    S = (G + G') / 2
    τ[x] = -2 * visc[x] * S
end

"Eddy viscosity closure model."
function eddyviscosity_model!(viscosity!, force, visc, ∇u, u, C, Δ)
    (; grid) = force
    apply!(velocitygradient!, grid, ∇u, u)
    apply!(viscosity!, grid, visc, ∇u, C, Δ)
    apply!(tensorapply_add!, grid, eddyviscosity_closure, force, visc, ∇u)
    force
end

function strainnorm(G)
    s11, s22, s33 = G[1, 1], G[2, 2], G[3, 3]
    s12, s13, s23 =
        (G[1, 2] + G[2, 1]) / 2, (G[1, 3] + G[3, 1]) / 2, (G[2, 3] + G[3, 2]) / 2
    sqrt(2s11^2 + 2s22^2 + 2s33^2 + 4s12^2 + 4s13^2 + 4s23^2)
end

"""
Compute Smagorinsky's original eddy viscosity [smagorinskyGeneralCirculationExperiments1963](@cite).
Proposed value for `C` is 0.17.
"""
@kernel function smagorinsky_viscosity!(visc, ∇u, C, Δ)
    x = @index(Global, Cartesian)
    G = pol_tensor_collocated(∇u, x)
    s = strainnorm(G)
    visc[x] = (C * Δ)^2 * s
end

"""
Compute WALE [nicoudSubgridScaleStressModelling1999,triasBuildingProperInvariants2015](@cite).
Proposed value for `C` is 0.569.
"""
@kernel function wale_viscosity!(visc, ∇u, C, Δ)
    T = eltype(visc)
    x = @index(Global, Cartesian)
    G = pol_tensor_collocated(∇u, x)
    S = (G + G') / 2
    Ω = (G - G') / 2
    QG = -tr(G * G) / 2
    QS = -tr(S * S) / 2
    QΩ = -tr(Ω * Ω) / 2
    V2 = 4 * (tr(S * S * Ω * Ω) − 2 * QS * QΩ)
    visc[x] =
        (C * Δ)^2 * (V2 / 2 + 2 * QG^2 / 3)^T(3 / 2) /
        ((-2 * QS)^T(5 / 2) + (V2 / 2 + 2 * QG^2 / 3)^T(5 / 4))
end

"""
Compute Vreman's eddy viscosity [vremanEddyviscositySubgridscaleModel2004,triasBuildingProperInvariants2015](@cite).
Proposed value for `C` is 0.28.
"""
@kernel function vreman_viscosity!(visc, ∇u, C, Δ)
    x = @index(Global, Cartesian)
    G = pol_tensor_collocated(∇u, x)
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
@kernel function verstappen_viscosity!(visc, ∇u, C, Δ)
    x = @index(Global, Cartesian)
    G = pol_tensor_collocated(∇u, x)
    S = (G + G') / 2
    QS = -tr(S * S) / 2
    RS = tr(S * S * S) / 3
    visc[x] = -(C * Δ)^2 * abs(RS) / QS
end

"""
Compute Nicoud's eddy viscosity (``\\sigma``-model) [nicoudUsingSingularValues2011,triasBuildingProperInvariants2015](@cite)
Proposed value for `C` is 1.35.
"""
@kernel function nicoud_viscosity!(visc, ∇u, C, Δ)
    x = @index(Global, Cartesian)
    G = pol_tensor_collocated(∇u, x)
    # Note: λ3 ≤ λ2 ≤ λ1
    λ3, λ2, λ1 = G * G' |> Symmetric |> eigvals
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
@kernel function s3pqr_viscosity!(visc, ∇u, (C, valp), Δ)
    T = eltype(visc)
    p = getval(valp)
    x = @index(Global, Cartesian)
    G = pol_tensor_collocated(∇u, x)
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
    visc[x] = (C * Δ)^2 * PGG^p * QGG^-(p + 1) * RGG^((p + T(5) / 2) / 3)
end
