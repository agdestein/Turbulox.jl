@testitem "Commutation" begin
    x, y, z = X(), Y(), Z()
    compression = 5
    g_dns = Grid(; n = 50, L = 1.0)
    g_les = Grid(; n = 10, L = 1.0)
    @assert g_les.n * compression == g_dns.n

    # Scalar field in pressure point
    for i in (x, y, z)
        p = ScalarField(g_dns, randn(g_dns.n, g_dns.n, g_dns.n))
        xp = ScalarField(vectorposition(i), g_dns)
        sp = ScalarField(g_les)
        vxp = ScalarField(vectorposition(i), g_les)
        xsp = ScalarField(vectorposition(i), g_les)
        Turbulox.surfacefilter!(sp, p, compression, i)
        materialize!(xsp, LazyScalarField(vectorposition(i), g_les, δ, sp, i))
        materialize!(xp, LazyScalarField(vectorposition(i), g_les, δ, p, i))
        Turbulox.volumefilter!(vxp, xp, compression)
        @test xsp ≈ vxp
    end

    # Scalar field in velocity point
    for i in (x, y, z)
        p = ScalarField(vectorposition(i), g_dns, randn(g_dns.n, g_dns.n, g_dns.n))
        xp = ScalarField(g_dns)
        sp = ScalarField(vectorposition(i), g_les)
        vxp = ScalarField(g_les)
        xsp = ScalarField(g_les)
        Turbulox.surfacefilter!(sp, p, compression, i)
        materialize!(xsp, LazyScalarField(g_les, δ, sp, i))
        materialize!(xp, LazyScalarField(g_les, δ, p, i))
        Turbulox.volumefilter!(vxp, xp, compression)
        @test xsp ≈ vxp
    end

    # Scalar field in pressure point
    let
        for (i, j, k) in ((x, y, z), (x, z, y), (y, x, z), (y, z, x), (z, x, y), (z, y, x))
            p = ScalarField(g_dns, randn(g_dns.n, g_dns.n, g_dns.n))
            xp = ScalarField(vectorposition(j), g_dns)
            lp = ScalarField(g_les)
            sxp = ScalarField(vectorposition(j), g_les)
            xlp = ScalarField(vectorposition(j), g_les)
            linefilter!(lp, p, compression, k)
            materialize!(xlp, LazyScalarField(vectorposition(j), g_les, δ, lp, j))
            materialize!(xp, LazyScalarField(vectorposition(j), g_les, δ, p, j))
            surfacefilter!(sxp, xp, compression, i)
            @test xlp ≈ sxp
        end
    end

    # Tensor field
    let
        σxy = ScalarField(EdgeZ(), g_dns, randn(g_dns.n, g_dns.n, g_dns.n))
        lx_σxy = ScalarField(EdgeZ(), g_les)
        δz_lx_σxy = ScalarField(Corner(), g_les)
        δz_σxy = ScalarField(Corner(), g_dns)
        sy_δz_σxy = ScalarField(Corner(), g_les)
        linefilter!(lx_σxy, σxy, compression, x)
        materialize!(δz_lx_σxy, LazyScalarField(Corner(), g_les, δ, lx_σxy, z))
        materialize!(δz_σxy, LazyScalarField(Corner(), g_dns, δ, σxy, z))
        surfacefilter!(sy_δz_σxy, δz_σxy, compression, y)
        @test sy_δz_σxy ≈ δz_lx_σxy
    end

    # Tensor field
    let
        σxy = ScalarField(EdgeZ(), g_dns, randn(g_dns.n, g_dns.n, g_dns.n))
        lz_σxy = ScalarField(EdgeZ(), g_les)
        δx_lz_σxy = ScalarField(FaceY(), g_les)
        δx_σxy = ScalarField(FaceY(), g_dns)
        sy_δx_σxy = ScalarField(FaceY(), g_les)
        linefilter!(lz_σxy, σxy, compression, z)
        materialize!(δx_lz_σxy, LazyScalarField(FaceY(), g_les, δ, lz_σxy, x))
        materialize!(δx_σxy, LazyScalarField(FaceY(), g_dns, δ, σxy, x))
        surfacefilter!(sy_δx_σxy, δx_σxy, compression, y)
        @test sy_δx_σxy ≈ δx_lz_σxy
    end

    # Vector point to center
    let
        u = ScalarField(FaceX(), g_dns, randn(g_dns.n, g_dns.n, g_dns.n))
        lz_u = ScalarField(FaceX(), g_les)
        δx_lz_u = ScalarField(Center(), g_les)
        δx_u = ScalarField(Center(), g_dns)
        sy_δx_u = ScalarField(Center(), g_les)
        linefilter!(lz_u, u, compression, z)
        materialize!(δx_lz_u, LazyScalarField(Center(), g_les, δ, lz_u, x))
        materialize!(δx_u, LazyScalarField(Center(), g_dns, δ, u, x))
        surfacefilter!(sy_δx_u, δx_u, compression, y)
        @test sy_δx_u ≈ δx_lz_u
    end

    sy_δx_σxy.data
    δx_lx_σxy.data
end
