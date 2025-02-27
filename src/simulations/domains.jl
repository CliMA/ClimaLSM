using ClimaLand

function global_domain(
    FT;
    nelements = (101, 15),
    dz_tuple = (10.0, 0.05),
    depth = 50.0,
    npolynomial = 0,
)
    radius = FT(6378.1e3)
    dz_tuple = FT.(dz_tuple)
    depth = FT(depth)
    domain = ClimaLand.Domains.SphericalShell(;
        radius = radius,
        depth = depth,
        nelements = nelements,
        npolynomial = npolynomial,
        dz_tuple = dz_tuple,
    )
    return domain
end
