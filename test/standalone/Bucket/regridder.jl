using ClimaLSM.Regridder: MapInfo, regrid_netcdf_to_field
using ClimaLSM.Bucket:
    BulkAlbedoMap, bareground_albedo_dataset_path, set_initial_parameter_field!
using ClimaLSM.Domains: SphericalSurface
import ClimaLSM
using ClimaComms
using ClimaCore
using Test


@testset "Spatially varying map" begin
    FT = Float32
    path = bareground_albedo_dataset_path()
    regrid_dirpath =
        joinpath(pkgdir(ClimaLSM), "test/Bucket/regridder_tmpfiles")
    mkpath(regrid_dirpath)
    rm(regrid_dirpath, recursive = true)

    varname = "sw_alb"
    comms = ClimaComms.SingletonCommsContext()
    albedo = MapInfo(path, varname, regrid_dirpath, comms)

    surface_domain =
        SphericalSurface(; radius = FT(1), nelements = 2, npolynomial = 3)
    boundary_space = surface_domain.space
    field = regrid_netcdf_to_field(
        FT,
        regrid_dirpath,
        comms,
        path,
        varname,
        boundary_space,
    )
    @test axes(field) == boundary_space


    p = (; :bucket => (; :α_sfc => ClimaCore.Fields.zeros(boundary_space)))
    set_initial_parameter_field!(
        BulkAlbedoMap{FT}(FT(0.08), albedo),
        p,
        ClimaCore.Fields.coordinate_field(boundary_space),
    )
    @test p.bucket.α_sfc == field
    rm(regrid_dirpath, recursive = true)

end
