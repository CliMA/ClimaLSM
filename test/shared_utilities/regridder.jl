using ClimaLand.Regridder: regrid_netcdf_to_field
using ClimaLand.FileReader: PrescribedDataStatic
using ClimaLand.Bucket:
    BulkAlbedoStatic,
    bareground_albedo_dataset_path,
    set_initial_parameter_field!
using ClimaLand.Domains: SphericalSurface
import ClimaLand
using ClimaComms
using ClimaCore
using Test

FT = Float32

# Add tests which use TempestRemap here -
# TempestRemap is not built on Windows because of NetCDF support limitations
# `regrid_netcdf_to_field` uses TR via a call to `hdwrite_regridfile_rll_to_cgll`
if !Sys.iswindows()
    @testset "Spatially varying map - regrid to field, FT = $FT" begin
        get_infile = bareground_albedo_dataset_path
        infile_path = get_infile()

        regrid_dirpath = joinpath(
            pkgdir(ClimaLand),
            "test/standalone/Bucket/regridder_tmpfiles",
        )
        mkpath(regrid_dirpath)

        surface_domain =
            SphericalSurface(; radius = FT(1), nelements = 2, npolynomial = 3)
        boundary_space = surface_domain.space.surface
        comms_ctx = ClimaComms.context(boundary_space)
        varname = "sw_alb"
        field = regrid_netcdf_to_field(
            FT,
            regrid_dirpath,
            comms_ctx,
            infile_path,
            varname,
            boundary_space,
        )
        @test axes(field) == boundary_space

        albedo =
            PrescribedDataStatic(get_infile, regrid_dirpath, varname, comms_ctx)

        p = (; :bucket => (; :α_sfc => ClimaCore.Fields.zeros(boundary_space)))
        set_initial_parameter_field!(
            BulkAlbedoStatic{FT}(FT(0.08), albedo),
            p,
            ClimaCore.Fields.coordinate_field(boundary_space),
        )
        @test p.bucket.α_sfc == field
        rm(regrid_dirpath, recursive = true)
    end
end
