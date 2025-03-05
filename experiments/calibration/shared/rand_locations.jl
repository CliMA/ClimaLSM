using NCDatasets
import EnsembleKalmanProcesses as EKP
using ClimaUtilities.ClimaArtifacts
import ClimaLand
using StaticArrays

# Returns a 2 SVectors of (lat, lon) tuples for n random locations on the land
# surface. The two sets of locations are designed to be used as a training and
# validation set.
function rand_locations(surface_space, regridder_type, n = 100)
    # Load the landsea mask data
    datapath = ClimaLand.Artifacts.topmodel_data_path()
    landsea_mask = SpaceVaryingInput(
        datapath,
        "landsea_mask",
        surface_space;
        regridder_type,
    )

    # Get the corresponding latitude and longitude values
    lat = ClimaCore.Fields.coordinate_field(surface_space).lat
    lon = ClimaCore.Fields.coordinate_field(surface_space).long

    # Find the coordinates of 2n random land locations
    land_inds = rand(findall(x -> x == 1.0, Array(parent(landsea_mask))), 2 * n)

    # Since this is run very rarely (once at start of calibration run), we don't
    # mind scalar iteration when running on the GPU
    CUDA.@allowscalar land_locs = StaticArrays.sacollect(
        SVector{2 * n, Tuple{FT, FT}},
        zip(parent(lon)[land_inds], parent(lat)[land_inds]),
    )

    # Return a uniform random sample of n land locations
    return (
        SVector{n}(land_locs[1:n]...),
        SVector{n}(land_locs[(n + 1):end]...),
    )
end
