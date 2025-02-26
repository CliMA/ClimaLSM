module Artifacts

import ClimaUtilities.ClimaArtifacts: @clima_artifact

import LazyArtifacts

using ArtifactWrappers

"""
    soil_ic_2008_50m_path(; context)

Return the path to the file that contains the spun-up soil and snow initial
conditions for Jan 1, 2008.

The soil domain has a depth of 50m; we have ensured that surface properties
and fluxes are spun-up, but the deep soil water may not be.
"""
function soil_ic_2008_50m_path(; context = nothing)
    dir = @clima_artifact("soil_ic_2008_50m", context)
    return joinpath(dir, "soil_ic_2008_50m.nc")
end


"""
    era5_land_forcing_data2008_path(; context, lowres=false)

Return the path to the directory that contains the ERA5 forcing data for 2008.

Optionally, you can pass the lowres=true keyword to download a lower spatial resolution version of the data.
"""
function era5_land_forcing_data2008_folder_path(;
    context = nothing,
    lowres = false,
)
    if lowres
        return @clima_artifact("era5_land_forcing_data2008_lowres", context)
    else
        return @clima_artifact("era5_land_forcing_data2008", context)
    end
end

"""
    era5_lai_forcing_data2008_path(; context)

Return the path to the directory that contains the ERA5 LAI forcing data for 2008.
"""
function era5_lai_forcing_data2008_folder_path(; context = nothing)
    return @clima_artifact("era5_land_forcing_data2008_lai", context)
end

"""
    era5_lai_forcing_data2008_folder_path(; context = nothing)

Return the path to the directory that contains the ERA5 LAI covers.
"""
function era5_lai_covers_data_folder_path(; context = nothing)
    return @clima_artifact("era5_lai_covers", context)
end

#"""
#    era5_surface_data_1979_2024_path(; context)
#
#Return the path to the folder that contains the ERA5 monthly surface data
#from 1979 to 2024.
#"""
#function era5_surface_data_1979_2024_path(; context = nothing)
#    return @clima_artifact("era5_monthly_averages_surface_single_level_1979_2024", context)
#end

"""
    modis_lai_forcing_data2008_path(; context)

Return the path to the directory that contains the MODIS LAI forcing data for
the year 2008.
"""
function modis_lai_forcing_data2008_path(; context = nothing)
    return @clima_artifact("modis_lai", context)
end

"""
    clm_data__folder_path(; context, lowres = false)

Return the path to the folder that contains the clm data. If the lowres flag is set to true,
the 0.9x1.25 version of the data is returned. Otherwise, the 0.125x0.125 version is returned.
"""
function clm_data_folder_path(; context = nothing, lowres = false)
    if lowres
        return @clima_artifact("clm_data_0.9x1.25", context)
    else
        return @clima_artifact("clm_data_0.125x0.125", context)
    end
end

"""
    modis_ci_data_folder_path(; context = nothing)

Return the path to the folder that contains the MODIS clumping index data.
"""
function modis_ci_data_folder_path(; context = nothing)
    return @clima_artifact("modis_clumping_index", context)
end

"""
    soil_params_artifact_path(; context)

Return the path to the folder that contains the soil parameters.
"""
function soil_params_artifact_folder_path(; context = nothing)
    return @clima_artifact("soil_params_Gupta2020_2022", context)
end

"""
    soil_grids_params_artifact_path(; lowres = true, context)

Return the path to the file that contains the soil texture parameters
needed for the Balland and Arp (2005) thermal conductivity model.

Returns a ~1 degree version by default (lowres = true).
"""
function soil_grids_params_artifact_path(; context = nothing, lowres = true)
    if lowres
        dir = @clima_artifact("soilgrids_lowres", context)
        file = "soil_solid_vol_fractions_soilgrids_lowres.nc"
        return joinpath(dir, file)
    else
        dir = @clima_artifact("soilgrids", context)
        file = "soil_solid_vol_fractions_soilgrids.nc"
        return joinpath(dir, file)
    end
end

"""
    experiment_fluxnet_data_path(
        site_ID;
        context = nothing,
    )

Return the path to the file that contains a year of fluxnet data
corresponding to a `site_ID` in the set (US-MOz, US-NR1, US-Ha1, US-Var).

Site publications and licenses:
US-Ha1:

Urbanski, S., Barford, C., Wofsy, S., Kucharik, C., Pyle, E., Budney, J., McKain, K., Fitzjarrald, D., Czikowsky, M., Munger, J. W. 2007. Factors Controlling CO2 Exchange On Timescales From Hourly To Decadal At Harvard Forest, Journal Of Geophysical Research, 112:G2, .

AmeriFlux FLUXNET: https://doi.org/10.17190/AMF/1871137
Citation: J. William Munger (2022), AmeriFlux FLUXNET-1F US-Ha1 Harvard Forest EMS Tower (HFR1), Ver. 3-5, AmeriFlux AMP, (Dataset). https://doi.org/10.17190/AMF/1871137

AmeriFlux CC-BY-4.0 Policy

US-MOz

Gu, L., Pallardy, S. G., Yang, B., Hosman, K. P., Mao, J., Ricciuto, D., Shi, X., Sun, Y. 2016. Testing a Land Model In Ecosystem Functional Space via a Comparison of Observed and Modeled Ecosystem Responses to Precipitation Regimes and Associated Stresses in a Central U.S. Forest, Journal Of Geophysical Research: Biogeosciences, 121:7, 1884-1902.

AmeriFlux FLUXNET: https://doi.org/10.17190/AMF/1854370
Citation: Jeffrey Wood, Lianhong Gu (2021), AmeriFlux FLUXNET-1F US-MOz Missouri Ozark Site, Ver. 3-5, AmeriFlux AMP, (Dataset). https://doi.org/10.17190/AMF/1854370

AmeriFlux CC-BY-4.0 Policy

US-NR1

Burns, S. P., Blanken, P. D., Turnipseed, A. A., Hu, J., Monson, R. K. 2015. The Influence Of Warm-Season Precipitation On The Diel Cycle Of The Surface Energy Balance And Carbon Dioxide At A Colorado Subalpine Forest Site, Biogeosciences, 12:23, 7349-7377.

AmeriFlux FLUXNET: https://doi.org/10.17190/AMF/1871141
Citation: Peter D. Blanken, Russel K. Monson, Sean P. Burns, David R. Bowling, Andrew A. Turnipseed (2022), AmeriFlux FLUXNET-1F US-NR1 Niwot Ridge Forest (LTER NWT1), Ver. 3-5, AmeriFlux AMP, (Dataset). https://doi.org/10.17190/AMF/1871141

AmeriFlux CC-BY-4.0 License

US-Var

AmeriFlux FLUXNET: https://doi.org/10.17190/AMF/1993904
Citation: Siyan Ma, Liukang Xu, Joseph Verfaillie, Dennis Baldocchi (2023), AmeriFlux FLUXNET-1F US-Var Vaira Ranch- Ione, Ver. 3-5, AmeriFlux AMP, (Dataset). https://doi.org/10.17190/AMF/1993904

AmeriFlux CC-BY-4.0 License
"""
function experiment_fluxnet_data_path(site_ID; context = nothing)
    @assert site_ID ∈ ("US-MOz", "US-Var", "US-NR1", "US-Ha1")

    folder_path = @clima_artifact("fluxnet_sites", context)
    data_path = joinpath(folder_path, "$(site_ID).csv")
    return data_path
end

"""
    esm_snowmip_data_path(; context = nothing)

Returns the path to the ESM-Snowmip data set.

Citation:
Menard, Cecile; Essery, Richard (2019): ESM-SnowMIP meteorological and evaluation datasets at ten reference sites (in situ and bias corrected reanalysis data) [dataset]. PANGAEA, https://doi.org/10.1594/PANGAEA.897575, Supplement to: Menard, Cecile; Essery, Richard; Barr, Alan; Bartlett, Paul; Derry, Jeff; Dumont, Marie; Fierz, Charles; Kim, Hyungjun; Kontu, Anna; Lejeune, Yves; Marks, Danny; Niwano, Masashi; Raleigh, Mark; Wang, Libo; Wever, Nander (2019): Meteorological and evaluation datasets for snow modelling at 10 reference sites: description of in situ and bias-corrected reanalysis data. Earth System Science Data, 11(2), 865-880, https://doi.org/10.5194/essd-11-865-2019

Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC-4.0)
"""
function esm_snowmip_data_path(; context = nothing)
    return @clima_artifact("snowmip", context)
end

"""
    richards_eqn_bonan_data_path(; context = nothing)

Returns the file path for data created solving Richards equation
with G. Bonan's matlab code, found here:
https://github.com/gbonan/bonanmodeling/tree/master/sp_08_01
This folder contains two data files: bonan_data_clay.txt and
bonan_data_sand.txt.

The data files correspond to the clay and sand data set described in
that code and in G. Bonan's book,
Climate Change and Terrestrial Ecosystem Modeling
DOI: https://doi.org/10.1017/9781107339217
Publisher: Cambridge University Press
Print publication year: 2019
"""
function richards_eqn_bonan_data_path(; context = nothing)
    return @clima_artifact("bonan_richards_eqn", context)
end

"""
    topmodel_data_path(; context = nothing)

Returns the path to the file which contains the necessary information for the TOPMODEL
runoff parameterization at 1.0 degrees resolution.

This file was created using the data provided by
Marthews, T.R., Dadson, S.J., Lehner, B., Abele, S., Gedney, N. (2015). High-resolution global topographic index values. NERC Environmental Information Data Centre. (Dataset). https://doi.org/10.5285/6b0c4358-2bf3-4924-aa8f-793d468b92be

This resource is available under the Open Government Licence (OGL), and contains data supplied by Natural Environment Research Council.

This product, High-resolution global topographic index values, has been created with use of data from the HydroSHEDS database which is © World Wildlife Fund, Inc. (2006-2013) and has been used herein under license. The HydroSHEDS database and more information are available at http://www.hydrosheds.org.
"""
function topmodel_data_path(; context = nothing)
    dir = @clima_artifact("topmodel", context)
    return joinpath(dir, "topographic_index_statistics_1.0x1.0.nc")
end

"""
    lehmann2008_evaporation_data(; context=nothing)

Returns the path to file containing measured evaporation rate as a function of time
for bare soil.

Data was originally collected by Lehmann, Peter, Shmuel Assouline,
and Dani Or. "Characteristic lengths affecting evaporative drying of
porous media." Physical Review E 77.5 (2008): 056309 and presented
in Figure 8 of that work.

https://doi.org/10.1103/PhysRevE.77.056309
"""
function lehmann2008_evaporation_data(; context = nothing)
    return joinpath(
        @clima_artifact("lehmann2008_evaporation", context),
        "lehmann2008_fig8_evaporation.csv",
    )
end

"""
    huang_et_al2011_soil_van_genuchten_data(; context=nothing)

Local path to file containing soil van Genuchten parameters as a
function of depth for soil
from site SV62 in Fort McMurray, Alberta, Canada.

Data was originally collected by Huang, Mingbin, et al.
"Infiltration and drainage processes in multi-layered coarse soils."
Canadian Journal of Soil Science 91.2 (2011): 169-183
and presented in Table 1b of that work.

https://doi.org/10.4141/cjss09118
"""
function huang_et_al2011_soil_van_genuchten_data(; context = nothing)
    dir = joinpath(@__DIR__, "../")
    af = ArtifactFile(
        url = "https://caltech.box.com/shared/static/kbgxc8r2j6uzboxgg9h2ydi5145wdpxh.csv",
        filename = "sv_62.csv",
    )
    dataset = ArtifactWrapper(dir, "sv62", ArtifactFile[af])
    dataset_path = get_data_folder(dataset)
    return joinpath(dataset_path, af.filename)
end

"""
    mizoguchi1990_soil_freezing_data(; context=nothing)

Local path to file containing soil volumetric content as a function
of depth and time during a freezing
soil column experiment.

Data was originally collected in Mizoguchi, M. 1990. Water, heat and
salt transport in freezing soil, Ph.D. thesis. (In Japanese.)
University of Tokyo, Tokyo.

Data was obtained by us from Figure 4 of Hansson, Klas, et al.
"Water flow and heat transport in frozen soil: Numerical solution
and freeze–thaw applications." Vadose Zone Journal 3.2 (2004): 693-704
using a plot digitizer; we did not quantify uncertainties introduced
in this process.
"""
function mizoguchi1990_soil_freezing_data(; context = nothing)
    dir = joinpath(@__DIR__, "../")
    af = ArtifactFile(
        url = "https://caltech.box.com/shared/static/3xbo4rlam8u390vmucc498cao6wmqlnd.csv",
        filename = "mizoguchi_all_data.csv",
    )
    dataset = ArtifactWrapper(dir, "mizoguchi", ArtifactFile[af])
    dataset_path = joinpath(get_data_folder(dataset), "mizoguchi_all_data.csv")
    return dataset_path
end

"""
    sw_albedo_dataset_folder(; context = nothing)

Triggers the download of the sw_albedo folder, if not
already downloaded, using Julia Artifacts, and returns the path to
this file.

This dataset contains surface shortwave albedo calculated from CESM2 and CERES data.
"""
function sw_albedo_dataset_folder(; context = nothing)
    return artifact_path = @clima_artifact("sw_albedo", context)
end

"""
    cesm2_albedo_dataset_path(; context = nothing)

Triggers the download of the CESM2 land albedo dataset, if not
already downloaded, using Julia Artifacts, and returns the path to
this file.

This dataset contains monthly albedo data from 15/01/1850
to 15/12/2014.
"""
function cesm2_albedo_dataset_path(; context = nothing)
    return joinpath(
        sw_albedo_dataset_folder(; context),
        "sw_albedo_Amon_CESM2_historical_r1i1p1f1_gn_185001-201412_v2_no-nans.nc",
    )
end

"""
    bareground_albedo_dataset_path(; context = nothing)

Triggers the download of the average bareground land albedo dataset, if not
already downloaded, using Julia Artifacts, and returns the path to
this file.

This dataset does not contain a time component.
"""
function bareground_albedo_dataset_path(; context = nothing)
    return joinpath(sw_albedo_dataset_folder(; context), "bareground_albedo.nc")
end

"""
    ceres_albedo_dataset_path(; context = nothing)

Triggers the download of the CERES land albedo dataset, if not
already downloaded, using Julia Artifacts, and returns the path to
this file.

This dataset contains the monthly average shortwave albedo in the
`sw_alb` variable, and the monthly average clear-sky shortwave albedo in the
`sw_alb_clr` variable. Both variables cover from 15/03/2000 to 15/10/2019.
"""
function ceres_albedo_dataset_path(; context = nothing)
    return joinpath(
        sw_albedo_dataset_folder(; context),
        "sw_albedo_Amon_CERES_EBAF_Ed4.2_Subset_200003-201910.nc",
    )
end

neural_snow_znetwork_link() =
    "https://caltech.box.com/shared/static/ay7cv0rhuiytrqbongpeq2y7m3cimhm4.bson"

"""
    ilamb_dataset_path(filename; context = nothing)

Triggers the download of the ILAMB dataset, if not already downloaded, using
Julia Artifacts, and returns the path to this file.

There are only three datasets available which are "rlus_CERESed4.2_rlus.nc",
"gpp_FLUXCOM_gpp.nc", and "evspsbl_MODIS_et_0.5x0.5.nc".
"""
function ilamb_dataset_path(filename; context = nothing)
    return joinpath(@clima_artifact("ilamb_data", context), filename)
end

"""
    earth_orography_file_path(; context=nothing)

Construct the file path for the 60arcsecond orography data NetCDF file.

Downloads the 60arc-second dataset by default.
"""
function earth_orography_file_path(; context = nothing)
    filename = "ETOPO_2022_v1_60s_N90W180_surface.nc"
    return joinpath(
        @clima_artifact("earth_orography_60arcseconds", context),
        filename,
    )
end

"""
    bedrock_depth_file_path(; context=nothing)

Construct the file path for the 60arcsecond bedrock depth data NetCDF file.

Downloads the 60arc-second dataset by default.
"""
function bedrock_depth_file_path(; context = nothing)
    filename = "ETOPO_2022_v1_60s_N90W180_bed.nc"
    return joinpath(
        @clima_artifact("bedrock_depth_60arcseconds", context),
        filename,
    )
end

"""
    era5_surface_data2008_path(; context)

Return the path to the folder that contains the ERA5 monthly surface data
"""
function era5_surface_data2008_path(; context = nothing)
    return @clima_artifact("era5_surface_fluxes", context)
end

end
