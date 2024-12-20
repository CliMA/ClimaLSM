# Fetch site data
path = ClimaLand.Artifacts.esm_snowmip_data_path(; context = context)
metadata_path = joinpath(path, "site_metadata.txt")
site_metadata = readdlm(metadata_path, '\t', skipstart = 1)
# Get the latitude, longitude, start year, and end year for a given site from the metadata file
function get_loc_yr(metadata::Array{Any}, site::String)
    try
        i_site = findfirst(isequal(site), metadata[:, 1])

        if i_site === nothing
            throw(ErrorException("No metadata found for $site in $metadata_fp"))
        end

        lat, lon, start_yr, end_yr = metadata[i_site, 2],
        metadata[i_site, 3],
        metadata[i_site, 6],
        metadata[i_site, 7]

        return lat, lon, start_yr, end_yr
    catch e
        println(e)
        return nothing
    end
end
lat, long, start_year, end_year = get_loc_yr(site_metadata, SITE_NAME)
site_data_stem = "insitu_$(SITE_NAME)_$(start_year)_$(end_year).nc"
met_filename = join(["met_", site_data_stem])
obs_filename = join(["obs_", site_data_stem])


# Meteorological data
met_data = NCDataset(joinpath(path, met_filename))
timestamp = met_data["time"][:]
year = Dates.year.(timestamp)
# convert from milliseconds to seconds
mask = (year .== 2009) .| (year .== 2010) .| (year .== 2011)
seconds = Dates.value.(timestamp[mask] .- timestamp[mask][1]) / 1000

LWdown = met_data["LWdown"][:][mask]
SWdown = met_data["SWdown"][:][mask]
Psurf = met_data["Psurf"][:][mask]
q_air = met_data["Qair"][:][mask]
Tair = met_data["Tair"][:][mask]
u = met_data["Wind"][:][mask]
# convert mass flux into volume flux of liquid water
# additionally, our convention is that a negative flux is downwards
rainf = -met_data["Rainf"][:][mask] ./ 1000.0 # convert to dS/dt
snowf = -met_data["Snowf"][:][mask] ./ 1000.0 # convert to dS/dt

# "Radiation"
SW_d = TimeVaryingInput(seconds, SWdown; context)
LW_d = TimeVaryingInput(seconds, LWdown; context)

start_date = timestamp[mask][1]
@info start_date
function zenith_angle(
    t,
    start_date;
    latitude = FT(lat),
    longitude = FT(long),
    insol_params::Insolation.Parameters.InsolationParameters{FT} = param_set.insol_params,
) where {FT}
    # This should be time in UTC
    current_datetime = start_date + Dates.Second(round(t))

    # Orbital Data uses Float64, so we need to convert to our sim FT
    d, δ, η_UTC =
        FT.(
            Insolation.helper_instantaneous_zenith_angle(
                current_datetime,
                start_date,
                insol_params,
            )
        )

    FT(
        Insolation.instantaneous_zenith_angle(
            d,
            δ,
            η_UTC,
            longitude,
            latitude,
        )[1],
    )
end

radiation = ClimaLand.PrescribedRadiativeFluxes(
    FT,
    SW_d,
    LW_d,
    start_date;
    θs = zenith_angle,
)


"Atmos"
earth_param_set = LP.LandParameters(FT)
liquid_precip = TimeVaryingInput(seconds, rainf; context)
snow_precip = TimeVaryingInput(seconds, snowf; context)
T_atmos = TimeVaryingInput(seconds, Tair; context)
u_atmos = TimeVaryingInput(seconds, u; context)
q_atmos = TimeVaryingInput(seconds, q_air; context)
# What is this value?
h_atmos = FT(3)
P_atmos = TimeVaryingInput(seconds, Psurf; context)
atmos = PrescribedAtmosphere(
    liquid_precip,
    snow_precip,
    T_atmos,
    u_atmos,
    q_atmos,
    P_atmos,
    start_date,
    h_atmos,
    earth_param_set,
)
# Snow data
snow_data = NCDataset(joinpath(path, obs_filename))

albedo = snow_data["albs"][:][mask]
z = snow_data["snd_man"][:][mask]
mass = snow_data["snw_man"][:][mask]

snow_data_avail = .!(typeof.(mass) .<: Missing)
fill_missing(x, FT) = typeof(x) <: Missing ? FT(NaN) : x

T_snow = fill_missing.(snow_data["ts"][:][mask][snow_data_avail], FT)
ρ_snow = fill_missing.(mass[snow_data_avail] ./ z[snow_data_avail], FT)
depths = fill_missing.(z[snow_data_avail], FT)
SWE = depths .* ρ_snow ./ 1000.0
α = median(
    albedo[snow_data_avail][.!(typeof.(albedo[snow_data_avail]) .<: Missing)],
)
ρ = median(ρ_snow[.~isnan.(ρ_snow)])
