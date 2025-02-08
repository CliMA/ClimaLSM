"""
    @diagnostic_compute name model compute

Macro generating a function to compute a land diagnostic,
needed in the ClimaDiagnostics framework.

See the ClimaDiagnostics docmentation for more information.

For example,
@diagnostic_compute "soil_net_radiation" SoilCanopyModel p.soil.R_n

generates the function

function compute_soil_net_radiation!(out, Y, p, t, land_model::SoilCanopyModel)
    if isnothing(out)
        return copy(p.soil.R_n)
    else
        out .= p.soil.R_n
    end
end
"""
macro diagnostic_compute(name, model, compute)
    function_name = Symbol("compute_", name, "!")
    return esc(
        quote
            @with_error function $function_name(
                out,
                Y,
                p,
                t,
                land_model::$model,
            )
                if isnothing(out)
                    return copy($compute)
                else
                    out .= $compute
                end
            end
        end,
    )
end



### BucketModel ###

# variables stored in p (diagnostics variables stored in the cache)
@diagnostic_compute "aerodynamic_resistance" BucketModel p.bucket.turbulent_fluxes.r_ae
@diagnostic_compute "sw_albedo" BucketModel p.bucket.α_sfc
@diagnostic_compute "latent_heat_flux" BucketModel p.bucket.turbulent_fluxes.lhf
@diagnostic_compute "net_radiation" BucketModel p.bucket.R_n
@diagnostic_compute "sensible_heat_flux" BucketModel p.bucket.turbulent_fluxes.shf
@diagnostic_compute "surface_air_density" BucketModel p.bucket.ρ_sfc
@diagnostic_compute "specific_humidity" BucketModel p.bucket.q_sfc
@diagnostic_compute "surface_temperature" BucketModel p.bucket.T_sfc
@diagnostic_compute "vapor_flux" BucketModel p.bucket.turbulent_fluxes.vapor_flux

# variables stored in Y (prognostic or state variables)
@diagnostic_compute "snow_water_equivalent" BucketModel Y.bucket.σS
@diagnostic_compute "soil_temperature" BucketModel Y.bucket.T
@diagnostic_compute "subsurface_water_storage" BucketModel Y.bucket.W
@diagnostic_compute "surface_water_content" BucketModel Y.bucket.Ws

### Union{SoilCanopyModel, LandModel} ###

# variables stored in p (diagnostics variables stored in the cache)

## Canopy Module ##

# Canopy - Solar Induced Fluorescence
@diagnostic_compute "solar_induced_fluorescence" Union{
    SoilCanopyModel,
    LandModel,
} p.canopy.sif.SIF

# Canopy - Autotrophic respiration
function compute_autotrophic_respiration!(
        out,
        Y,
        p,
        t,
        land_model::Union{SoilCanopyModel, LandModel},
    )
    if inothing(out)
        return p.canopy.autotrophic_respiration.Ra ./ FT(83.26) # [3.664 kg CO2/ kg C] x [10^3 g CO2/ kg CO2] x [1 mol CO2/44.009 g CO2] = 83.26 mol CO2/kg C
    else
        out .= p.canopy.autotrophic_respiration.Ra ./ FT(83.26)
    end
end

# Canopy - Conductance
@diagnostic_compute "stomatal_conductance" Union{SoilCanopyModel, LandModel} p.canopy.conductance.gs
@diagnostic_compute "canopy_transpiration" Union{SoilCanopyModel, LandModel} p.canopy.energy.turbulent_fluxes.transpiration

# Canopy - Energy
@diagnostic_compute "canopy_latent_heat_flux" Union{SoilCanopyModel, LandModel} p.canopy.energy.turbulent_fluxes.lhf
@diagnostic_compute "canopy_sensible_heat_flux" Union{
    SoilCanopyModel,
    LandModel,
} p.canopy.energy.turbulent_fluxes.shf

# Canopy - Hydraulics
function compute_leaf_water_potential!(
    out,
    Y,
    p,
    t,
    land_model::Union{SoilCanopyModel, LandModel},
)
    hydraulics = land_model.canopy.hydraulics
    n_stem = hydraulics.n_stem
    n_leaf = hydraulics.n_leaf
    n = n_stem + n_leaf
    if isnothing(out)
        return p.canopy.hydraulics.ψ.:($n)
    else
        out .= p.canopy.hydraulics.ψ.:($n)
    end
end

# @diagnostic_compute "flux_per_ground_area" Union{SoilCanopyModel, LandModel} p.canopy.hydraulics.fa # return a Tuple
@diagnostic_compute "root_flux_per_ground_area" Union{
    SoilCanopyModel,
    LandModel,
} p.canopy.hydraulics.fa_roots
@diagnostic_compute "leaf_area_index" Union{SoilCanopyModel, LandModel} p.canopy.hydraulics.area_index.leaf
@diagnostic_compute "moisture_stress_factor" Union{SoilCanopyModel, LandModel} p.canopy.hydraulics.β
@diagnostic_compute "root_area_index" Union{SoilCanopyModel, LandModel} p.canopy.hydraulics.area_index.root
@diagnostic_compute "stem_area_index" Union{SoilCanopyModel, LandModel} p.canopy.hydraulics.area_index.stem

# Canopy - Photosynthesis
function compute_photosynthesis_gross_canopy!(
    out,
    Y,
    p,
    t,
    land_model::Union{SoilCanopyModel, LandModel},
)
    if isnothing(out)
        return p.canopy.photosynthesis.GPP ./ FT(83.26) # [3.664 kg CO2/ kg C] x [10^3 g CO2/ kg CO2] x [1 mol CO2/44.009 g CO2] = 83.26 mol CO2/kg C
    else
        out .= p.canopy.photosynthesis.GPP ./ FT(83.26) # [3.664 kg CO2/ kg C] x [10^3 g CO2/ kg CO2] x [1 mol CO2/44.009 g CO2] = 83.26 mol CO2/kg C
    end
end

function compute_photosynthesis_net_leaf!(
    out,
    Y,
    p,
    t,
    land_model::Union{SoilCanopyModel, LandModel},
)
    if isnothing(out)
        return p.canopy.photosynthesis.An ./ FT(83.26) # [3.664 kg CO2/ kg C] x [10^3 g CO2/ kg CO2] x [1 mol CO2/44.009 g CO2] = 83.26 mol CO2/kg C
    else
        out .= p.canopy.photosynthesis.An ./ FT(83.26) # [3.664 kg CO2/ kg C] x [10^3 g CO2/ kg CO2] x [1 mol CO2/44.009 g CO2] = 83.26 mol CO2/kg C
    end
end

function compute_respiration_leaf!(
    out,
    Y,
    p,
    t,
    land_model::Union{SoilCanopyModel, LandModel},
)
    if isnothing(out)
        return p.canopy.photosynthesis.Rd ./ FT(83.26) # [3.664 kg CO2/ kg C] x [10^3 g CO2/ kg CO2] x [1 mol CO2/44.009 g CO2] = 83.26 mol CO2/kg C
    else
        out .= p.canopy.photosynthesis.Rd ./ FT(83.26) # [3.664 kg CO2/ kg C] x [10^3 g CO2/ kg CO2] x [1 mol CO2/44.009 g CO2] = 83.26 mol CO2/kg C
    end
end
@diagnostic_compute "vcmax25" Union{SoilCanopyModel, LandModel} p.canopy.photosynthesis.Vcmax25

# Canopy - Radiative Transfer
@diagnostic_compute "near_infrared_radiation_down" Union{
    SoilCanopyModel,
    LandModel,
} p.canopy.radiative_transfer.nir_d
@diagnostic_compute "near_infrared_radiation_absorbed" Union{
    SoilCanopyModel,
    LandModel,
} p.canopy.radiative_transfer.nir.abs
@diagnostic_compute "near_infrared_radiation_reflected" Union{
    SoilCanopyModel,
    LandModel,
} p.canopy.radiative_transfer.nir.refl
@diagnostic_compute "near_infrared_radiation_transmitted" Union{
    SoilCanopyModel,
    LandModel,
} p.canopy.radiative_transfer.nir.trans
@diagnostic_compute "photosynthetically_active_radiation_down" Union{
    SoilCanopyModel,
    LandModel,
} p.canopy.radiative_transfer.par_d
@diagnostic_compute "photosynthetically_active_radiation_absorbed" Union{
    SoilCanopyModel,
    LandModel,
} p.canopy.radiative_transfer.par.abs
@diagnostic_compute "photosynthetically_active_radiation_reflected" Union{
    SoilCanopyModel,
    LandModel,
} p.canopy.radiative_transfer.par.refl
@diagnostic_compute "photosynthetically_active_radiation_transmitted" Union{
    SoilCanopyModel,
    LandModel,
} p.canopy.radiative_transfer.par.trans
@diagnostic_compute "radiation_longwave_net" Union{SoilCanopyModel, LandModel} p.canopy.radiative_transfer.LW_n
@diagnostic_compute "radiation_shortwave_net" Union{SoilCanopyModel, LandModel} p.canopy.radiative_transfer.SW_n

## Drivers Module ##

@diagnostic_compute "soil_organic_carbon" Union{SoilCanopyModel, LandModel} p.drivers.soc # need to fix this in src/shared_utilities/drivers
@diagnostic_compute "pressure" Union{SoilCanopyModel, LandModel} p.drivers.P
@diagnostic_compute "rainfall" Union{SoilCanopyModel, LandModel} p.drivers.P_liq
@diagnostic_compute "radiation_longwave_down" Union{SoilCanopyModel, LandModel} p.drivers.LW_d
@diagnostic_compute "radiation_shortwave_down" Union{SoilCanopyModel, LandModel} p.drivers.SW_d
@diagnostic_compute "snowfall" Union{SoilCanopyModel, LandModel} p.drivers.P_snow
@diagnostic_compute "solar_zenith_angle" Union{SoilCanopyModel, LandModel} p.drivers.θs
@diagnostic_compute "specific_humidity" Union{SoilCanopyModel, LandModel} p.drivers.q
@diagnostic_compute "wind_speed" Union{SoilCanopyModel, LandModel} p.drivers.u

## Soil Module ##

@diagnostic_compute "infiltration" Union{SoilCanopyModel, LandModel} p.soil.infiltration
@diagnostic_compute "soil_hydraulic_conductivity" Union{
    SoilCanopyModel,
    LandModel,
} p.soil.K
@diagnostic_compute "soil_thermal_conductivity" Union{
    SoilCanopyModel,
    LandModel,
} p.soil.κ
@diagnostic_compute "soil_water_potential" Union{SoilCanopyModel, LandModel} p.soil.ψ
@diagnostic_compute "soil_net_radiation" Union{SoilCanopyModel, LandModel} p.soil.R_n
@diagnostic_compute "soil_temperature" Union{SoilCanopyModel, LandModel} p.soil.T

function compute_soil_albedo!(
    out,
    Y,
    p,
    t,
    land_model::SoilCanopyModel{FT},
) where {FT}
    if isnothing(out)
        return (p.soil.PAR_albedo .+ p.soil.NIR_albedo) ./ 2
    else
        @. out = (p.soil.PAR_albedo + p.soil.NIR_albedo) / 2
    end
end

# Soil - Turbulent Fluxes
@diagnostic_compute "soil_latent_heat_flux" Union{SoilCanopyModel, LandModel} p.soil.turbulent_fluxes.lhf
@diagnostic_compute "soil_sensible_heat_flux" Union{SoilCanopyModel, LandModel} p.soil.turbulent_fluxes.shf
@diagnostic_compute "vapor_flux" Union{SoilCanopyModel, LandModel} p.soil.turbulent_fluxes.vapor_flux_liq # should add ice here

# Soil - SoilCO2
@diagnostic_compute "heterotrophic_respiration" Union{SoilCanopyModel, LandModel} p.soilco2.top_bc
@diagnostic_compute "soilco2_diffusivity" Union{SoilCanopyModel, LandModel} p.soilco2.D
@diagnostic_compute "soilco2_source_microbe" Union{SoilCanopyModel, LandModel} p.soilco2.Sm

## Other ##
@diagnostic_compute "sw_albedo" Union{SoilCanopyModel, LandModel} p.α_sfc
@diagnostic_compute "lw_up" Union{SoilCanopyModel, LandModel} p.LW_u
@diagnostic_compute "sw_up" Union{SoilCanopyModel, LandModel} p.SW_u
@diagnostic_compute "surface_runoff" Union{SoilCanopyModel, LandModel} p.soil.R_s

function compute_evapotranspiration!(
    out,
    Y,
    p,
    t,
    land_model::SoilCanopyModel{FT},
) where {FT}
    if isnothing(out)
        return (
            p.soil.turbulent_fluxes.vapor_flux_liq .+
            p.soil.turbulent_fluxes.vapor_flux_ice .+
            p.canopy.energy.turbulent_fluxes.transpiration
        ) .* 1000 # density of liquid water (1000kg/m^3)
    else
        out .=
            (
                p.soil.turbulent_fluxes.vapor_flux_liq .+
                p.soil.turbulent_fluxes.vapor_flux_ice .+
                p.canopy.energy.turbulent_fluxes.transpiration
            ) .* 1000 # density of liquid water (1000kg/m^3)
    end
end

function compute_evapotranspiration!(
    out,
    Y,
    p,
    t,
    land_model::LandModel{FT},
) where {FT}
    if isnothing(out)
        return @. (
            (1 - p.snow.snow_cover_fraction) *
            p.soil.turbulent_fluxes.vapor_flux_liq +
            (1 - p.snow.snow_cover_fraction) *
            p.soil.turbulent_fluxes.vapor_flux_ice +
            p.canopy.energy.turbulent_fluxes.transpiration +
            p.snow.snow_cover_fraction * p.snow.turbulent_fluxes.vapor_flux
        ) * 1000 # density of liquid water (1000kg/m^3)
    else
        @. out =
            (
                (1 - p.snow.snow_cover_fraction) *
                p.soil.turbulent_fluxes.vapor_flux_liq +
                (1 - p.snow.snow_cover_fraction) *
                p.soil.turbulent_fluxes.vapor_flux_ice +
                p.canopy.energy.turbulent_fluxes.transpiration +
                p.snow.snow_cover_fraction * p.snow.turbulent_fluxes.vapor_flux
            ) * 1000 # density of liquid water (1000kg/m^3)
    end
end

function compute_total_respiration!(
    out,
    Y,
    p,
    t,
    land_model::Union{SoilCanopyModel{FT}, LandModel{FT}},
) where {FT}
    if isnothing(out)
        return p.soilco2.top_bc .+
               p.canopy.autotrophic_respiration.Ra ./ FT(83.26)
    else
        out .=
            p.soilco2.top_bc .+ p.canopy.autotrophic_respiration.Ra ./ FT(83.26)
    end
end

function compute_latent_heat_flux!(
    out,
    Y,
    p,
    t,
    land_model::SoilCanopyModel{FT},
) where {FT}
    if isnothing(out)
        return p.soil.turbulent_fluxes.lhf .+
               p.canopy.energy.turbulent_fluxes.lhf
    else
        out .=
            p.soil.turbulent_fluxes.lhf .+ p.canopy.energy.turbulent_fluxes.lhf
    end
end

function compute_sensible_heat_flux!(
    out,
    Y,
    p,
    t,
    land_model::SoilCanopyModel{FT},
) where {FT}
    if isnothing(out)
        return p.soil.turbulent_fluxes.shf .+
               p.canopy.energy.turbulent_fluxes.shf
    else
        out .=
            p.soil.turbulent_fluxes.shf .+ p.canopy.energy.turbulent_fluxes.shf
    end
end

function compute_latent_heat_flux!(
    out,
    Y,
    p,
    t,
    land_model::LandModel{FT},
) where {FT}
    if isnothing(out)
        return @. p.soil.turbulent_fluxes.lhf *
                  (1 - p.snow.snow_cover_fraction) +
                  p.canopy.energy.turbulent_fluxes.lhf +
                  p.snow.snow_cover_fraction * p.snow.turbulent_fluxes.lhf
    else
        @. out =
            p.soil.turbulent_fluxes.lhf * (1 - p.snow.snow_cover_fraction) +
            p.canopy.energy.turbulent_fluxes.lhf +
            p.snow.snow_cover_fraction * p.snow.turbulent_fluxes.lhf
    end
end

function compute_sensible_heat_flux!(
    out,
    Y,
    p,
    t,
    land_model::LandModel{FT},
) where {FT}
    if isnothing(out)
        return @. p.soil.turbulent_fluxes.shf *
                  (1 - p.snow.snow_cover_fraction) +
                  p.canopy.energy.turbulent_fluxes.shf +
                  p.snow.snow_cover_fraction * p.snow.turbulent_fluxes.shf
    else
        @. out =
            p.soil.turbulent_fluxes.shf * (1 - p.snow.snow_cover_fraction) +
            p.canopy.energy.turbulent_fluxes.shf +
            p.snow.snow_cover_fraction * p.snow.turbulent_fluxes.shf
    end
end

function compute_net_radiation!(
    out,
    Y,
    p,
    t,
    land_model::Union{SoilCanopyModel{FT}, LandModel{FT}},
) where {FT}
    if isnothing(out)
        return p.drivers.LW_d .- p.LW_u .+ p.drivers.SW_d .- p.SW_u

    else
        out .= p.drivers.LW_d .- p.LW_u .+ p.drivers.SW_d .- p.SW_u

    end
end

function compute_ground_heat_flux!(
    out,
    Y,
    p,
    t,
    land_model::Union{SoilCanopyModel{FT}, LandModel{FT}},
) where {FT}
    if isnothing(out)
        return p.soil.turbulent_fluxes.shf .+
               p.canopy.energy.turbulent_fluxes.shf .- p.soil.R_n
    else
        out .=
            p.soil.turbulent_fluxes.shf .+
            p.canopy.energy.turbulent_fluxes.shf .- p.soil.R_n
    end
end

# variables stored in Y (prognostic or state variables)
nan_if_no_canopy(T::FT, AI::FT) where {FT <: Real} = AI > 0 ? T : FT(NaN)
function compute_canopy_temperature!(
    out,
    Y,
    p,
    t,
    land_model::Union{SoilCanopyModel{FT}, LandModel{FT}},
) where {FT}
    AI =
        p.canopy.hydraulics.area_index.leaf .+
        p.canopy.hydraulics.area_index.stem
    if isnothing(out)
        return nan_if_no_canopy.(Y.canopy.energy.T, AI)
    else
        out .= nan_if_no_canopy.(Y.canopy.energy.T, AI)
    end
end

@diagnostic_compute "soilco2" Union{SoilCanopyModel, LandModel} Y.soilco2.C
@diagnostic_compute "soil_water_content" Union{SoilCanopyModel, LandModel} Y.soil.ϑ_l
# @diagnostic_compute "plant_water_content" Union{SoilCanopyModel, LandModel} Y.canopy.hydraulics.ϑ_l # return a Tuple
@diagnostic_compute "soil_ice_content" Union{SoilCanopyModel, LandModel} Y.soil.θ_i
@diagnostic_compute "soil_internal_energy" Union{SoilCanopyModel, LandModel} Y.soil.ρe_int

@diagnostic_compute "snow_water_equivalent" LandModel Y.snow.S
@diagnostic_compute "snow_depth" LandModel p.snow.z_snow

### EnergyHydrology ###

@diagnostic_compute "soil_water_content" EnergyHydrology Y.soil.ϑ_l
@diagnostic_compute "soil_ice_content" EnergyHydrology Y.soil.θ_i
@diagnostic_compute "soil_internal_energy" EnergyHydrology Y.soil.ρe_int
@diagnostic_compute "soil_temperature" EnergyHydrology p.soil.T
