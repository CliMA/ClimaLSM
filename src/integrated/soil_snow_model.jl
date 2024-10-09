export LandHydrologyModel
using ClimaCore.Operators: column_integral_definite!


"""
    struct LandHydrologyModel{
        FT,
        SnM <: Snow.SnowModel{FT},
        SoM <: Soil.EnergyHydrology{FT},
    } <: AbstractLandModel{FT}
        "The snow model to be used"
        snow::SnM
        "The soil model to be used"
        soil::SoM
    end

A concrete type of land model used for simulating systems with
snow and soil (and eventually rivers).
$(DocStringExtensions.FIELDS)
"""
struct LandHydrologyModel{
    FT,
    SnM <: Snow.SnowModel{FT},
    SoM <: Soil.EnergyHydrology{FT},
} <: AbstractLandModel{FT}
    "The snow model to be used"
    snow::SnM
    "The soil model to be used"
    soil::SoM
end

"""
    LandHydrologyModel{FT}(;
        land_args::NamedTuple = (;),
        snow_model_type::Type{SnM},
        snow_args::NamedTuple = (;),
        soil_model_type::Type{SoM},
        soil_args::NamedTuple = (;),
        ) where {
            FT,
            SnM <: Snow.SnowModel{FT},
            SoM <: Soil.EnergyHydrology{FT},
            }

A constructor for the `LandHydrology`, which takes in the concrete model
type and required arguments for each component, constructs those models,
and constructs the `LandHydrologyModel` from them.

Each component model is constructed with everything it needs to be stepped
forward in time, including boundary conditions, source terms, and interaction
terms.
"""
function LandHydrologyModel{FT}(;
    land_args::NamedTuple = (;),
    snow_model_type::Type{SnM},
    snow_args::NamedTuple = (;),
    soil_model_type::Type{SoM},
    soil_args::NamedTuple = (;),
) where {FT, SnM <: Snow.SnowModel, SoM <: Soil.EnergyHydrology}
    (; atmos, radiation, domain) = land_args
    prognostic_land_components = (:snow, :soil)
    if :runoff ∈ propertynames(land_args)
        top_bc = ClimaLand.AtmosDrivenFluxBC(
            atmos,
            radiation,
            land_args.runoff,
            prognostic_land_components,
        )
    else #no runoff model
        top_bc = AtmosDrivenFluxBC(
            atmos,
            radiation,
            ClimaLand.Soil.Runoff.NoRunoff(),
            prognostic_land_components,
        )
    end
    sources = (Soil.PhaseChange{FT}(),)
    zero_flux = Soil.HeatFluxBC((p, t) -> 0.0)
    boundary_conditions = (;
        top = top_bc,
        bottom = Soil.WaterHeatBC(;
            water = Soil.FreeDrainage(),
            heat = zero_flux,
        ),
    )
    soil = soil_model_type{FT}(;
        boundary_conditions = boundary_conditions,
        sources = sources,
        domain = domain,
        soil_args...,
    )
    snow = snow_model_type(;
        boundary_conditions = Snow.AtmosDrivenSnowBC(
            atmos,
            radiation,
            prognostic_land_components,
        ),
        domain = Domains.obtain_surface_domain(domain),
        snow_args...,
    )

    return LandHydrologyModel{FT, typeof(snow), typeof(soil)}(snow, soil)
end

"""
    lsm_aux_vars(m::LandHydrologyModel)

The names of the additional auxiliary variables that are
included in the integrated Soil-Snow model.
"""
lsm_aux_vars(m::LandHydrologyModel) = (
    :excess_water_flux,
    :excess_heat_flux,
    :atmos_energy_flux,
    :atmos_water_flux,
    :ground_heat_flux,
    :effective_soil_sfc_T,
    :sfc_scratch,
    :subsfc_scratch,
    :effective_soil_sfc_depth,
)
"""
    lsm_aux_types(m::LandHydrologyModel)

The types of the additional auxiliary variables that are
included in the integrated Soil-Snow model.
"""
lsm_aux_types(m::LandHydrologyModel{FT}) where {FT} =
    (FT, FT, FT, FT, FT, FT, FT, FT, FT)

"""
    lsm_aux_domain_names(m::LandHydrologyModel)

The domain names of the additional auxiliary variables that are
included in the integrated Soil-Snow model.
"""
lsm_aux_domain_names(m::LandHydrologyModel) = (
    :surface,
    :surface,
    :surface,
    :surface,
    :surface,
    :surface,
    :surface,
    :subsurface,
    :surface,
)

"""
    make_update_boundary_fluxes(
        land::LandHydrologyModel{FT, SnM, SoM},
    ) where {
        FT,
        SnM <: Snow.SnowModel{FT},
        SoM <: Soil.EnergyHydrology{FT},
        }

A method which makes a function; the returned function
updates the additional auxiliary variables for the integrated model,
as well as updates the boundary auxiliary variables for all component
models. 

This function is called each ode function evaluation, prior to the tendency function
evaluation.

In this method, we
1. Compute the ground heat flux between soil and snow. This is required to update the snow and soil boundary fluxes
2. Update the snow boundary fluxes, which also computes any excess flux of energy or water which occurs when the snow
completely melts in a step. In this case, that excess must go to the soil for conservation
3. Update the soil boundary fluxes use precomputed ground heat flux and excess fluxes from snow.
4. Compute the net flux for the atmosphere, which is useful for assessing conservation.
"""
function make_update_boundary_fluxes(
    land::LandHydrologyModel{FT, SnM, SoM},
) where {FT, SnM <: Snow.SnowModel{FT}, SoM <: Soil.EnergyHydrology{FT}}
    update_soil_bf! = make_update_boundary_fluxes(land.soil)
    update_snow_bf! = make_update_boundary_fluxes(land.snow)
    function update_boundary_fluxes!(p, Y, t)
        earth_param_set = land.soil.parameters.earth_param_set
        # First compute the ground heat flux in place:
        update_soil_snow_ground_heat_flux!(
            p,
            Y,
            land.soil.parameters,
            land.snow.parameters,
            land.soil.domain,
            FT,
        )
        #Now update snow boundary conditions, which rely on the ground heat flux
        update_snow_bf!(p, Y, t)
        # Now we have access to the actual applied and initially computed fluxes for snow
        @. p.excess_water_flux =
            (p.snow.total_water_flux - p.snow.applied_water_flux)
        @. p.excess_heat_flux =
            (p.snow.total_energy_flux - p.snow.applied_energy_flux)
        # Now we can update the soil BC, and use the excess fluxes there in order
        # to conserve energy and water
        update_soil_bf!(p, Y, t)

        # compute net flux with atmosphere, this is useful for monitoring conservation
        _LH_f0 = FT(LP.LH_f0(earth_param_set))
        _ρ_liq = FT(LP.ρ_cloud_liq(earth_param_set))
        ρe_falling_snow = -_LH_f0 * _ρ_liq # per unit vol of liquid water
        @. p.atmos_energy_flux =
            (1 - p.snow.snow_cover_fraction) * (
                p.soil.turbulent_fluxes.lhf + p.soil.turbulent_fluxes.shf -
                p.soil.R_n
            ) +
            p.snow.snow_cover_fraction * (
                p.snow.turbulent_fluxes.lhf + p.snow.turbulent_fluxes.shf -
                p.snow.R_n
            ) +
            p.drivers.P_snow * ρe_falling_snow
        @. p.atmos_water_flux =
            p.drivers.P_snow +
            p.drivers.P_liq +
            (1 - p.snow.snow_cover_fraction) * (
                p.soil.turbulent_fluxes.vapor_flux_liq +
                p.soil.turbulent_fluxes.vapor_flux_ice
            ) +
            p.snow.snow_cover_fraction * p.snow.turbulent_fluxes.vapor_flux

    end
    return update_boundary_fluxes!
end

"""
    update_soil_snow_ground_heat_flux!(p, Y, soil_params, snow_params, soil_domain, FT)

Computes and updates `p.ground_heat_flux` with the ground heat flux. We approximate this 
as
    F_g =  - κ_eff (T_snow - T_soil)/Δz_eff,

where:
    κ_eff = κ_soil * κ_snow / (κ_snow * Δz_soil / 2 + κ_soil * Δz_snow / 2)* (Δz_soil + Δz_snow)/2
    Δz_eff =( Δz_soil + Δz_snow)/2

This is what JULES does to compute the diffusive heat flux between snow and soil, for example, 
see equation 24 and 25, with k=N, of Best et al, Geosci. Model Dev., 4, 677–699, 2011

However, this is for a multi-layer snow model, with
T_snow and Δz_snow related to the bottom layer.
It's not clear this is ideal when we have a single layer snow model. We can revisit this.
"""
function update_soil_snow_ground_heat_flux!(
    p,
    Y,
    soil_params,
    snow_params,
    soil_domain,
    FT,
)
    # Thermal conductivities
    κ_snow = p.snow.κ
    κ_soil = ClimaLand.Domains.top_center_to_surface(p.soil.κ)

    # Depth of snow and soil layers interacting thermally at interface
    Δz_snow = p.snow.z # Snow depth
    Δz_soil = p.effective_soil_sfc_depth
    (; ρc_ds, earth_param_set) = soil_params

    # The snow assumes a thickness D of soil interacts thermally with the snow
    # but this is specified by a parameter ρcD_g (volumetric heat capacity x depth).
    # Therefore we infer the depth as D = ρcD_g / ρc_g, where ρc_g is volumetric
    # heat capacity of the first soil layer
    @. p.subsfc_scratch =
        snow_params.ρcD_g /
        volumetric_heat_capacity(p.soil.θ_l, Y.soil.θ_i, ρc_ds, earth_param_set)
    Δz_soil .= ClimaLand.Domains.top_center_to_surface(p.subsfc_scratch)
    # If the soil layers are very large, no layer center may lie within Δz_soil of the snow.
    # In this case, we take the maximum of the first center layer and the thickness of the layer
    # interacting with the snow, plus a small amount
    @. Δz_soil = max(Δz_soil, soil_domain.fields.Δz_top) + sqrt(eps(FT))

    # Find average temperature of soil in depth D
    ∫H_dz = p.sfc_scratch
    ∫H_T_dz = p.soil.sfc_scratch

    H = p.subsfc_scratch
    @. H = ClimaLand.heaviside(
        soil_domain.fields.z_sfc - soil_domain.fields.z,
        Δz_soil,
    )
    column_integral_definite!(∫H_dz, H)

    H_T = p.subsfc_scratch
    @. H_T =
        ClimaLand.heaviside(
            soil_domain.fields.z_sfc - soil_domain.fields.z,
            Δz_soil,
        ) * p.soil.T
    column_integral_definite!(∫H_T_dz, H_T)

    T_soil = p.effective_soil_sfc_T

    @. T_soil = ∫H_T_dz / ∫H_dz

    # Snow temperature
    T_snow = p.snow.T

    # compute the flux
    @. p.ground_heat_flux =
        -κ_soil * κ_snow / (κ_snow * Δz_soil / 2 + κ_soil * Δz_snow / 2) *
        (T_snow - T_soil)
end


### Extensions of existing functions to account for prognostic soil/snow
"""
    snow_boundary_fluxes!(
        bc::AtmosDrivenSnowBC,
        prognostic_land_components::Val{(:snow, :soil)},
        model::SnowModel{FT},
        Y,
        p,
        t,
    ) where {FT}

A method of `snow_boundary_fluxes!` which computes 
the boundary fluxes for the snow model accounting
for a heat flux between the soil and snow.

The snow surface is 
assumed to be bare (no vegetation).

Currently this is almost identical to the method for snow alone,
except for the inclusion of the ground heat flux (precomputed by 
the integrated land model). However, this will change more if e.g.
we allow for transmission of radiation through the snowpack.
"""
function snow_boundary_fluxes!(
    bc::Snow.AtmosDrivenSnowBC,
    prognostic_land_components::Val{(:snow, :soil)},
    model::SnowModel{FT},
    Y,
    p,
    t,
) where {FT}
    p.snow.turbulent_fluxes .= turbulent_fluxes(bc.atmos, model, Y, p, t)
    p.snow.R_n .= net_radiation(bc.radiation, model, Y, p, t)
    # How does rain affect the below?
    P_snow = p.drivers.P_snow

    @. p.snow.total_water_flux =
        P_snow +
        (p.snow.turbulent_fluxes.vapor_flux - p.snow.water_runoff) *
        p.snow.snow_cover_fraction

    # I think we want dU/dt to include energy of falling snow.
    # otherwise snow can fall but energy wont change
    # We are assuming that the sensible heat portion of snow is negligible.
    _LH_f0 = FT(LP.LH_f0(model.parameters.earth_param_set))
    _ρ_liq = FT(LP.ρ_cloud_liq(model.parameters.earth_param_set))
    ρe_falling_snow = -_LH_f0 * _ρ_liq # per unit vol of liquid water

    # positive fluxes are TOWARDS atmos
    @. p.snow.total_energy_flux =
        P_snow * ρe_falling_snow +
        (
            p.snow.turbulent_fluxes.lhf + p.snow.turbulent_fluxes.shf -
            p.snow.R_n - p.snow.energy_runoff - p.ground_heat_flux
        ) * p.snow.snow_cover_fraction
end

"""
    soil_boundary_fluxes!(
        bc::AtmosDrivenFluxBC{<:PrescribedAtmosphere, <:PrescribedRadiativeFluxes},
         prognostic_land_components::Val{(:snow, :soil)},
        soil::EnergyHydrology{FT},
        Y,
        p,
        t,
    ) where {FT}

A method of `ClimaLand.Soil.soil_boundary_fluxes!` which is used for
integrated land surface models; this computes and returns the net
energy and water flux at the surface of the soil for use as boundary
conditions, taking into account the presence of snow on the surface.
"""
function soil_boundary_fluxes!(
    bc::AtmosDrivenFluxBC{<:PrescribedAtmosphere, <:PrescribedRadiativeFluxes},
    prognostic_land_components::Val{(:snow, :soil)},
    soil::EnergyHydrology{FT},
    Y,
    p,
    t,
) where {FT}
    bc = soil.boundary_conditions.top
    p.soil.turbulent_fluxes .= turbulent_fluxes(bc.atmos, soil, Y, p, t)
    p.soil.R_n .= net_radiation(bc.radiation, soil, Y, p, t)
    Soil.Runoff.update_runoff!(
        p,
        bc.runoff,
        p.drivers.P_liq .+ p.snow.water_runoff .* p.snow.snow_cover_fraction .+
        p.excess_water_flux,
        Y,
        t,
        soil,
    )
    @. p.soil.top_bc.water =
        p.soil.infiltration +
        (1 - p.snow.snow_cover_fraction) *
        p.soil.turbulent_fluxes.vapor_flux_liq

    @. p.soil.top_bc.heat =
        (1 - p.snow.snow_cover_fraction) * (
            -p.soil.R_n +
            p.soil.turbulent_fluxes.lhf +
            p.soil.turbulent_fluxes.shf
        ) +
        p.excess_heat_flux +
        p.snow.snow_cover_fraction * p.ground_heat_flux
end

function ClimaLand.Soil.sublimation_source(::Val{(:snow, :soil)}, FT)
    return SoilSublimationwithSnow{FT}()
end

"""
    SoilSublimationwithSnow{FT} <: AbstractSoilSource{FT}

Soil Sublimation source type. Used to defined a method
of `ClimaLand.source!` for soil sublimation with snow present.
"""
struct SoilSublimationwithSnow{FT} <: ClimaLand.Soil.AbstractSoilSource{FT} end

"""
     source!(dY::ClimaCore.Fields.FieldVector,
             src::SoilSublimationwithSnow{FT},
             Y::ClimaCore.Fields.FieldVector,
             p::NamedTuple,
             model
             )

Updates dY.soil.θ_i in place with a term due to sublimation; this only affects
the surface layer of soil.

"""
function ClimaLand.source!(
    dY::ClimaCore.Fields.FieldVector,
    src::SoilSublimationwithSnow{FT},
    Y::ClimaCore.Fields.FieldVector,
    p::NamedTuple,
    model,
) where {FT}
    _ρ_i = FT(LP.ρ_cloud_ice(model.parameters.earth_param_set))
    _ρ_l = FT(LP.ρ_cloud_liq(model.parameters.earth_param_set))
    z = model.domain.fields.z
    Δz_top = model.domain.fields.Δz_top # this returns the center-face distance, not layer thickness
    @. dY.soil.θ_i +=
        -p.soil.turbulent_fluxes.vapor_flux_ice *
        (1 - p.snow.snow_cover_fraction) *
        _ρ_l / _ρ_i * heaviside(z + 2 * Δz_top) # only apply to top layer, recall that z is negative
end

function ClimaLand.get_drivers(model::LandHydrologyModel)
    return (
        model.snow.boundary_conditions.atmos,
        model.snow.boundary_conditions.radiation,
    )
end
