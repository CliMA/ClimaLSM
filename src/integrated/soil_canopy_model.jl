export SoilCanopyModel
"""
    struct SoilCanopyModel{
        FT,
        MM <: Soil.Biogeochemistry.SoilCO2Model{FT},
        SM <: Soil.EnergyHydrology{FT},
        VM <: Canopy.CanopyModel{FT},
    } <: AbstractLandModel{FT}
        "The soil microbe model to be used"
        soilco2::MM
        "The soil model to be used"
        soil::SM
        "The canopy model to be used"
        canopy::VM
    end

A concrete type of land model used for simulating systems with a
canopy and a soil component.
$(DocStringExtensions.FIELDS)
"""
struct SoilCanopyModel{
    FT,
    MM <: Soil.Biogeochemistry.SoilCO2Model{FT},
    SM <: Soil.EnergyHydrology{FT},
    VM <: Canopy.CanopyModel{FT},
} <: AbstractLandModel{FT}
    "The soil microbe model to be used"
    soilco2::MM
    "The soil model to be used"
    soil::SM
    "The canopy model to be used"
    canopy::VM
end



"""
    SoilCanopyModel{FT}(;
        soilco2_type::Type{MM},
        soilco2_args::NamedTuple = (;),
        land_args::NamedTuple = (;),
        soil_model_type::Type{SM},
        soil_args::NamedTuple = (;),
        canopy_component_types::NamedTuple = (;),
        canopy_component_args::NamedTuple = (;),
        canopy_model_args::NamedTuple = (;),
        ) where {
            FT,
            SM <: Soil.EnergyHydrology{FT},
            MM <: Soil.Biogeochemistry.SoilCO2Model{FT},
            }

A constructor for the `SoilCanopyModel`, which takes in the concrete model
type and required arguments for each component, constructs those models,
and constructs the `SoilCanopyModel` from them.

Each component model is constructed with everything it needs to be stepped
forward in time, including boundary conditions, source terms, and interaction
terms.
"""
function SoilCanopyModel{FT}(;
    soilco2_type::Type{MM},
    soilco2_args::NamedTuple = (;),
    land_args::NamedTuple = (;),
    soil_model_type::Type{SM},
    soil_args::NamedTuple = (;),
    canopy_component_types::NamedTuple = (;),
    canopy_component_args::NamedTuple = (;),
    canopy_model_args::NamedTuple = (;),
) where {
    FT,
    SM <: Soil.EnergyHydrology{FT},
    MM <: Soil.Biogeochemistry.SoilCO2Model{FT},
}

    (; atmos, radiation, soil_organic_carbon) = land_args
    # These should always be set by the constructor.
    sources = (RootExtraction{FT}(), Soil.PhaseChange{FT}())
    prognostic_land_components = (:canopy, :soil, :soilco2)
    if :runoff ∈ propertynames(land_args)
        top_bc = ClimaLand.AtmosDrivenFluxBC(
            atmos,
            radiation,
            land_args.runoff,
            prognostic_land_components,
        )
    else #no runoff model
        top_bc = ClimaLand.AtmosDrivenFluxBC(
            atmos,
            radiation,
            ClimaLand.Soil.Runoff.NoRunoff(),
            prognostic_land_components,
        )
    end
    zero_flux = Soil.HeatFluxBC((p, t) -> 0.0)
    boundary_conditions = (;
        top = top_bc,
        bottom = Soil.WaterHeatBC(;
            water = Soil.FreeDrainage(),
            heat = zero_flux,
        ),
    )
    soil = soil_model_type(;
        boundary_conditions = boundary_conditions,
        sources = sources,
        soil_args...,
    )

    transpiration = Canopy.PlantHydraulics.DiagnosticTranspiration{FT}()
    canopy_soil_driver = PrognosticSoil{typeof(soil.parameters.PAR_albedo)}(
        soil.parameters.PAR_albedo,
        soil.parameters.NIR_albedo,
    )
    if :energy in propertynames(canopy_component_args)

        canopy = Canopy.CanopyModel{FT}(;
            autotrophic_respiration = canopy_component_types.autotrophic_respiration(
                canopy_component_args.autotrophic_respiration...,
            ),
            radiative_transfer = canopy_component_types.radiative_transfer(
                canopy_component_args.radiative_transfer...,
            ),
            photosynthesis = canopy_component_types.photosynthesis(
                canopy_component_args.photosynthesis...,
            ),
            conductance = canopy_component_types.conductance(
                canopy_component_args.conductance...,
            ),
            hydraulics = canopy_component_types.hydraulics(;
                transpiration = transpiration,
                canopy_component_args.hydraulics...,
            ),
            energy = canopy_component_types.energy(
                canopy_component_args.energy.parameters,
            ),
            soil_driver = canopy_soil_driver,
            atmos = atmos,
            radiation = radiation,
            canopy_model_args...,
        )
    else
        canopy = Canopy.CanopyModel{FT}(;
            autotrophic_respiration = canopy_component_types.autotrophic_respiration(
                canopy_component_args.autotrophic_respiration...,
            ),
            radiative_transfer = canopy_component_types.radiative_transfer(
                canopy_component_args.radiative_transfer...,
            ),
            photosynthesis = canopy_component_types.photosynthesis(
                canopy_component_args.photosynthesis...,
            ),
            conductance = canopy_component_types.conductance(
                canopy_component_args.conductance...,
            ),
            hydraulics = canopy_component_types.hydraulics(;
                transpiration = transpiration,
                canopy_component_args.hydraulics...,
            ),
            soil_driver = canopy_soil_driver,
            atmos = atmos,
            radiation = radiation,
            canopy_model_args...,
        )
    end

    co2_prognostic_soil = Soil.Biogeochemistry.PrognosticMet(soil.parameters)
    soilco2_drivers = Soil.Biogeochemistry.SoilDrivers(
        co2_prognostic_soil,
        soil_organic_carbon,
        atmos,
    )
    soilco2 = soilco2_type(; soilco2_args..., drivers = soilco2_drivers)

    return SoilCanopyModel{FT, typeof(soilco2), typeof(soil), typeof(canopy)}(
        soilco2,
        soil,
        canopy,
    )
end

"""
    lsm_aux_vars(m::SoilCanopyModel)

The names of the additional auxiliary variables that are
included in the integrated Soil-Canopy model.
"""
lsm_aux_vars(m::SoilCanopyModel) = (
    :root_extraction,
    :root_energy_extraction,
    :T_ground,
    :LW_out,
    :SW_out,
    :scratch1,
    :scratch2,
)

"""
    lsm_aux_types(m::SoilCanopyModel)

The types of the additional auxiliary variables that are
included in the integrated Soil-Canopy model.
"""
lsm_aux_types(m::SoilCanopyModel{FT}) where {FT} = (FT, FT, FT, FT, FT, FT, FT)

"""
    lsm_aux_domain_names(m::SoilCanopyModel)

The domain names of the additional auxiliary variables that are
included in the integrated Soil-Canopy model.
"""
lsm_aux_domain_names(m::SoilCanopyModel) =
    (:subsurface, :subsurface, :surface, :surface, :surface, :surface, :surface)

"""
    make_update_boundary_fluxes(
        land::SoilCanopyModel{FT, MM, SM, RM},
    ) where {
        FT,
        MM <: Soil.Biogeochemistry.SoilCO2Model{FT},
        SM <: Soil.RichardsModel{FT},
        RM <: Canopy.CanopyModel{FT}
        }

A method which makes a function; the returned function
updates the additional auxiliary variables for the integrated model,
as well as updates the boundary auxiliary variables for all component
models.

This function is called each ode function evaluation, prior to the tendency function
evaluation.
"""
function make_update_boundary_fluxes(
    land::SoilCanopyModel{FT, MM, SM, RM},
) where {
    FT,
    MM <: Soil.Biogeochemistry.SoilCO2Model{FT},
    SM <: Soil.EnergyHydrology{FT},
    RM <: Canopy.CanopyModel{FT},
}
    update_soil_bf! = make_update_boundary_fluxes(land.soil)
    update_soilco2_bf! = make_update_boundary_fluxes(land.soilco2)
    update_canopy_bf! = make_update_boundary_fluxes(land.canopy)
    function update_boundary_fluxes!(p, Y, t)
        # update root extraction
        z =
            ClimaCore.Fields.coordinate_field(
                land.soil.domain.space.subsurface,
            ).z
        (; conductivity_model) = land.canopy.hydraulics.parameters

        area_index = p.canopy.hydraulics.area_index

        above_ground_area_index = getproperty(
            area_index,
            land.canopy.hydraulics.compartment_labels[1],
        )
        # Note that we model the flux between each soil layer and the canopy as:
        # Flux = -K_eff x [(ψ_canopy - ψ_soil)/(z_canopy - z_soil) + 1], where
        # K_eff = K_soil K_canopy /(K_canopy + K_soil)

        # Note that in `PrescribedSoil` mode, we compute the flux using K_soil = K_plant(ψ_soil)
        # and K_canopy = K_plant(ψ_canopy). In `PrognosticSoil` mode here, we compute the flux using
        # K_soil = K_soil(ψ_soil) and K_canopy = K_plant(ψ_canopy).
        @. p.root_extraction =
            above_ground_area_index *
            PlantHydraulics.water_flux(
                z,
                land.canopy.hydraulics.compartment_midpoints[1],
                p.soil.ψ,
                p.canopy.hydraulics.ψ.:1,
                p.soil.K,
                PlantHydraulics.hydraulic_conductivity(
                    conductivity_model,
                    p.canopy.hydraulics.ψ.:1,
                ),
            ) *
            (root_distribution(
                z,
                land.canopy.hydraulics.parameters.rooting_depth,
            ))
        @. p.root_energy_extraction =
            p.root_extraction * ClimaLand.Soil.volumetric_internal_energy_liq(
                p.soil.T,
                land.soil.parameters.earth_param_set,
            )

        # Radiation
        lsm_radiant_energy_fluxes!(
            p,
            land.canopy.radiative_transfer,
            land.canopy,
            land.soil,
            Y,
            t,
        )
        update_soil_bf!(p, Y, t)
        update_canopy_bf!(p, Y, t)
        update_soilco2_bf!(p, Y, t)
    end
    return update_boundary_fluxes!
end

"""
    lsm_radiant_energy_fluxes!(p,
                                canopy_radiation::Canopy.AbstractRadiationModel{FT},
                                canopy,
                                ground_model::Soil.EnergyHydrology,
                                Y,
                                t,
                                ) where {FT}


A function which computes the net radiation at the ground surface
give the canopy radiation model, as well as the outgoing radiation,
and the net canopy radiation.

Returns the correct radiative fluxes for bare ground in the case
where the canopy LAI is zero. Note also that this serves the role of
`canopy_radiant_energy_fluxes!`, which computes the net canopy radiation
when the Canopy is run in standalone mode.
"""
function lsm_radiant_energy_fluxes!(
    p,
    canopy_radiation::Canopy.AbstractRadiationModel{FT},
    canopy,
    ground_model::Soil.EnergyHydrology,
    Y,
    t,
) where {(FT)}
    radiation = canopy.radiation
    earth_param_set = canopy.parameters.earth_param_set
    _σ = LP.Stefan(earth_param_set)
    LW_d = p.drivers.LW_d
    SW_d = p.drivers.SW_d
    c = LP.light_speed(earth_param_set)
    h = LP.planck_constant(earth_param_set)
    N_a = LP.avogadro_constant(earth_param_set)
    (; λ_γ_PAR, λ_γ_NIR) = canopy_radiation.parameters
    energy_per_photon_PAR = h * c / λ_γ_PAR
    energy_per_photon_NIR = h * c / λ_γ_NIR
    T_canopy =
        ClimaLand.Canopy.canopy_temperature(canopy.energy, canopy, Y, p, t)
    p.T_ground .= surface_temperature(ground_model, Y, p, t)

    α_soil_PAR = Canopy.ground_albedo_PAR(canopy.soil_driver, Y, p, t)
    α_soil_NIR = Canopy.ground_albedo_NIR(canopy.soil_driver, Y, p, t)
    ϵ_soil = ground_model.parameters.emissivity

    # in W/m^2
    LW_d_canopy = p.scratch1
    LW_u_soil = p.scratch2
    LW_net_canopy = p.canopy.radiative_transfer.LW_n
    SW_net_canopy = p.canopy.radiative_transfer.SW_n
    R_net_soil = p.soil.R_n
    LW_out = p.LW_out
    SW_out = p.SW_out
    # in total: INC - OUT = CANOPY_ABS + (1-α_soil)*CANOPY_TRANS
    # SW out  = reflected par + reflected nir
    @. SW_out =
        energy_per_photon_NIR * N_a * p.canopy.radiative_transfer.nir.refl +
        energy_per_photon_PAR * N_a * p.canopy.radiative_transfer.par.refl

    # net canopy
    @. SW_net_canopy =
        energy_per_photon_NIR * N_a * p.canopy.radiative_transfer.nir.abs +
        energy_per_photon_PAR * N_a * p.canopy.radiative_transfer.par.abs


    # net soil = (1-α)*trans for par and nir
    @. R_net_soil .=
        energy_per_photon_NIR *
        N_a *
        p.canopy.radiative_transfer.nir.trans *
        (1 - α_soil_NIR) +
        energy_per_photon_PAR *
        N_a *
        p.canopy.radiative_transfer.par.trans *
        (1 - α_soil_PAR)
    ϵ_canopy = p.canopy.radiative_transfer.ϵ # this takes into account LAI/SAI
    @. LW_d_canopy = ((1 - ϵ_canopy) * LW_d + ϵ_canopy * _σ * T_canopy^4) # double checked
    @. LW_u_soil = ϵ_soil * _σ * p.T_ground^4 + (1 - ϵ_soil) * LW_d_canopy # double checked
    @. R_net_soil += ϵ_soil * LW_d_canopy - ϵ_soil * _σ * p.T_ground^4 # double checked
    @. LW_net_canopy =
        ϵ_canopy * LW_d - 2 * ϵ_canopy * _σ * T_canopy^4 + ϵ_canopy * LW_u_soil
    @. LW_out = (1 - ϵ_canopy) * LW_u_soil + ϵ_canopy * _σ * T_canopy^4 # double checked
end


### Extensions of existing functions to account for prognostic soil/canopy
"""
    soil_boundary_fluxes!(
        bc::AtmosDrivenFluxBC{<:PrescribedAtmosphere, <:PrescribedRadiativeFluxes},
        prognostic_land_components::Val{(:canopy, :soil,:soilco2,)},
        soil::EnergyHydrology{FT},
        Y,
        p,
        t,
    ) where {FT}

A method of `ClimaLand.Soil.soil_boundary_fluxes!` which is used for
integrated land surface models; this computes and returns the net
energy and water flux at the surface of the soil for use as boundary
conditions when a canopy and Soil CO2  model is also included, though only
the presence of the canopy modifies the soil BC.
"""
function soil_boundary_fluxes!(
    bc::AtmosDrivenFluxBC{<:PrescribedAtmosphere, <:PrescribedRadiativeFluxes},
    prognostic_land_components::Val{(:canopy, :soil, :soilco2)},
    soil::EnergyHydrology{FT},
    Y,
    p,
    t,
) where {FT}
    bc = soil.boundary_conditions.top
    p.soil.turbulent_fluxes .= turbulent_fluxes(bc.atmos, soil, Y, p, t)
    Soil.Runoff.update_runoff!(p, bc.runoff, p.drivers.P_liq, Y, t, soil)
    @. p.soil.top_bc.water =
        p.soil.infiltration + p.soil.turbulent_fluxes.vapor_flux_liq
    @. p.soil.top_bc.heat =
        -p.soil.R_n + p.soil.turbulent_fluxes.lhf + p.soil.turbulent_fluxes.shf
end


"""
     PrognosticSoil{FT} <: AbstractSoilDriver

Concrete type of AbstractSoilDriver used for dispatch in cases where both
a canopy model and soil model are run.
$(DocStringExtensions.FIELDS)
"""
struct PrognosticSoil{F <: Union{AbstractFloat, ClimaCore.Fields.Field}} <:
       AbstractSoilDriver
    "Soil albedo for PAR"
    α_PAR::F
    "Soil albedo for NIR"
    α_NIR::F
end

function Canopy.ground_albedo_PAR(soil_driver::PrognosticSoil, Y, p, t)
    return soil_driver.α_PAR
end

function Canopy.ground_albedo_NIR(soil_driver::PrognosticSoil, Y, p, t)
    return soil_driver.α_NIR
end


"""
    PlantHydraulics.root_water_flux_per_ground_area!(
        fa::ClimaCore.Fields.Field,
        s::PrognosticSoil,
        model::Canopy.PlantHydraulics.PlantHydraulicsModel{FT},
        Y::ClimaCore.Fields.FieldVector,
        p::NamedTuple,
        t,
    )

An extension of the `PlantHydraulics.root_water_flux_per_ground_area!` function,
 which returns the
net flux of water between the
roots and the soil, per unit ground area,
when both soil and plant
hydraulics are modeled prognostically. This is for use in an LSM.

It is computed by summing the flux of water per ground area between
roots and soil at each soil layer.
"""
function PlantHydraulics.root_water_flux_per_ground_area!(
    fa::ClimaCore.Fields.Field,
    s::PrognosticSoil,
    model::Canopy.PlantHydraulics.PlantHydraulicsModel{FT},
    Y::ClimaCore.Fields.FieldVector,
    p::NamedTuple,
    t,
) where {FT}
    ClimaCore.Operators.column_integral_definite!(fa, p.root_extraction)
end


"""
    root_energy_flux_per_ground_area!(
        fa_energy::ClimaCore.Fields.Field,
        s::PrognosticSoil{F},
        model::Canopy.AbstractCanopyEnergyModel{FT},
        Y::ClimaCore.Fields.FieldVector,
        p::NamedTuple,
        t,
    ) where {FT, F}


A method computing the energy flux associated with the root-soil
water flux, which returns 0 in cases where we do not need to track
this quantity: in this case, when the canopy energy is tracked,
but we are using a `PrescribedSoil` model (non-prognostic soil model).

Note that this energy flux is not typically included in land surface
models. We account for it when the soil model is prognostic because
the soil model includes the energy in the soil water in its energy
balance; therefore, in order to conserve energy, the canopy model
must account for it as well.
"""
function Canopy.root_energy_flux_per_ground_area!(
    fa_energy::ClimaCore.Fields.Field,
    s::PrognosticSoil{F},
    model::Canopy.AbstractCanopyEnergyModel{FT},
    Y::ClimaCore.Fields.FieldVector,
    p::NamedTuple,
    t,
) where {FT, F}
    ClimaCore.Operators.column_integral_definite!(
        fa_energy,
        p.root_energy_extraction,
    )
end

"""
    RootExtraction{FT} <: Soil.AbstractSoilSource{FT}

Concrete type of Soil.AbstractSoilSource, used for dispatch
in an LSM with both soil and plant hydraulic components.

This is paired with the source term `Canopy.PrognosticSoil`:both
are used at the same time,
ensuring that the water flux into the roots is extracted correctly
from the soil.
"""
struct RootExtraction{FT} <: Soil.AbstractSoilSource{FT} end


"""
    ClimaLand.source!(dY::ClimaCore.Fields.FieldVector,
                     src::RootExtraction,
                     Y::ClimaCore.Fields.FieldVector,
                     p::NamedTuple
                     model::EnergyHydrology)

An extension of the `ClimaLand.source!` function,
 which computes source terms for the
soil model; this method returns the water and energy loss/gain due
to root extraction.
"""
function ClimaLand.source!(
    dY::ClimaCore.Fields.FieldVector,
    src::RootExtraction,
    Y::ClimaCore.Fields.FieldVector,
    p::NamedTuple,
    model::EnergyHydrology,
)
    @. dY.soil.ϑ_l += -1 * p.root_extraction
    @. dY.soil.ρe_int += -1 * p.root_energy_extraction
    # if flow is negative, towards soil -> soil water increases, add in sign here.
end

"""
    Canopy.canopy_radiant_energy_fluxes!(p::NamedTuple,
                                         s::PrognosticSoil{F},
                                         canopy,
                                         radiation::PrescribedRadiativeFluxes,
                                         earth_param_set::PSE,
                                         Y::ClimaCore.Fields.FieldVector,
                                         t,
                                        ) where {FT, PSE}

In standalone mode, this function computes and stores the net
long and short wave radition, in W/m^2,
absorbed by the canopy.

In integrated mode, we have already computed those quantities in
`lsm_radiant_energy_fluxes!`, so this method does nothing additional.

LW and SW net radiation are stored in `p.canopy.radiative_transfer.LW_n`
and `p.canopy.radiative_transfer.SW_n`.
"""
function Canopy.canopy_radiant_energy_fluxes!(
    p::NamedTuple,
    s::PrognosticSoil{F},
    canopy,
    radiation::PrescribedRadiativeFluxes,
    earth_param_set::PSE,
    Y::ClimaCore.Fields.FieldVector,
    t,
) where {F, PSE}
    nothing
end
function ClimaLand.Soil.sublimation_source(
    ::Val{(:canopy, :soil, :soilco2)},
    FT,
)
    return ClimaLand.Soil.SoilSublimation{FT}()
end

function ClimaLand.get_drivers(model::SoilCanopyModel)
    return (
        model.canopy.atmos,
        model.canopy.radiation,
        model.soilco2.drivers.soc,
    )
end
