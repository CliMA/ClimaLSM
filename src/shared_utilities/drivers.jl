import ClimaUtilities.TimeVaryingInputs:
    TimeVaryingInput,
    AbstractTimeVaryingInput,
    LinearInterpolation,
    PeriodicCalendar
import ClimaUtilities.Regridders: InterpolationsRegridder
import ClimaUtilities.FileReaders: NCFileReader, read
using Thermodynamics
using ClimaCore
using Dates
using DocStringExtensions
using Insolation
using SurfaceFluxes
import SurfaceFluxes.Parameters as SFP
using StaticArrays
import ..Parameters as LP
export AbstractAtmosphericDrivers,
    AbstractRadiativeDrivers,
    PrescribedAtmosphere,
    PrescribedPrecipitation,
    PrescribedSoilOrganicCarbon,
    CoupledAtmosphere,
    PrescribedRadiativeFluxes,
    CoupledRadiativeFluxes,
    compute_ρ_sfc,
    set_atmos_ts!,
    turbulent_fluxes!,
    net_radiation!,
    turbulent_fluxes_at_a_point,
    vapor_pressure_deficit,
    displacement_height,
    relative_humidity,
    specific_humidity_from_dewpoint,
    make_update_drivers,
    prescribed_lai_era5,
    prescribed_forcing_era5,
    prescribed_analytic_forcing

"""
     AbstractClimaLandDrivers{FT <: AbstractFloat}

An abstract type of radiative drivers of land models.
"""
abstract type AbstractClimaLandDrivers{FT <: AbstractFloat} end

"""
    initialize_drivers(::AbstractClimaLandDrivers, coords)

Creates and returns a default empty NamedTuple for AbstractClimaLandDrivers.
More generally this should return a named tuple of the driver fields, which will
then be stored in the cache under `p.drivers`.
"""
initialize_drivers(::AbstractClimaLandDrivers, coords) = (;)


"""
    make_update_drivers(::AbstractClimaLandDrivers)

Creates and returns a function which updates the driver variables
in the default case of no drivers. More generally, this should return
a function which updates the driver fields stored in `p.drivers`.
"""
function make_update_drivers(::AbstractClimaLandDrivers)
    update_drivers!(p, t) = nothing
    return update_drivers!
end


"""
     AbstractAtmosphericDrivers{FT}

An abstract type of atmospheric drivers of land models.
"""
abstract type AbstractAtmosphericDrivers{FT} <: AbstractClimaLandDrivers{FT} end

"""
     AbstractRadiativeDrivers{FT}

An abstract type of radiative drivers of land models.
"""
abstract type AbstractRadiativeDrivers{FT} <: AbstractClimaLandDrivers{FT} end


"""
     PrescribedSoilOrganicCarbon{FT}

A type for prescribing soil organic carbon.
$(DocStringExtensions.FIELDS)
"""
struct PrescribedSoilOrganicCarbon{FT, SOC <: AbstractTimeVaryingInput} <:
       AbstractClimaLandDrivers{FT}
    "Soil organic carbon, function of time and space: kg C/m^3"
    soc::SOC
end

PrescribedSoilOrganicCarbon{FT}(soc) where {FT} =
    PrescribedSoilOrganicCarbon{FT, typeof(soc)}(soc)

"""
    PrescribedAtmosphere{FT, CA, DT} <: AbstractAtmosphericDrivers{FT}

Container for holding prescribed atmospheric drivers and other
information needed for computing turbulent surface fluxes when
driving land models in standalone mode.

The default CO2 concentration is a constant as a function of time, equal to
4.2e-4 mol/mol.

Since not all models require co2 concentration, the default for that
is `nothing`.
$(DocStringExtensions.FIELDS)
"""
struct PrescribedAtmosphere{
    FT,
    LP <: Union{Nothing, AbstractTimeVaryingInput},
    SP <: Union{Nothing, AbstractTimeVaryingInput},
    TA <: Union{Nothing, AbstractTimeVaryingInput},
    UA <: Union{Nothing, AbstractTimeVaryingInput},
    QA <: Union{Nothing, AbstractTimeVaryingInput},
    RA <: Union{Nothing, AbstractTimeVaryingInput},
    CA <: Union{Nothing, AbstractTimeVaryingInput},
    DT,
    TP,
} <: AbstractAtmosphericDrivers{FT}
    "Precipitation (m/s) function of time: positive by definition"
    liquid_precip::LP
    "Snow precipitation (m/s) function of time: positive by definition"
    snow_precip::SP
    "Prescribed atmospheric temperature (function of time)  at the reference height (K)"
    T::TA
    "Prescribed wind speed (function of time)  at the reference height (m/s)"
    u::UA
    "Prescribed specific humidity (function of time)  at the reference height (_)"
    q::QA
    "Prescribed air pressure (function of time)  at the reference height (Pa)"
    P::RA
    "CO2 concentration in atmosphere (mol/mol)"
    c_co2::CA
    "Start date - the datetime corresponding to t=0 for the simulation"
    start_date::DT
    "Reference height (m), relative to surface elevation"
    h::FT
    "Minimum wind speed (gustiness; m/s)"
    gustiness::FT
    "Thermodynamic parameters"
    thermo_params::TP
    function PrescribedAtmosphere(
        liquid_precip,
        snow_precip,
        T,
        u,
        q,
        P,
        start_date,
        h::FT,
        earth_param_set;
        gustiness = FT(1),
        c_co2 = TimeVaryingInput((t) -> 4.2e-4),
    ) where {FT}
        thermo_params = LP.thermodynamic_parameters(earth_param_set)
        args = (liquid_precip, snow_precip, T, u, q, P, c_co2, start_date)
        return new{typeof(h), typeof.(args)..., typeof(thermo_params)}(
            args...,
            h,
            gustiness,
            thermo_params,
        )
    end
end

"""
    PrescribedPrecipitation{FT, LP} <: AbstractAtmosphericDrivers{FT}
Container for holding prescribed precipitation driver
for models which only require precipitation (RichardsModel).
$(DocStringExtensions.FIELDS)
"""
struct PrescribedPrecipitation{FT, LP <: AbstractTimeVaryingInput} <:
       AbstractAtmosphericDrivers{FT}
    "Precipitation (m/s) function of time: positive by definition"
    liquid_precip::LP
end

PrescribedPrecipitation{FT}(liquid_precip) where {FT} =
    PrescribedPrecipitation{FT, typeof(liquid_precip)}(liquid_precip)

"""
    CoupledRadiativeFluxes{FT} <: AbstractRadiativeDrivers{FT}

To be used when coupling to an atmosphere model.
"""
struct CoupledRadiativeFluxes{FT} <: AbstractRadiativeDrivers{FT} end

"""
    CoupledAtmosphere{FT} <: AbstractAtmosphericDrivers{FT}

To be used when coupling to an atmosphere model.
"""
struct CoupledAtmosphere{FT, DT} <: AbstractAtmosphericDrivers{FT}
    "Start date - the datetime corresponding to t=0 for the simulation" #TODO: move somewhere else
    start_date::DT
end

"""
    compute_ρ_sfc(thermo_params, ts_in, T_sfc)

Computes the density of air at the surface, given the temperature
at the surface T_sfc, the thermodynamic state of the atmosphere,
ts_in, and a set of Clima.Thermodynamics parameters thermo_params.

This assumes the ideal gas law and hydrostatic balance to
extrapolate to the surface.

"""
function compute_ρ_sfc(thermo_params, ts_in, T_sfc)
    T_int = Thermodynamics.air_temperature(thermo_params, ts_in)
    Rm_int = Thermodynamics.gas_constant_air(thermo_params, ts_in)
    ρ_air = Thermodynamics.air_density(thermo_params, ts_in)
    ρ_sfc =
        ρ_air *
        (T_sfc / T_int)^(Thermodynamics.cv_m(thermo_params, ts_in) / Rm_int)
    return ρ_sfc
end

"""
    set_atmos_ts!(ts_in, atmos::PrescribedAtmosphere{FT}, p)

Fill the pre-allocated ts_in `Field` with a thermodynamic state computed from the
atmosphere.
"""
function set_atmos_ts!(ts_in, atmos::PrescribedAtmosphere{FT}, p) where {FT}
    P = p.drivers.P
    T = p.drivers.T
    q = p.drivers.q
    ts_in .= Thermodynamics.PhaseEquil_pTq.(atmos.thermo_params, P, T, q)
    return nothing
end

"""
    turbulent_fluxes!(dest,
                      atmos::PrescribedAtmosphere,
                      model::AbstractModel,
                      Y::ClimaCore.Fields.FieldVector,
                      p::NamedTuple,
                      t
                      )

Computes the turbulent surface flux terms at the ground for a standalone simulation,
including turbulent energy fluxes as well as the water vapor flux
(in units of m^3/m^2/s of water).
Positive fluxes indicate flow from the ground to the atmosphere.

It solves for these given atmospheric conditions, stored in `atmos`,
model parameters, and the surface conditions.
"""
function turbulent_fluxes!(
    dest,
    atmos::PrescribedAtmosphere,
    model::AbstractModel,
    Y::ClimaCore.Fields.FieldVector,
    p::NamedTuple,
    t,
)
    T_sfc = surface_temperature(model, Y, p, t)
    ρ_sfc = surface_air_density(atmos, model, Y, p, t, T_sfc)
    q_sfc = surface_specific_humidity(model, Y, p, T_sfc, ρ_sfc)
    β_sfc = surface_evaporative_scaling(model, Y, p)
    h_sfc = surface_height(model, Y, p)
    r_sfc = surface_resistance(model, Y, p, t)
    d_sfc = displacement_height(model, Y, p)
    u_air = p.drivers.u
    h_air = atmos.h

    dest .=
        turbulent_fluxes_at_a_point.(
            T_sfc,
            q_sfc,
            ρ_sfc,
            β_sfc,
            h_sfc,
            r_sfc,
            d_sfc,
            p.drivers.thermal_state,
            u_air,
            h_air,
            atmos.gustiness,
            model.parameters.z_0m,
            model.parameters.z_0b,
            model.parameters.earth_param_set,
        )
    return nothing
end


"""
    turbulent_fluxes_at_a_point(T_sfc::FT,
                                q_sfc::FT,
                                ρ_sfc::FT,
                                β_sfc::FT,
                                h_sfc::FT,
                                r_sfc::FT,
                                d_sfc::FT,
                                ts_in,
                                u::FT,
                                h::FT,
                                gustiness::FT,
                                z_0m::FT,
                                z_0b::FT,
                                earth_param_set::EP,
                               ) where {FT <: AbstractFloat, P}

Computes turbulent surface fluxes at a point on a surface given
(1) the surface temperature (T_sfc), specific humidity (q_sfc),
    and air density (ρ_sfc),
(2) Other surface properties, such as the factor
    β_sfc  which scales the evaporation from the potential rate
    (used in bucket models), and the surface resistance r_sfc (used
    in more complex land models), and the topographical height of the surface (h_sfc)
(3) the roughness lengths `z_0m, z_0b`, and the Earth parameter set for the model
    `earth_params`.
(4) the prescribed atmospheric state, `ts_in`, u, h the height
    at which these measurements are made, and the gustiness parameter (m/s).
(5) the displacement height for the model d_sfc

This returns an energy flux and a liquid water volume flux, stored in
a tuple with self explanatory keys.

Please note that this function, if r_sfc is set to zero, makes no alteration
to the output of the `SurfaceFluxes.surface_conditions` call, aside
from unit conversion of evaporation from a mass to a liquid water volume
flux. When r_sfc is nonzero, an additional resistance is applied in series
to the vapor flux (and hence also the latent heat flux).
"""
function turbulent_fluxes_at_a_point(
    T_sfc::FT,
    q_sfc::FT,
    ρ_sfc::FT,
    β_sfc::FT,
    h_sfc::FT,
    r_sfc::FT,
    d_sfc::FT,
    ts_in,
    u::FT,
    h::FT,
    gustiness::FT,
    z_0m::FT,
    z_0b::FT,
    earth_param_set::EP,
) where {FT <: AbstractFloat, EP}
    thermo_params = LP.thermodynamic_parameters(earth_param_set)
    ts_sfc = Thermodynamics.PhaseEquil_ρTq(thermo_params, ρ_sfc, T_sfc, q_sfc)

    # SurfaceFluxes.jl expects a relative difference between where u = 0
    # and the atmosphere height. Here, we assume h and h_sfc are measured
    # relative to a common reference. Then d_sfc + h_sfc + z_0m is the apparent
    # source of momentum, and
    # Δh ≈ h - d_sfc - h_sfc is the relative height difference between the
    # apparent source of momentum and the atmosphere height.

    # In this we have neglected z_0m and z_0b (i.e. assumed they are small
    # compared to Δh).
    state_sfc = SurfaceFluxes.StateValues(FT(0), SVector{2, FT}(0, 0), ts_sfc)
    state_in = SurfaceFluxes.StateValues(
        h - d_sfc - h_sfc,
        SVector{2, FT}(u, 0),
        ts_in,
    )
    # The following line wont work on GPU
    #    h - d_sfc - h_sfc < 0 &&
    #        @error("Surface height is larger than atmos height in surface fluxes")
    # State containers
    states = SurfaceFluxes.ValuesOnly(
        state_in,
        state_sfc,
        z_0m,
        z_0b,
        beta = β_sfc,
        gustiness = gustiness,
    )
    surface_flux_params = LP.surface_fluxes_parameters(earth_param_set)
    scheme = SurfaceFluxes.PointValueScheme()
    conditions =
        SurfaceFluxes.surface_conditions(surface_flux_params, states, scheme)
    _LH_v0::FT = LP.LH_v0(earth_param_set)
    _ρ_liq::FT = LP.ρ_cloud_liq(earth_param_set)

    # aerodynamic resistance
    r_ae::FT = 1 / (conditions.Ch * SurfaceFluxes.windspeed(states))

    # latent heat flux
    E0::FT =
        SurfaceFluxes.evaporation(surface_flux_params, states, conditions.Ch) # mass flux at potential evaporation rate
    E = E0 * r_ae / (r_sfc + r_ae) # mass flux accounting for additional surface resistance
    LH = _LH_v0 * E # Latent heat flux

    # sensible heat flux
    SH = SurfaceFluxes.sensible_heat_flux(
        surface_flux_params,
        conditions.Ch,
        states,
        scheme,
    )

    # vapor flux in volume of liquid water with density 1000kg/m^3
    Ẽ = E / _ρ_liq

    return (lhf = LH, shf = SH, vapor_flux = Ẽ, r_ae = r_ae)
end

"""
    turbulent_fluxes!(dest,
                    atmos::CoupledAtmosphere,
                    model::AbstractModel,
                    Y,
                    p,
                    t)

Computes the turbulent surface fluxes terms at the ground for a coupled simulation.
"""
function ClimaLand.turbulent_fluxes!(
    dest,
    atmos::CoupledAtmosphere,
    model::AbstractModel,
    Y,
    p,
    t,
)
    # coupler has done its thing behind the scenes already
    model_name = ClimaLand.name(model)
    model_cache = getproperty(p, model_name)
    if model_name == :canopy
        dest .= model_cache.energy.turbulent_fluxes
    else
        dest .= model_cache.turbulent_fluxes
    end
    return nothing
end


"""
    PrescribedRadiativeFluxes{FT, SW, LW, DT, T, TP} <: AbstractRadiativeDrivers{FT}

Container for the prescribed radiation functions needed to drive land models in standalone mode.

Note that some models require the zenith angle AND diffuse fraction. This
requires the thermodynamic parameters as well to compute. Therefore,
you must either pass the optional zenith angle function argument and
thermodynamic parameters, or pass neither.
$(DocStringExtensions.FIELDS)
"""
struct PrescribedRadiativeFluxes{
    FT,
    SW <: Union{Nothing, AbstractTimeVaryingInput},
    LW <: Union{Nothing, AbstractTimeVaryingInput},
    DT,
    T,
    TP,
} <: AbstractRadiativeDrivers{FT}
    "Downward shortwave radiation function of time (W/m^2): positive indicates towards surface"
    SW_d::SW
    "Downward longwave radiation function of time (W/m^2): positive indicates towards surface"
    LW_d::LW
    "Start date - the datetime corresponding to t=0 for the simulation"
    start_date::DT
    "Sun zenith angle, in radians"
    θs::T
    "Thermodynamic parameters"
    thermo_params::TP
    function PrescribedRadiativeFluxes(
        FT,
        SW_d,
        LW_d,
        start_date;
        θs = nothing,
        earth_param_set = nothing,
    )
        if isnothing(earth_param_set)
            thermo_params = nothing
        else
            thermo_params = LP.thermodynamic_parameters(earth_param_set)
        end

        args = (SW_d, LW_d, start_date, θs, thermo_params)
        return new{FT, typeof.(args)...}(args...)
    end
end

"""
    net_radiation!(dest::ClimaCore.Fields.Field,
                   radiation::PrescribedRadiativeFluxes{FT},
                   model::AbstractModel{FT},
                   Y::ClimaCore.Fields.FieldVector,
                   p::NamedTuple,
                   t,
                   ) where {FT}


Computes net radiative  fluxes for a prescribed incoming  longwave and shortwave
radiation.
"""
function net_radiation!(
    dest::ClimaCore.Fields.Field,
    radiation::PrescribedRadiativeFluxes,
    model::AbstractModel,
    Y::ClimaCore.Fields.FieldVector,
    p::NamedTuple,
    t,
)
    LW_d = p.drivers.LW_d
    SW_d = p.drivers.SW_d
    earth_param_set = model.parameters.earth_param_set
    _σ = LP.Stefan(earth_param_set)
    T_sfc = surface_temperature(model, Y, p, t)
    α_sfc = surface_albedo(model, Y, p)
    ϵ_sfc = surface_emissivity(model, Y, p)
    # Recall that the user passed the LW and SW downwelling radiation,
    # where positive values indicate toward surface, so we need a negative sign out front
    # in order to inidicate positive R_n  = towards atmos.
    @. dest = -(1 - α_sfc) * SW_d - ϵ_sfc * (LW_d - _σ * T_sfc^4)
    return nothing
end


"""
    net_radiation!(dest,
                  radiation::CoupledRadiativeFluxes,
                  model::AbstractModel,
                  Y,
                  p,
                  t)

Computes the net radiative flux at the ground for a coupled simulation.
Your model cache must contain the field `R_n`.
"""
function ClimaLand.net_radiation!(
    dest,
    radiation::CoupledRadiativeFluxes,
    model::AbstractModel,
    Y::ClimaCore.Fields.FieldVector,
    p::NamedTuple,
    t,
)
    # coupler has done its thing behind the scenes already
    model_name = ClimaLand.name(model)
    model_cache = getproperty(p, model_name)
    dest .= model_cache.R_n
    return nothing
end

"""
    surface_temperature(model::AbstractModel, Y, p, t)

A helper function which returns the surface temperature for a given
model, needed because different models compute and store surface temperature in
different ways and places.

Extending this function for your model is only necessary if you need to
compute surface fluxes and radiative fluxes at the surface using
the functions in this file.
"""
function surface_temperature(model::AbstractModel, Y, p, t) end

"""
    surface_resistance(model::AbstractModel, Y, p, t)

A helper function which returns the surface resistance for a given
model, needed because different models compute and store surface resistance in
different ways and places.

Extending this function for your model is only necessary if you need to
compute surface fluxes and radiative fluxes at the surface using
the functions in this file.

The default is 0, which is no additional resistance aside from the usual
aerodynamic resistance from MOST.
"""
function surface_resistance(model::AbstractModel{FT}, Y, p, t) where {FT}
    return FT(0)
end


"""
    surface_air_density(
                        atmos::PrescribedAtmosphere,
                        model::AbstractModel,
                        Y,
                        p,
                        t,
                        T_sfc,
                        )

A helper function which returns the surface air density; this assumes that
the `model` has a property called `parameters` containing `earth_param_set`.

We additionally include the `atmos` type as an argument because
the surface air density computation will change between a coupled simulation
and a prescibed atmos simulation.

Extending this function for your model is only necessary if you need to
compute the air density in a different way.
"""
function surface_air_density(
    atmos::PrescribedAtmosphere,
    model::AbstractModel,
    Y,
    p,
    t,
    T_sfc,
)
    thermo_params =
        LP.thermodynamic_parameters(model.parameters.earth_param_set)
    return compute_ρ_sfc.(thermo_params, p.drivers.thermal_state, T_sfc)
end


"""
    ClimaLand.surface_air_density(
                    atmos::CoupledAtmosphere,
                    model::AbstractModel,
                    Y,
                    p,
                    _...,
                )
Returns the air density at the surface in the case of a coupled simulation.

This requires the field `ρ_sfc` to be present in the cache `p` under the name
of the model.
"""
function surface_air_density(
    atmos::CoupledAtmosphere,
    model::AbstractModel,
    Y,
    p,
    _...,
)
    model_name = ClimaLand.name(model)
    model_cache = getproperty(p, model_name)
    return model_cache.ρ_sfc
end

"""
    surface_specific_humidity(model::AbstractModel, Y, p, T_sfc, ρ_sfc)

A helper function which returns the surface specific humidity for a given
model, needed because different models compute and store q_sfc in
different ways and places.

Extending this function for your model is only necessary if you need to
compute surface fluxes and radiative fluxes at the surface using
the functions in this file.
"""
function surface_specific_humidity(model::AbstractModel, Y, p, T_sfc, ρ_sfc) end

"""
    surface_evaporative_scaling(model::AbstractModel{FT}, Y, p) where {FT}

A helper function which returns the surface evaporative scaling factor
 for a given model, needed because different models compute and store β_sfc in
different ways and places. Currently, this factor is 1 for all models
besides the bucket model, so we have chosen a default of 1.

Extending this function for your model is only necessary if you need to
compute surface fluxes and radiative fluxes at the surface using
the functions in this file.
"""
function surface_evaporative_scaling(model::AbstractModel{FT}, Y, p) where {FT}
    return FT(1)
end


"""
    surface_albedo(model::AbstractModel, Y, p)

A helper function which returns the surface albedo
for a given model, needed because different models compute and store α_sfc in
different ways and places.

Extending this function for your model is only necessary if you need to
compute surface fluxes and radiative fluxes at the surface using
the functions in this file.
"""
function surface_albedo(model::AbstractModel, Y, p) end

"""
    surface_emissivity(model::AbstractModel, Y, p)

A helper function which returns the surface emissivity
 for a given model, needed because different models compute and store ϵ_sfc in
different ways and places.

Extending this function for your model is only necessary if you need to
compute surface fluxes and radiative fluxes at the surface using
the functions in this file.
"""
function surface_emissivity(model::AbstractModel, Y, p) end


"""
    surface_height(model::AbstractModel, Y, p)

A helper function which returns the surface height (canopy height+elevation)
 for a given model, needed because different models compute and store h_sfc in
different ways and places.

Extending this function for your model is only necessary if you need to
compute surface fluxes and radiative fluxes at the surface using
the functions in this file.
"""
function surface_height(model::AbstractModel, Y, p) end

"""
    displacement_height(model::AbstractModel, Y, p)

A helper function which returns the displacement height
 for a given model; the default is zero.

Extending this function for your model is only necessary if you need to
compute surface fluxes and radiative fluxes at the surface using
the functions in this file.
"""
function displacement_height(model::AbstractModel{FT}, Y, p) where {FT}
    return FT(0)
end

"""
    vapor_pressure_deficit(T_air, P_air, q_air, thermo_params)

Computes the vapor pressure deficit for air with temperature T_air,
pressure P_air, and specific humidity q_air, using thermo_params,
a Thermodynamics.jl param set.
"""
function vapor_pressure_deficit(T_air, P_air, q_air, thermo_params)
    es = Thermodynamics.saturation_vapor_pressure(
        thermo_params,
        T_air,
        Thermodynamics.Liquid(),
    )
    ea = Thermodynamics.partial_pressure_vapor(
        thermo_params,
        P_air,
        Thermodynamics.PhasePartition(q_air),
    )
    return es - ea
end

"""
    relative_humidity(T_air, P_air, q_air, thermo_params)

Computes the vapor pressure deficit for air with temperature T_air,
pressure P_air, and specific humidity q_air, using thermo_params,
a Thermodynamics.jl param set.
"""
function relative_humidity(
    T_air::FT,
    P_air::FT,
    q_air::FT,
    thermo_params,
) where {FT}
    es = Thermodynamics.saturation_vapor_pressure(
        thermo_params,
        T_air,
        Thermodynamics.Liquid(),
    )
    ea = Thermodynamics.partial_pressure_vapor(
        thermo_params,
        P_air,
        Thermodynamics.PhasePartition(q_air),
    )
    return es / ea
end

"""
    specific_humidity_from_dewpoint(dewpoint_temperature, temperature, air_pressure, earth_param_set)

Estimates the specific humidity given the dewpoint temperature, temperature of the air
in Kelvin, and air pressure in Pa, along with the ClimaLand earth_param_set. This is useful
for creating the PrescribedAtmosphere - which needs specific humidity - from ERA5 reanalysis data.

We first compute the relative humidity using the Magnus formula, then the saturated vapor pressure, and then
we compute q from vapor pressure and saturated vapor pressure.

For more information on the Magnus formula, see e.g.
Lawrence, Mark G. (1 February 2005).
"The Relationship between Relative Humidity and the Dewpoint Temperature in Moist Air:
A Simple Conversion and Applications".
Bulletin of the American Meteorological Society. 86 (2): 225–234.
"""
function specific_humidity_from_dewpoint(
    T_dew_air::data_FT,
    T_air::data_FT,
    P_air::data_FT,
    earth_param_set,
) where {data_FT <: Real}
    thermo_params = LP.thermodynamic_parameters(earth_param_set)
    _T_freeze = LP.T_freeze(earth_param_set)
    sim_FT = typeof(_T_freeze)
    # Obtain the relative humidity. This function requires temperatures in Celsius
    rh::sim_FT = rh_from_dewpoint(
        sim_FT(T_dew_air) - _T_freeze,
        sim_FT(T_air) - _T_freeze,
    )

    # In the future, we will use Thermodynamics functions for all of the below. Note that the type of T_air must match the type of the parameters in thermo_params.
    esat::sim_FT = Thermodynamics.saturation_vapor_pressure(
        thermo_params,
        sim_FT(T_air),
        Thermodynamics.Liquid(),
    )

    e = rh * esat
    q = data_FT(0.622 * e / (P_air - 0.378 * e))
    return q
end

"""
    rh_from_dewpoint(Td_C, T_C)

Returns the relative humidity given the dewpoint temperature in Celsius and the
air temperature in Celsius, using the Magnus formula.

For more information on the Magnus formula, see e.g.
Lawrence, Mark G. (1 February 2005).
"The Relationship between Relative Humidity and the Dewpoint Temperature in Moist Air:
A Simple Conversion and Applications".
Bulletin of the American Meteorological Society. 86 (2): 225–234.
"""
function rh_from_dewpoint(Td_C::FT, T_C::FT) where {FT <: Real}
    c = FT(243.04) # C
    b = FT(17.625) # unitless
    γ = Td_C * b / (c + Td_C)
    rh = exp(γ - b * T_C / (c + T_C))
    return rh
end


"""
    initialize_drivers(a::PrescribedAtmosphere{FT}, coords) where {FT}

Creates and returns a NamedTuple for the `PrescribedAtmosphere` driver,
with variables `P_liq`, `P_snow`, and air temperature `T`, pressure `P`,
horizontal wind speed `u`, specific humidity `q`, and CO2 concentration
`c_co2`.
"""
function initialize_drivers(a::PrescribedAtmosphere{FT}, coords) where {FT}
    keys = (:P_liq, :P_snow, :T, :P, :u, :q, :c_co2, :thermal_state)
    # The thermal state is a different type
    types = ([FT for k in keys[1:(end - 1)]]..., Thermodynamics.PhaseEquil{FT})
    domain_names = ([:surface for k in keys]...,)
    model_name = :drivers
    # intialize_vars packages the variables as a named tuple,
    # as part of a named tuple with `model_name` as the key.
    # Here we just want the variable named tuple itself
    vars =
        ClimaLand.initialize_vars(keys, types, domain_names, coords, model_name)
    return vars.drivers
end

"""
    initialize_drivers(a::PrescribedPrecipitation{FT}, coords) where {FT}
Creates and returns a NamedTuple for the `PrescribedPrecipitation` driver,
with variable `P_liq`.
"""
function initialize_drivers(a::PrescribedPrecipitation{FT}, coords) where {FT}
    keys = (:P_liq,)
    types = (FT,)
    domain_names = (:surface,)
    model_name = :drivers
    # intialize_vars packages the variables as a named tuple,
    # as part of a named tuple with `model_name` as the key.
    # Here we just want the variable named tuple itself
    vars =
        ClimaLand.initialize_vars(keys, types, domain_names, coords, model_name)
    return vars.drivers
end

"""
    initialize_drivers(r::CoupledAtmosphere{FT}, coords) where {FT}

Creates and returns a NamedTuple for the `CoupledAtmosphere` driver,
with variables `P_liq`, and `P_snow` TODO: fill remaining. This is intended to be used in coupled
simulations with ClimaCoupler.jl
"""
function ClimaLand.initialize_drivers(
    a::CoupledAtmosphere{FT},
    coords,
) where {FT}
    # TODO: drop thermal state, P, T, q, ρ (maybe).
    # things that need to be changed to not have the above variables in driver cache:
    #  the canopy update_aux uses q, P, T to calculate diffuse_frac. Get it from coupler
    # canopy update aux also uses them for medlyn_term
    # the canopy_temperature uses T from driver when using prescribedcanopytemp
    # snow update_aux uses thermal state to calc q_sfc.
    # compute_jacobian! ofr BigLeafEnergyModel uses P
    keys = (:P_liq, :P_snow, :c_co2, :T, :P, :ρ, :q, :thermal_state)
    ρ_types =
        NamedTuple{(:ρ_eff, :ρ_soil, :ρ_snow, :ρ_canopy), Tuple{FT, FT, FT, FT}}
    q_types =
        NamedTuple{(:q_eff, :q_soil, :q_snow, :q_canopy), Tuple{FT, FT, FT, FT}}

    types = (
        [FT for k in keys[1:(end - 3)]]...,
        ρ_types,
        q_types,
        Thermodynamics.PhaseEquil{FT},
    )
    domain_names = ([:surface for k in keys]...,)
    model_name = :drivers
    # intialize_vars packages the variables as a named tuple,
    # as part of a named tuple with `model_name` as the key.
    # Here we just want the variable named tuple itself
    vars =
        ClimaLand.initialize_vars(keys, types, domain_names, coords, model_name)
    return vars.drivers
end

"""
    initialize_drivers(r::PrescribedRadiativeFluxes{FT}, coords) where {FT}

Creates and returns a NamedTuple for the `PrescribedRadiativeFluxes` driver,
 with variables `SW_d`, `LW_d`, cosine of the zenith angle `θ_s`, and the diffuse fraction
of shortwave radiation `frac_diff`.
"""
function initialize_drivers(r::PrescribedRadiativeFluxes{FT}, coords) where {FT}
    keys = (:SW_d, :LW_d, :cosθs, :frac_diff)
    types = ([FT for k in keys]...,)
    domain_names = ([:surface for k in keys]...,)
    model_name = :drivers
    # intialize_vars packages the variables as a named tuple,
    # as part of a named tuple with `model_name` as the key.
    # Here we just want the variable named tuple itself
    vars =
        ClimaLand.initialize_vars(keys, types, domain_names, coords, model_name)
    return vars.drivers
end

"""
    initialize_drivers(r::PrescribedSoilOrganicCarbon{FT}, coords) where {FT}

Creates and returns a NamedTuple for the `PrescribedSoilOrganicCarbon` driver,
 with variable `soc`.
"""
function initialize_drivers(
    r::PrescribedSoilOrganicCarbon{FT},
    coords,
) where {FT}
    keys = (:soc,)
    types = (FT,)
    domain_names = (:subsurface,)
    model_name = :drivers
    # intialize_vars packages the variables as a named tuple,
    # as part of a named tuple with `model_name` as the key.
    # Here we just want the variable named tuple itself
    vars =
        ClimaLand.initialize_vars(keys, types, domain_names, coords, model_name)
    return vars.drivers
end

"""
    initialize_drivers(driver_tuple::Tuple,
                       coords)

Creates and returns a NamedTuple with the cache variables required by the
model drivers.

If no forcing is required,  driver_tuple is an empty tuple, and an
empty NamedTuple is returned.
"""
function initialize_drivers(driver_tuple::Tuple, coords)
    if isempty(driver_tuple)
        return (;)
    else
        tmp = map(driver_tuple) do (driver)
            nt = initialize_drivers(driver, coords)
        end
        merge(tmp...)
    end

end

"""
    make_update_drivers(driver_tuple)

Creates and returns a function which updates the forcing variables ("drivers"). If no drivers are being used, driver_tuple is empty, and the update
function does nothing.
"""
function make_update_drivers(driver_tuple::Tuple)
    if isempty(driver_tuple)
        return (p, t) -> nothing
    else
        update_driver_list = map(driver_tuple) do (driver)
            make_update_drivers(driver)
        end
        function update_drivers!(p, t)
            for ud! in update_driver_list
                ud!(p, t)
            end
        end
        return update_drivers!
    end
end


"""
    make_update_drivers(a::PrescribedAtmosphere{FT}) where {FT}

Creates and returns a function which updates the driver variables
in the case of a PrescribedAtmosphere.
"""
function make_update_drivers(a::PrescribedAtmosphere{FT}) where {FT}
    function update_drivers!(p, t)
        evaluate!(p.drivers.P_liq, a.liquid_precip, t)
        evaluate!(p.drivers.P_snow, a.snow_precip, t)
        evaluate!(p.drivers.T, a.T, t)
        evaluate!(p.drivers.P, a.P, t)
        evaluate!(p.drivers.u, a.u, t)
        evaluate!(p.drivers.q, a.q, t)
        evaluate!(p.drivers.c_co2, a.c_co2, t)
        set_atmos_ts!(p.drivers.thermal_state, a, p)
    end
    return update_drivers!
end

"""
    make_update_drivers(a::PrescribedPrecipitation{FT}) where {FT}
Creates and returns a function which updates the driver variables
in the case of a PrescribedPrecipitation.
"""
function make_update_drivers(a::PrescribedPrecipitation{FT}) where {FT}
    function update_drivers!(p, t)
        evaluate!(p.drivers.P_liq, a.liquid_precip, t)
    end
    return update_drivers!
end

"""
    make_update_drivers(r::PrescribedRadiativeFluxes{FT}) where {FT}

Creates and returns a function which updates the driver variables
in the case of a PrescribedRadiativeFluxes.
"""
function make_update_drivers(r::PrescribedRadiativeFluxes{FT}) where {FT}
    function update_drivers!(p, t)
        evaluate!(p.drivers.SW_d, r.SW_d, t)
        evaluate!(p.drivers.LW_d, r.LW_d, t)
        # This assumes that temperature, pressure, and specific humidity have already updated
        # and that PrescribedRadiation is only used with PrescribedAtmos.
        # If this is not the case, this function would use the vaues of T, P, q
        # from the previous step.
        # Since this function is empirical and our steps are O(10 minutes), this is
        # not a big issue.
        # In practice, the diffuse fraction should be supplied by the Radiative
        # Transfer module of the atmosphere, or obtained from ERA5,
        # and not be computed empirically
        # If we obtain this factor from ERA5 in the future, we should remove thermo_params from the struct.
        if !isnothing(r.θs) & !isnothing(r.thermo_params)
            p.drivers.cosθs .= FT.(cos.(r.θs(t, r.start_date)))
            @. p.drivers.frac_diff = diffuse_fraction(
                t,
                r.start_date,
                p.drivers.T,
                p.drivers.P,
                p.drivers.q,
                p.drivers.SW_d,
                p.drivers.cosθs,
                r.thermo_params,
            )
        elseif isnothing(r.θs) & isnothing(r.thermo_params)
            # These should not be accessed or used. Set to NaN.
            p.drivers.cosθs .= FT(NaN)
            p.drivers.frac_diff .= FT(NaN)
        else
            @error(
                "Zenith angle and thermodynamic params must either both be supplied or both not supplied. See doc string for PrescribedRadiativeFluxes"
            )
        end
    end

    return update_drivers!
end

"""
    make_update_drivers(d::PrescribedSoilOrganicCarbon{FT}) where {FT}
Creates and returns a function which updates the driver variables
in the case of a PrescribedSoilOrganicCarbon.
"""
function make_update_drivers(d::PrescribedSoilOrganicCarbon{FT}) where {FT}
    function update_drivers!(p, t)
        evaluate!(p.drivers.soc, d.soc, t)
    end
    return update_drivers!
end

"""
     prescribed_forcing_era5(era5_ncdata_path,
                             surface_space,
                             start_date,
                             earth_param_set,
                             FT;
                             gustiness=1,
                             c_co2 = TimeVaryingInput((t) -> 4.2e-4),
                             time_interpolation_method = LinearInterpolation(PeriodicCalendar()),
                             regridder_type = :InterpolationsRegridder)

A helper function which constructs the `PrescribedAtmosphere` and `PrescribedRadiativeFluxes`
from a file path pointing to the ERA5 data in a netcdf file, the surface_space, the start date,
and the earth_param_set.
"""
function prescribed_forcing_era5(
    era5_ncdata_path,
    surface_space,
    start_date,
    earth_param_set,
    FT;
    gustiness = 1,
    c_co2 = TimeVaryingInput((t) -> 4.2e-4),
    time_interpolation_method = LinearInterpolation(PeriodicCalendar()),
    regridder_type = :InterpolationsRegridder,
)

    # Precip is provide as a mass flux; convert to volume flux of liquid water with ρ =1000 kg/m^3
    precip = TimeVaryingInput(
        era5_ncdata_path,
        ["mtpr", "msr"],
        surface_space;
        start_date,
        regridder_type,
        file_reader_kwargs = (; preprocess_func = (data) -> -data / 1000,),
        method = time_interpolation_method,
        compose_function = (mtpr, msr) -> mtpr - msr,
    )
    # Precip is provide as a mass flux; convert to volume flux of liquid water with ρ =1000 kg/m^3
    snow_precip = TimeVaryingInput(
        era5_ncdata_path,
        "msr",
        surface_space;
        start_date,
        regridder_type,
        file_reader_kwargs = (; preprocess_func = (data) -> -data / 1000,),
        method = time_interpolation_method,
    )

    u_atmos = TimeVaryingInput(
        era5_ncdata_path,
        ["u10", "v10"],
        surface_space;
        start_date,
        regridder_type,
        compose_function = (u, v) -> sqrt.(u .^ 2 .+ v .^ 2),
        method = time_interpolation_method,
    )
    specific_humidity(Td, T, P; params = earth_param_set) =
        ClimaLand.specific_humidity_from_dewpoint.(Td, T, P, params)
    q_atmos = TimeVaryingInput(
        era5_ncdata_path,
        ["d2m", "t2m", "sp"],
        surface_space;
        start_date,
        regridder_type,
        compose_function = specific_humidity,
        method = time_interpolation_method,
    )
    P_atmos = TimeVaryingInput(
        era5_ncdata_path,
        "sp",
        surface_space;
        start_date,
        regridder_type,
        method = time_interpolation_method,
    )

    T_atmos = TimeVaryingInput(
        era5_ncdata_path,
        "t2m",
        surface_space;
        start_date,
        regridder_type,
        method = time_interpolation_method,
    )
    h_atmos = FT(10)

    atmos = PrescribedAtmosphere(
        precip,
        snow_precip,
        T_atmos,
        u_atmos,
        q_atmos,
        P_atmos,
        start_date,
        h_atmos,
        earth_param_set;
        gustiness = FT(gustiness),
        c_co2 = c_co2,
    )

    SW_d = TimeVaryingInput(
        era5_ncdata_path,
        "msdwswrf",
        surface_space;
        start_date,
        regridder_type,
        method = time_interpolation_method,
    )
    LW_d = TimeVaryingInput(
        era5_ncdata_path,
        "msdwlwrf",
        surface_space;
        start_date,
        regridder_type,
        method = time_interpolation_method,
    )

    function zenith_angle(
        t,
        start_date;
        latitude = ClimaCore.Fields.coordinate_field(surface_space).lat,
        longitude = ClimaCore.Fields.coordinate_field(surface_space).long,
        insol_params::Insolation.Parameters.InsolationParameters{FT} = earth_param_set.insol_params,
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

        Insolation.instantaneous_zenith_angle.(
            d,
            δ,
            η_UTC,
            longitude,
            latitude,
        ).:1
    end

    radiation = PrescribedRadiativeFluxes(
        FT,
        SW_d,
        LW_d,
        start_date;
        θs = zenith_angle,
        earth_param_set = earth_param_set,
    )
    return atmos, radiation
end


"""
     prescribed_lai_era5(era5_lai_ncdata_path,
                         era5_lai_cover_ncdata_path,
                         surface_space,
                         start_date,
                         earth_param_set;
                         time_interpolation_method = LinearInterpolation(PeriodicCalendar()),
                         regridder_type = :InterpolationsRegridder)

A helper function which constructs the TimeVaryingInput object for Leaf Area Index, from a
file path pointing to the ERA5 LAI data in a netcdf file, a file path pointing to the ERA5
LAI cover data in a netcdf file, the surface_space, the start date, and the earth_param_set.
"""
function prescribed_lai_era5(
    era5_lai_ncdata_path,
    era5_lai_cover_ncdata_path,
    surface_space,
    start_date;
    time_interpolation_method = LinearInterpolation(PeriodicCalendar()),
    regridder_type = :InterpolationsRegridder,
)
    hvc_ds = NCFileReader(era5_lai_cover_ncdata_path, "cvh")
    lvc_ds = NCFileReader(era5_lai_cover_ncdata_path, "cvl")
    hv_cover = read(hvc_ds)
    lv_cover = read(lvc_ds)
    close(hvc_ds)
    close(lvc_ds)
    compose_function = let hv_cover = hv_cover, lv_cover = lv_cover
        (lai_hv, lai_lv) -> lai_hv .* hv_cover .+ lai_lv .* lv_cover
    end
    return TimeVaryingInput(
        era5_lai_ncdata_path,
        ["lai_hv", "lai_lv"],
        surface_space;
        start_date,
        regridder_type,
        method = time_interpolation_method,
        compose_function = compose_function,
    )
end

"""
     prescribed_lai_modis(modis_lai_ncdata_path,
                          surface_space,
                          start_date,
                          earth_param_set;
                          time_interpolation_method =
                                        LinearInterpolation(PeriodicCalendar()))
                          regridder_type = :InterpolationsRegridder)

A helper function which constructure the TimeVaryingInput object for Lead Area
Index from a file path pointint to the MODIS LAI data in a netcdf file, the
surface_space, the start date, and the earth_param_set.
"""
function prescribed_lai_modis(
    modis_lai_ncdata_path,
    surface_space,
    start_date;
    time_interpolation_method = LinearInterpolation(PeriodicCalendar()),
    regridder_type = :InterpolationsRegridder,
)
    return TimeVaryingInput(
        modis_lai_ncdata_path,
        ["lai"],
        surface_space;
        start_date,
        regridder_type,
        method = time_interpolation_method,
    )
end

"""
     prescribed_analytic_forcing(FT = Float32;
                                 earth_param_set = LP.LandParameters(FT),
                                 start_date = DateTime(2005),
                                 SW_d = (t) -> 0,
                                 LW_d = (t) -> 5.67e-8 * 280.0^4.0,
                                 precip = (t) -> 0, # no precipitation
                                 T_atmos = (t) -> 280.0,
                                 u_atmos = (t) -> 1.0,
                                 q_atmos = (t) -> 0.0, # no atmos water
                                 h_atmos = FT(1e-8),
                                 P_atmos = (t) -> 101325,
                                 atmos = PrescribedAtmosphere(
                                     TimeVaryingInput(precip),
                                     TimeVaryingInput(precip),
                                     TimeVaryingInput(T_atmos),
                                     TimeVaryingInput(u_atmos),
                                     TimeVaryingInput(q_atmos),
                                     TimeVaryingInput(P_atmos),
                                     start_date,
                                     h_atmos,
                                     earth_param_set,
                                 ),
                                 radiation = PrescribedRadiativeFluxes(
                                     FT,
                                     TimeVaryingInput(SW_d),
                                     TimeVaryingInput(LW_d),
                                     start_date,
                                 ),
                             )

A helper function which constructs the `PrescribedAtmosphere` and `PrescribedRadiativeFluxes`
for a simple analytic case.
"""
function prescribed_analytic_forcing(
    FT = Float32;
    earth_param_set = LP.LandParameters(FT),
    start_date = DateTime(2005),
    SW_d = (t) -> 0,
    LW_d = (t) -> 5.67e-8 * 280.0^4.0,
    precip = (t) -> 0, # no precipitation
    T_atmos = (t) -> 280.0,
    u_atmos = (t) -> 1.0,
    q_atmos = (t) -> 0.0, # no atmos water
    h_atmos = FT(1e-8),
    P_atmos = (t) -> 101325,
    atmos = PrescribedAtmosphere(
        TimeVaryingInput(precip),
        TimeVaryingInput(precip),
        TimeVaryingInput(T_atmos),
        TimeVaryingInput(u_atmos),
        TimeVaryingInput(q_atmos),
        TimeVaryingInput(P_atmos),
        start_date,
        h_atmos,
        earth_param_set,
    ),
    radiation = PrescribedRadiativeFluxes(
        FT,
        TimeVaryingInput(SW_d),
        TimeVaryingInput(LW_d),
        start_date,
    ),
)
    return atmos, radiation
end


"""
    diffuse_fraction(td::FT, T::FT, P, q, SW_d::FT, cosθs::FT, thermo_params) where {FT}

Computes the fraction of diffuse radiation (`diff_frac`) as a function
of the solar zenith angle (`θs`), the total surface downwelling shortwave radiation (`SW_d`),
the air temperature (`T`), air pressure (`P`), specific humidity (`q`), and the day of the year
(`td`).

See Appendix A of Braghiere, "Evaluation of turbulent fluxes of CO2, sensible heat,
and latent heat as a function of aerosol optical depth over the course of deforestation
in the Brazilian Amazon" 2013.

Note that cos(θs) is equal to zero when θs = π/2, and this is a coefficient
of k₀, which we divide by in this expression. This can amplify small errors
when θs is near π/2.

This formula is empirical and can yied negative numbers depending on the
input, which, when dividing by something very near zero,
can become large negative numbers.

Because of that, we cap the returned value to lie within [0,1].
"""
function diffuse_fraction(
    t,
    start_date,
    T::FT,
    P::FT,
    q::FT,
    SW_d::FT,
    cosθs::FT,
    thermo_params,
) where {FT}
    DOY = Dates.dayofyear(start_date + Dates.Second(floor(Int64, t)))
    RH = ClimaLand.relative_humidity(T, P, q, thermo_params)
    k₀ = FT(1370 * (1 + 0.033 * cos(2π * DOY / 365))) * cosθs
    kₜ = SW_d / k₀
    if kₜ ≤ 0.3
        diff_frac = FT(
            kₜ * (1 - 0.232 * kₜ + 0.0239 * cosθs - 6.82e-4 * T + 0.0195 * RH),
        )
    elseif kₜ ≤ 0.78
        diff_frac = FT(
            kₜ *
            (1.329 - 1.716 * kₜ + 0.267 * cosθs - 3.57e-3 * T + 0.106 * RH),
        )
    else
        diff_frac =
            FT(kₜ * (0.426 * kₜ + 0.256 * cosθs - 3.49e-3 * T + 0.0734 * RH))
    end
    return max(min(diff_frac, FT(1)), FT(0))
end
