export snow_surface_temperature,
    snow_depth,
    specific_heat_capacity,
    snow_thermal_conductivity,
    snow_bulk_temperature,
    maximum_liquid_mass_fraction,
    runoff_timescale,
    compute_water_runoff,
    energy_from_q_l_and_swe,
    energy_from_T_and_swe,
    snow_cover_fraction,
    snow_bulk_density,
    phase_change_flux

"""
    snow_cover_fraction(x::FT; α = FT(1e-3))::FT where {FT}

Returns the snow cover fraction, assuming it is a heaviside
function at 1e-3 meters.

In the future we can play around with other forms.
"""
function snow_cover_fraction(x::FT; α = FT(1e-3))::FT where {FT}
    return heaviside(x - α)
end

"""
    snow_depth(model::AbstractDensityModel{FT}, Y, p, params) where {FT}

Returns the snow depth given SWE, snow density ρ_snow, and
the density of liquid water ρ_l.
This can be extended for additional types of parameterizations.
"""
function snow_depth(density::AbstractDensityModel{FT}, Y, p, params) where {FT}
    ρ_l = FT(LP.ρ_cloud_liq(params.earth_param_set))
    return @. ρ_l * Y.snow.S / p.snow.ρ_snow
end


"""
    ClimaLand.surface_height(
        model::SnowModel{FT},
        Y,
        p,
    ) where {FT}

Returns the surface height of the `Snow` model; surface (ground) elevation
and ignores snow depth (CLM).

Once topography or land ice is incorporated, this will need to change to
z_sfc + land_ice_depth. Note that land ice can
be ~1-3 km thick on Greenland/

In order to compute surface fluxes, this cannot be larger than the
height of the atmosphere measurement location (z_atmos > z_land always).

This assumes that the surface elevation is zero.
"""
function ClimaLand.surface_height(model::SnowModel{FT}, Y, p) where {FT}
    return FT(0)
end


"""
    surface_albedo(model::SnowModel, Y, p)

A helper function which computes and returns the snow albedo.
"""
function ClimaLand.surface_albedo(model::SnowModel, _...)
    return model.parameters.α_snow
end

"""
    surface_emissivity(model::SnowModel, Y, p)

A helper function which computes and returns the snow emissivity.
"""
function ClimaLand.surface_emissivity(model::SnowModel, _...)
    return model.parameters.ϵ_snow
end

"""
    ClimaLand.surface_temperature(model::SnowModel, Y, p)

a helper function which returns the surface temperature for the snow
model, which is stored in the aux state.
"""
function ClimaLand.surface_temperature(model::SnowModel, Y, p, t)
    return p.snow.T_sfc
end


"""
    ClimaLand.surface_specific_humidity(model::SnowModel, Y, p)

Computes and returns the specific humidity over snow as a weighted
fraction of the saturated specific humidity over liquid and frozen
water.
"""
function ClimaLand.surface_specific_humidity(
    model::SnowModel,
    Y,
    p,
    T_sfc,
    ρ_sfc,
    )
    return p.snow.q_sfc
end

"""
    snow_surface_specific_humidity(T_sfc::FT, q_l::FT, atmos_ts, parameters) where {FT}
 
Computes the snow surface specific humidity at a point, assuming a weighted averaged (by mass fraction)
of the saturated specific humidity over ice and over liquid, at temperature T_sfc.

This approximates the surface air density using an adiabatic approximation and the current atmospheric state.
"""
function snow_surface_specific_humidity(
    T_sfc::FT,
    q_l::FT,
    atmos_ts,
    parameters,
) where {FT}
    thermo_params = LP.thermodynamic_parameters(parameters.earth_param_set)
    ρ_sfc = compute_ρ_sfc(thermo_params, atmos_ts, T_sfc)
    qsat_over_ice = Thermodynamics.q_vap_saturation_generic(
        thermo_params,
        T_sfc,
        ρ_sfc,
        Thermodynamics.Ice(),
    )
    qsat_over_liq = Thermodynamics.q_vap_saturation_generic(
        thermo_params,
        T_sfc,
        ρ_sfc,
        Thermodynamics.Liquid(),
    )
    return qsat_over_ice * (1 - q_l) + q_l * (qsat_over_liq)
end



"""
    snow_surface_temperature(T::FT) where {FT}

Returns the snow surface temperature assuming it is the same
as the bulk temperature T.
"""
snow_surface_temperature(T::FT) where {FT} = T


"""
    specific_heat_capacity(q_l::FT,
                           parameters::SnowParameters{FT}
                           ) where {FT}

Computes the specific heat capacity of the snow, neglecting
any contribution from air in the pore spaces, given
the liquid water mass fraction q_l and other parameters.
"""
function specific_heat_capacity(
    q_l::FT,
    parameters::SnowParameters{FT},
) where {FT}
    q_i = 1 - q_l
    _cp_i = FT(LP.cp_i(parameters.earth_param_set))

    _cp_l = FT(LP.cp_l(parameters.earth_param_set))

    cp_s = q_l * _cp_l + q_i * _cp_i
    return cp_s
end

"""
    snow_thermal_conductivity(ρ_snow::FT,
                         parameters::SnowParameters{FT},
                         ) where {FT}

Computes the thermal conductivity, given the density
of the snow, according to Equation
5.33 from Bonan's textbook, which in turn is taken from
Jordan (1991).

We have adjusted the original equation to make the coefficients
non-dimensional by multiplying by the first by x = ρ_ice/ρ_ice
and the second by x², with ρ_ice in kg/m³.

When ρ_snow = ρ_ice, we recover κ_snow = κ_ice.
"""
function snow_thermal_conductivity(
    ρ_snow::FT,
    parameters::SnowParameters{FT},
) where {FT}
    _κ_air = FT(LP.K_therm(parameters.earth_param_set))
    _ρ_ice = FT(LP.ρ_cloud_ice(parameters.earth_param_set))
    κ_ice = parameters.κ_ice
    return _κ_air +
           (FT(0.07) * (ρ_snow / _ρ_ice) + FT(0.93) * (ρ_snow / _ρ_ice)^2) *
           (κ_ice - _κ_air)
end

"""
    snow_bulk_temperature(U::FT,
                          S::FT,
                          S_l::FT,
                          parameters::SnowParameters{FT}) where {FT}

Computes the bulk snow temperature from the snow water equivalent S,
energy per unit area U, liquid water S_l, and specific heat
capacity c_s, along with other needed parameters.

If there is no snow (U = S = S_l = 0), the bulk temperature is the reference temperature,
which is 273.16K.
"""
function snow_bulk_temperature(
    U::FT,
    S::FT,
    S_l::FT,
    parameters::SnowParameters{FT},
) where {FT}
    S_safe = max(S, eps(FT))
    S_l_safe = min(max(S_l, eps(FT)), S_safe)
    q_l_safe = S_l_safe / S_safe
    _ρ_l = FT(LP.ρ_cloud_liq(parameters.earth_param_set))
    _T_ref = FT(LP.T_0(parameters.earth_param_set))
    _LH_f0 = FT(LP.LH_f0(parameters.earth_param_set))
    cp_s = specific_heat_capacity(q_l_safe, parameters)
    _ρcD_g = parameters.ρcD_g
    _T_freeze = FT(LP.T_freeze(parameters.earth_param_set))
    return _T_ref +
           (U + _LH_f0 * _ρ_l * (S_safe - S_l_safe)) /
           (_ρ_l * S_safe * cp_s + _ρcD_g)
end

"""
    snow_bulk_density(SWE::FT, z::FT, parameters::SnowParameters{FT}) where {FT}
Returns the snow density given the current model state when depth and SWE are available.
"""
function snow_bulk_density(
    SWE::FT,
    z::FT,
    parameters::SnowParameters{FT},
)::FT where {FT}
    ρ_l = FT(LP.ρ_cloud_liq(parameters.earth_param_set))
    ε = eps(FT) #for preventing dividing by zero
    #return SWE/z * ρ_l but tend to ρ_l as SWE → 0
    #also handle instabilities when z, SWE both near machine precision
    return max(SWE, ε) / max(z, SWE, ε) * ρ_l
end


"""
    maximum_liquid_mass_fraction(ρ_snow::FT, T::FT, parameters::SnowParameters{FT}) where {FT}

Computes the maximum liquid water mass fraction, given
the density of the snow ρ_snow and other parameters.
"""
function maximum_liquid_mass_fraction(
    ρ_snow::FT,
    T::FT,
    parameters::SnowParameters{FT},
) where {FT}
    _ρ_l = FT(LP.ρ_cloud_liq(parameters.earth_param_set))
    _T_freeze = FT(LP.T_freeze(parameters.earth_param_set))
    if T > _T_freeze
        return FT(0)
    else
        return parameters.θ_r * _ρ_l / ρ_snow
    end
end


"""
    runoff_timescale(z::FT, Ksat::FT, Δt::FT) where {FT}

Computes the timescale for liquid water to percolate and leave the snowpack,
given the depth of the snowpack z and the hydraulic conductivity Ksat.
"""
function runoff_timescale(z::FT, Ksat::FT, Δt::FT) where {FT}
    τ = max(Δt, z / Ksat)
    return τ
end

"""
    volumetric_internal_energy_liq(T::FT, parameters) where {FT}

Computes the volumetric internal energy of the liquid water
in the snowpack at temperature T.
"""
function volumetric_internal_energy_liq(T::FT, parameters) where {FT}
    _ρ_l = FT(LP.ρ_cloud_liq(parameters.earth_param_set))
    _T_freeze = FT(LP.T_freeze(parameters.earth_param_set))
    _cp_l = FT(LP.cp_l(parameters.earth_param_set))
    _T_ref = FT(LP.T_0(parameters.earth_param_set))

    I_liq = _ρ_l * _cp_l * (T .- _T_ref)
    return I_liq
end


"""
    compute_energy_runoff(S::FT, S_l::FT, T::FT, parameters) where {FT}

Computes the rate of change in the snow water equivalent S due to loss of
liquid water (runoff) from the snowpack.

Runoff occurs as the snow melts and exceeds the water holding capacity.
"""
function compute_water_runoff(
    S::FT,
    S_l::FT,
    T::FT,
    ρ_snow::FT,
    z::FT,
    parameters,
) where {FT}
    τ = runoff_timescale(z, parameters.Ksat, parameters.Δt)
    q_l_max::FT = maximum_liquid_mass_fraction(ρ_snow, T, parameters)
    S_safe = max(S, eps(FT))
    S_l_safe = min(max(S_l, eps(FT)), S_safe)
    return -(S_l_safe - q_l_max * S_safe) / τ *
           heaviside(S_l_safe - q_l_max * S_safe)
end

"""
     phase_change_flux(U::FT, S::FT, S_l::FT, energy_flux::FT, parameters) where {FT}

Computes the volume flux of liquid water undergoing phase change, given the 
applied energy flux and current state of U,S,S_l. 
"""
function phase_change_flux(
    U::FT,
    S::FT,
    S_l::FT,
    energy_flux::FT,
    parameters,
) where {FT}
    S_safe = max(S, eps(FT))
    S_l_safe = min(max(S_l, eps(FT)), S_safe)
    q_l_safe = S_l_safe / S_safe

    energy_at_T_freeze = energy_from_q_l_and_swe(S_safe, q_l_safe, parameters)
    Upred = U - energy_flux * parameters.Δt
    energy_excess = Upred - energy_at_T_freeze

    _LH_f0 = FT(LP.LH_f0(parameters.earth_param_set))
    _ρ_liq = FT(LP.ρ_cloud_liq(parameters.earth_param_set))
    _cp_i = FT(LP.cp_i(parameters.earth_param_set))
    _cp_l = FT(LP.cp_l(parameters.earth_param_set))
    _T_ref = FT(LP.T_0(parameters.earth_param_set))
    _T_freeze = FT(LP.T_freeze(parameters.earth_param_set))
    if energy_excess > 0 || (energy_excess < 0 && S_l > eps(FT))
        return -energy_excess / parameters.Δt / _ρ_liq /
               ((_cp_l - _cp_i) * (_T_freeze - _T_ref) + _LH_f0)
    else
        return FT(0)
    end
end

"""
    energy_from_q_l_and_swe(S::FT, q_l::FT, parameters) where {FT}

A helper function for compute the snow energy per unit area, given snow
water equivalent S, liquid fraction q_l, and snow model parameters.

This assumes that the snow is at the freezing point.
"""
function energy_from_q_l_and_swe(S::FT, q_l::FT, parameters) where {FT}
    _T_freeze = FT(LP.T_freeze(parameters.earth_param_set))
    _ρ_l = FT(LP.ρ_cloud_liq(parameters.earth_param_set))
    _T_ref = FT(LP.T_0(parameters.earth_param_set))
    _LH_f0 = FT(LP.LH_f0(parameters.earth_param_set))

    c_snow = specific_heat_capacity(q_l, parameters)
    _ρcD_g = parameters.ρcD_g
    return _ρ_l * S * (c_snow * (_T_freeze - _T_ref) - (1 - q_l) * _LH_f0) +
           _ρcD_g * (_T_freeze - _T_ref)
end

"""
    energy_from_T_and_swe(S::FT, T::FT, parameters) where {FT}

A helper function for compute the snow energy per unit area, given snow
water equivalent S, bulk temperature T, and snow model parameters.

The liquid mass fraction is assumed to be zero if T<=T_freeze, and 1 otherwise.
"""
function energy_from_T_and_swe(S::FT, T::FT, parameters) where {FT}
    _ρ_l = FT(LP.ρ_cloud_liq(parameters.earth_param_set))
    _T_ref = FT(LP.T_0(parameters.earth_param_set))
    _T_freeze = FT(LP.T_freeze(parameters.earth_param_set))
    _LH_f0 = FT(LP.LH_f0(parameters.earth_param_set))
    _cp_i = FT(LP.cp_i(parameters.earth_param_set))
    _cp_l = FT(LP.cp_l(parameters.earth_param_set))
    _ρcD_g = parameters.ρcD_g
    if T <= _T_freeze
        return (_ρ_l * S * _cp_i + _ρcD_g) * (T - _T_ref) - _ρ_l * S * _LH_f0
    else
        return (_ρ_l * S * _cp_l + _ρcD_g) * (T - _T_ref)
    end

end


"""
    update_density!(density::ConstantDryDensityModel, params::SnowParameters, Y, p)

Extends the update_density! function for the ConstantDryDensityModel type.
"""
function update_density!(
    density::ConstantDryDensityModel{FT},
    params::SnowParameters,
    Y,
    p,
) where {FT}
    _ρ_l = FT(LP.ρ_cloud_liq(params.earth_param_set))
    @. p.snow.ρ_snow = density.ρ_drysnow * (1 - p.snow.q_l) + _ρ_l * p.snow.q_l
end

"""
    update_density_prog!(density::AbstractDensityModel{FT}, model::SnowModel{FT}, Y, p) where {FT}
Updates all prognostic variables associated with density/depth given the current model state.
This is the default method for all density model types, which can be extended for alternative paramterizations.
"""
function update_density_prog!(
    density::AbstractDensityModel,
    model::SnowModel,
    dY,
    Y,
    p,
)
    return nothing
end
