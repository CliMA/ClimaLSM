using SurfaceFluxes
using StaticArrays
import SurfaceFluxes.Parameters as SFP
using RootSolvers

export snow_surface_temperature,
    snow_depth,
    specific_heat_capacity,
    snow_thermal_conductivity,
    snow_bulk_temperature,
    snow_liquid_mass_fraction,
    maximum_liquid_mass_fraction,
    runoff_timescale,
    compute_water_runoff,
    energy_from_q_l_and_swe,
    energy_from_T_and_swe,
    snow_cover_fraction

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
    ClimaLand.surface_height(
        model::SnowModel{FT},
        Y,
        p,
    ) where {FT}

Returns the surface height of the `Snow` model; this returns the depth
of the snow and hence implicitly treats the surface (ground) elevation as
at zero.

Once topography or land ice is incorporated, this will need to change to
z_sfc + land_ice_depth + snow_depth. Note that land ice can 
be ~1-3 km thick on Greenland/

In order to compute surface fluxes, this cannot be large than the 
height of the atmosphere measurement location (z_atmos > z_land always). 
"""
function ClimaLand.surface_height(model::SnowModel{FT}, Y, p) where {FT}
    return p.snow.z
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
function ClimaLand.surface_temperature(model::SnowModel{FT}, Y, p, t) where {FT}
    return p.snow.T_sfc
end

"""
    partial_q_sat_partial_T_ice(P::FT, T::FT) where {FT}
Computes the quantity ∂q_sat∂T at temperature T and pressure P,
over ice. The temperature must be in Celsius.
Uses the polynomial approximation from Flatau et al. (1992).
"""
function partial_q_sat_partial_T_ice(P::FT, T::FT) where {FT}
    T_celsius = T
    esat = FT(
        6.11123516e2 +
        5.03109514e1 .* T_celsius +
        1.88369801 * T_celsius^2 +
        4.20547422e-2 * T_celsius^3 +
        6.14396778e-4 * T_celsius^4 +
        6.02780717e-6 * T_celsius^5 +
        3.87940929e-8 * T_celsius^6 +
        1.49436277e-10 * T_celsius^7 +
        2.62655803e-13 * T_celsius^8,
    )
    desatdT = FT(
        5.03277922e1 +
        3.77289173 * T_celsius +
        1.26801703e-1 * T_celsius^2 +
        2.49468427e-3 * T_celsius^3 +
        3.13703411e-5 * T_celsius^4 +
        2.57180651e-7 * T_celsius^5 +
        1.33268878e-9 * T_celsius^6 +
        3.94116744e-12 * T_celsius^7 +
        4.98070196e-15 * T_celsius^8,
    )

    return FT(0.622) * P / (P - FT(0.378) * esat)^2 * desatdT
end

function get_qsfc(thermo_params, T_s::FT, ρ_sfc::FT) where {FT}
    if T_s <= 273.15
        q_sat =
            Thermodynamics.q_vap_saturation_generic.(
                Ref(thermo_params),
                T_s,
                ρ_sfc,
                Ref(Thermodynamics.Ice()),
            )
    else
        q_sat =
            Thermodynamics.q_vap_saturation_generic.(
                Ref(thermo_params),
                T_s,
                ρ_sfc,
                Ref(Thermodynamics.Liquid()),
            )
    end
    return q_sat
end

function update_conditions(
    T_s::FT,
    thermo_params,
    ρ_sfc::FT,
    state_air,
    z_0m::FT,
    z_0b::FT,
    surface_flux_params,
) where {FT}
    q_sfc = get_qsfc(thermo_params, T_s, ρ_sfc)
    thermal_state_sfc =
        Thermodynamics.PhaseEquil_ρTq(thermo_params, ρ_sfc, T_s, q_sfc)
    state_sfc = SurfaceFluxes.StateValues(
        FT(0),
        SVector{2, FT}(0, 0),
        thermal_state_sfc,
    )
    sc =
        SurfaceFluxes.ValuesOnly(state_air, state_sfc, z_0m, z_0b, beta = FT(1))

    conditions = SurfaceFluxes.surface_conditions(
        surface_flux_params,
        sc;
        tol_neutral = SFP.cp_d(surface_flux_params) / 100000,
    )

    E0 = SurfaceFluxes.evaporation(surface_flux_params, sc, conditions.Ch)
    r_ae = 1 / (conditions.Ch * SurfaceFluxes.windspeed(sc))
    return E0, r_ae
end

function solve_Ts(fns, T_initial::FT)::FT where {FT}
    sol = RootSolvers.find_zero(
        fns,
        RootSolvers.NewtonsMethod(T_initial),
        CompactSolution(),
        nothing,
        10,
    )
    T_s = sol.root
    return T_s
end

function calc_U(ρ_l::FT, d::FT, c::FT, T::FT, T0::FT)::FT where {FT} # assuming q_l = 1
    return ρ_l * d * c * (T - T0)
end

function snow_turbulent_fluxes_at_a_point(
    T_sfc::FT,
    h_sfc::FT,
    d_sfc::FT,
    thermal_state_air,
    u::FT,
    h::FT,
    gustiness::FT,
    z_0m::FT,
    z_0b::FT,
    earth_param_set::EP,
) where {FT <: AbstractFloat, EP}
    thermo_params = LP.thermodynamic_parameters(earth_param_set)
    surface_flux_params = LP.surface_fluxes_parameters(earth_param_set)
    _T_freeze = LP.T_freeze(earth_param_set)
    ρ_sfc = ClimaLand.compute_ρ_sfc(thermo_params, thermal_state_air, T_sfc)
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
    q_sfc = T_sfc >= _T_freeze ? qsat_over_liq : qsat_over_ice
    thermal_state_sfc =
        Thermodynamics.PhaseEquil_ρTq(thermo_params, ρ_sfc, T_sfc, q_sfc)
    ρ_air = Thermodynamics.air_density(thermo_params, thermal_state_air)
    _LH_v0::FT = LP.LH_v0(earth_param_set)
    _ρ_liq::FT = LP.ρ_cloud_liq(earth_param_set)
    cp_m_sfc::FT = Thermodynamics.cp_m(thermo_params, thermal_state_sfc)
    T_air = Thermodynamics.air_temperature(thermo_params, thermal_state_air)
    P_sfc = Thermodynamics.air_pressure(thermo_params, thermal_state_sfc)
    Rm_air = Thermodynamics.gas_constant_air(thermo_params, thermal_state_air)

    # SurfaceFluxes.jl expects a relative difference between where u = 0
    # and the atmosphere height. Here, we assume h and h_sfc are measured
    # relative to a common reference. Then d_sfc + h_sfc + z_0m is the apparent
    # source of momentum, and
    # Δh ≈ h - d_sfc - h_sfc is the relative height difference between the
    # apparent source of momentum and the atmosphere height.

    # In this we have neglected z_0m and z_0b (i.e. assumed they are small
    # compared to Δh).
    state_sfc = SurfaceFluxes.StateValues(
        FT(0),
        SVector{2, FT}(0, 0),
        thermal_state_sfc,
    )
    state_in = SurfaceFluxes.StateValues(
        h - d_sfc - h_sfc,
        SVector{2, FT}(u, 0),
        thermal_state_air,
    )

    # State containers
    states = SurfaceFluxes.ValuesOnly(
        state_in,
        state_sfc,
        z_0m,
        z_0b,
        gustiness = gustiness,
    )
    scheme = SurfaceFluxes.PointValueScheme()
    conditions =
        SurfaceFluxes.surface_conditions(surface_flux_params, states, scheme)

    # aerodynamic resistance
    r_ae::FT = 1 / (conditions.Ch * SurfaceFluxes.windspeed(states))

    # latent heat flux
    E0::FT =
        SurfaceFluxes.evaporation(surface_flux_params, states, conditions.Ch) # mass flux at potential evaporation rate
    LH = _LH_v0 * E0 # Latent heat flux

    # sensible heat flux
    SH = SurfaceFluxes.sensible_heat_flux(
        surface_flux_params,
        conditions.Ch,
        states,
        scheme,
    )

    # vapor flux in volume of liquid water with density 1000kg/m^3
    Ẽ = E0 / _ρ_liq

    # Derivatives
    # We ignore ∂r_ae/∂T_sfc, ∂u*/∂T_sfc
    ∂ρsfc∂T =
        ρ_air *
        (Thermodynamics.cv_m(thermo_params, thermal_state_air) / Rm_air) *
        (
            T_sfc / T_air
        )^(Thermodynamics.cv_m(thermo_params, thermal_state_air) / Rm_air - 1) /
        T_air
    ∂cp_m_sfc∂T = 0 # Possibly can address at a later date

    ∂LHF∂q = ρ_sfc * _LH_v0 / r_ae + LH / ρ_sfc * ∂ρsfc∂T

    ∂SHF∂T =
        ρ_sfc * cp_m_sfc / r_ae +
        SH / ρ_sfc * ∂ρsfc∂T +
        SH / cp_m_sfc * ∂cp_m_sfc∂T

    ∂q∂T = ClimaLand.Snow.partial_q_sat_partial_T_ice(P_sfc, T_sfc - _T_freeze)

    return (
        lhf = LH,
        shf = SH,
        vapor_flux = Ẽ,
        ∂LHF∂T = ∂LHF∂q * ∂q∂T,
        ∂SHF∂T = ∂SHF∂T,
    )
end

function kat_snow_surface_temperature(
    u_air::FT,
    thermal_state_air::Thermodynamics.PhaseEquil,
    h_air::FT,
    SWE::FT,
    T_bulk::FT,
    LW_d::FT,
    SW_d::FT,
    q_l::FT,
    t,
    parameters,
) where {FT}
    (; z_0m, z_0b, earth_param_set) = parameters
    thermo_params = LP.thermodynamic_parameters(earth_param_set)
    _ρ_liq::FT = LP.ρ_cloud_liq(earth_param_set)
    _σ = LP.Stefan(earth_param_set)
    _T_freeze = LP.T_freeze(earth_param_set)
    _T_ref::FT = FT(LP.T_0(earth_param_set))
    _LH_f0::FT = LP.LH_f0(earth_param_set)

    T_air::FT = Thermodynamics.air_temperature(thermo_params, thermal_state_air)
    ϵ_sfc::FT = parameters.ϵ_snow
    α_sfc::FT = parameters.α_snow
    d_sfc = FT(0)

    ρ_snow::FT = parameters.ρ_snow
    h_sfc::FT = snow_depth(SWE, ρ_snow, _ρ_liq)
    d = FT(0.1)
    if h_sfc < d
        return (T_s = T_bulk, ΔF = FT(0))
    end
    gustiness = FT(1)

    function sum_fluxes(T_sfc)
        turb_fluxes = snow_turbulent_fluxes_at_a_point(
            T_sfc,
            FT(0),#h_sfc,
            d_sfc,
            thermal_state_air,
            u_air,
            h_air,
            gustiness,
            z_0m,
            z_0b,
            earth_param_set,
        )
        F_R = -(1 - α_sfc) * SW_d - ϵ_sfc * (LW_d - _σ * T_sfc^4) # positive up
        F_h = turb_fluxes.shf + turb_fluxes.lhf + F_R # positive up
        κ = ClimaLand.Snow.snow_thermal_conductivity(
            parameters.ρ_snow,
            parameters,
        )
        f_total = κ * (T_sfc - T_bulk) / (h_sfc / 2) + F_h # Fh = - k(T_sfc - T_bulk)/(h/2)

        # Derivatives
        F_R_prime = 4 * ϵ_sfc * _σ * T_sfc^3
        F_h_prime = turb_fluxes.∂SHF∂T + turb_fluxes.∂LHF∂T + F_R_prime
        f_total_prime = (2 * κ / h_sfc) + F_h_prime
        return (f_total, f_total_prime)
    end
    T_s = solve_Ts(sum_fluxes, T_air)
    # adjust T_s and internal energy accordingly if T_s > freezing temperature
    if T_s > _T_freeze
        c_sfc = specific_heat_capacity(FT(1), parameters)
        c_bulk = specific_heat_capacity(q_l, parameters)
        U_Ts::FT = _ρ_liq * d * c_sfc * (T_s - _T_ref)#calc_U(_ρ_liq, d, c_sfc, T_s, _T0)
        U_Tf::FT =
            _ρ_liq * d * c_bulk * (_T_freeze - _T_ref) - (1 - q_l) * _LH_f0# calc_U(_ρ_liq, d, c_sfc, _T_freeze, _T0)
        ΔU::FT = U_Tf - U_Ts
        ΔF::FT = ΔU / parameters.Δt
        @show(ΔF)
        return (T_s = _T_freeze, ΔF = ΔF)
    else
        return (T_s = T_s, ΔF = FT(0))
    end
end
function snow_surface_temperature(
    u_air::FT,
    T_air::FT,
    P_air::FT,
    z_0m::FT,
    z_0b::FT,
    q_air::FT,
    h_air::FT,
    SWE::FT,
    T_bulk::FT,
    ρ_sfc::FT,
    energy_runoff::FT,
    LW_d::FT,
    SW_d::FT,
    parameters,
) where {FT}
    earth_param_set = parameters.earth_param_set
    _LH_v0::FT = LP.LH_v0(earth_param_set)
    _ρ_liq::FT = LP.ρ_cloud_liq(earth_param_set)
    ρ_snow::FT = parameters.ρ_snow
    h_sfc::FT = snow_depth(SWE, ρ_snow, _ρ_liq)
    d = FT(0.1)
    if h_sfc < d
        return (T_s = T_bulk, ΔF = FT(0))
    end
    thermo_params = LP.thermodynamic_parameters(earth_param_set)
    thermal_state_air =
        Thermodynamics.PhaseEquil_pTq(thermo_params, P_air, T_air, q_air)
    cp_m::FT = Thermodynamics.cp_m(thermo_params, thermal_state_air)
    ϵ_sfc::FT = parameters.ϵ_snow
    α_sfc::FT = parameters.α_snow
    d_sfc = FT(0)

    state_air = SurfaceFluxes.StateValues(
        h_air - d_sfc - h_sfc,
        SVector{2, FT}(u_air, 0),
        thermal_state_air,
    )
    surface_flux_params = LP.surface_fluxes_parameters(earth_param_set)
    function conditions_func(T_s)
        update_conditions(
            T_s,
            thermo_params,
            ρ_sfc,
            state_air,
            z_0m,
            z_0b,
            surface_flux_params,
        )
    end
    E0(T_s) = conditions_func(T_s)[1]
    r_ae(T_s) = conditions_func(T_s)[2]
    LH(T_s) = _LH_v0 * E0(T_s)

    ρ_air = Thermodynamics.air_density(thermo_params, thermal_state_air)
    SH(T_s) = cp_m * -ρ_air * (T_air - T_s) / r_ae(T_s)

    _σ = LP.Stefan(earth_param_set)
    F_R(T_s) = -(1 - α_sfc) * SW_d - ϵ_sfc * (LW_d - _σ * T_s^4)

    F_h(T_s) = SH(T_s) + LH(T_s) + F_R(T_s) + energy_runoff
    # κ = parameters.κ_ice
    κ = snow_thermal_conductivity(ρ_snow, parameters)
    f(T_s) = κ * (T_s - T_bulk) / (h_sfc / 2) + F_h(T_s)

    # derivatives for Newton's method
    SH_prime(T_s) = cp_m * ρ_air / r_ae(T_s)

    ∂LHF∂qc(T_s) = ρ_air * _LH_v0 / r_ae(T_s)
    ∂qc∂T(T_s) = partial_q_sat_partial_T_ice(P_air, T_s)
    LH_prime(T_s) = ∂LHF∂qc(T_s) * ∂qc∂T(T_s)

    F_R_prime(T_s) = 4 * ϵ_sfc * _σ * T_s^3

    F_h_prime(T_s) = SH_prime(T_s) + LH_prime(T_s) + F_R_prime(T_s)
    f_prime(T_s) = (2 * κ / h_sfc) + F_h_prime(T_s)

    fns(T_s) = [f(T_s), f_prime(T_s)]
    T_s::FT = solve_Ts(fns, T_bulk)

    # adjust T_s and internal energy accordingly if T_s > freezing temperature
    if T_s > 273.15
        T_s_new = FT(273.15)
        _T0::FT = FT(LP.T_0(earth_param_set))
        U_Ts::FT = calc_U(_ρ_liq, d, cp_m, T_s, _T0)
        U_Tf::FT = calc_U(_ρ_liq, d, cp_m, T_s_new, _T0)
        ΔU::FT = U_Ts - U_Tf
        ΔF::FT = ΔU / parameters.Δt
        return (T_s = T_s_new, ΔF = ΔF)
    elseif T_s < 250.0
        return (T_s = FT(250), ΔF = FT(0))
    else
        return (T_s = T_s, ΔF = FT(0))
    end
end


"""
    ClimaLand.surface_specific_humidity(model::BucketModel, Y, p)

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
    thermo_params =
        LP.thermodynamic_parameters(model.parameters.earth_param_set)
    qsat_over_ice =
        Thermodynamics.q_vap_saturation_generic.(
            Ref(thermo_params),
            T_sfc,
            ρ_sfc,
            Ref(Thermodynamics.Ice()),
        )
    qsat_over_liq =
        Thermodynamics.q_vap_saturation_generic.(
            Ref(thermo_params),
            T_sfc,
            ρ_sfc,
            Ref(Thermodynamics.Liquid()),
        )
    q_l = p.snow.q_l
    helper.(T_sfc, qsat_over_liq, qsat_over_ice)
    #return @. qsat_over_ice * (1 - q_l) + q_l * (qsat_over_liq)
end
helper(x, o1, o2) = x >= 273.15 ? o1 : o2


"""
    snow_surface_temperature(T::FT) where {FT}

Returns the snow surface temperature assuming it is the same
as the bulk temperature T.
"""
snow_surface_temperature_bulk(T::FT) where {FT} = T


"""
    snow_depth(SWE::FT, ρ_snow::FT, ρ_l::FT) where {FT}

Returns the snow depth given SWE, snow density ρ_snow, and
the density of liquid water ρ_l.

"""
function snow_depth(SWE::FT, ρ_snow::FT, ρ_l::FT)::FT where {FT}
    return SWE * ρ_l / ρ_snow
end



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
"""
function snow_thermal_conductivity(
    ρ_snow::FT,
    parameters::SnowParameters{FT},
) where {FT}
    _κ_air = FT(LP.K_therm(parameters.earth_param_set))
    κ_ice = parameters.κ_ice
    return _κ_air +
           (FT(7.75e-5) * ρ_snow + FT(1.105e-6) * ρ_snow^2) * (κ_ice - _κ_air)
end

"""
    snow_bulk_temperature(U::FT,
                          SWE::FT,
                          q_l::FT,
                          parameters::SnowParameters{FT}) where {FT}

Computes the bulk snow temperature from the snow water equivalent SWE,
energy per unit area U, liquid water mass fraction q_l, and specific heat
capacity c_s, along with other needed parameters.

If there is no snow (U = SWE = 0), the bulk temperature is the reference temperature,
which is 273.16K.
"""
function snow_bulk_temperature(
    U::FT,
    SWE::FT,
    q_l::FT,
    parameters::SnowParameters{FT},
) where {FT}
    _ρ_i = FT(LP.ρ_cloud_ice(parameters.earth_param_set))
    _ρ_l = FT(LP.ρ_cloud_liq(parameters.earth_param_set))
    _T_ref = FT(LP.T_0(parameters.earth_param_set))
    _LH_f0 = FT(LP.LH_f0(parameters.earth_param_set))
    cp_s = specific_heat_capacity(q_l, parameters)
    _T_freeze = FT(LP.T_freeze(parameters.earth_param_set))
    _ρcD_g = parameters.ρcD_g
    return _T_ref +
           (U + (1 - q_l) * _LH_f0 * _ρ_l * SWE) / (_ρ_l * SWE * cp_s + _ρcD_g)

end

"""
    snow_liquid_mass_fraction(U::FT, SWE::FT, parameters::SnowParameters{FT}) where {FT}

Computes the snow liquid water mass fraction, given the snow water equivalent SWE,
snow energy per unit area U, and other needed parameters.
"""
function snow_liquid_mass_fraction(
    U::FT,
    SWE::FT,
    parameters::SnowParameters{FT},
) where {FT}
    _ρ_i = FT(LP.ρ_cloud_ice(parameters.earth_param_set))
    _ρ_l = FT(LP.ρ_cloud_liq(parameters.earth_param_set))
    _T_ref = FT(LP.T_0(parameters.earth_param_set))
    _T_freeze = FT(LP.T_freeze(parameters.earth_param_set))
    _LH_f0 = FT(LP.LH_f0(parameters.earth_param_set))
    _cp_i = FT(LP.cp_i(parameters.earth_param_set))
    _cp_l = FT(LP.cp_l(parameters.earth_param_set))

    ΔT = _T_freeze - _T_ref

    _ρcD_g = parameters.ρcD_g
    Uminus =
        (_ρ_l * SWE * _cp_i + _ρcD_g) * (_T_freeze - _T_ref) -
        _ρ_l * SWE * _LH_f0
    Uplus = (_ρ_l * SWE * _cp_l + _ρcD_g) * (_T_freeze - _T_ref)
    if U < Uminus
        FT(0)
    elseif U > Uplus
        FT(1)
    else
        (U - Uminus) / max((Uplus - Uminus), eps(FT))
    end
end


"""
    maximum_liquid_mass_fraction(T::FT, ρ_snow::FT, parameters::SnowParameters{FT}) where {FT}

Computes the maximum liquid water mass fraction, given the bulk temperature of the snow T,
the density of the snow ρ_snow, and parameters.
"""
function maximum_liquid_mass_fraction(
    T::FT,
    ρ_snow::FT,
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
    volumetric_internal_energy_liq(FT, parameters)

Computes the volumetric internal energy of the liquid water
in the snowpack.

Since liquid water can only exist in the snowpack at
the freezing point, this is a constant.
"""
function volumetric_internal_energy_liq(FT, parameters)
    _ρ_l = FT(LP.ρ_cloud_liq(parameters.earth_param_set))
    _T_freeze = FT(LP.T_freeze(parameters.earth_param_set))
    _cp_l = FT(LP.cp_l(parameters.earth_param_set))
    _T_ref = FT(LP.T_0(parameters.earth_param_set))

    I_liq = _ρ_l * _cp_l * (_T_freeze .- _T_ref)
    return I_liq
end


"""
    compute_energy_runoff(S::FT, q_l::FT, T::FT, parameters) where {FT}

Computes the rate of change in the snow water equivalent S due to loss of
liquid water (runoff) from the snowpack.

Runoff occurs as the snow melts and exceeds the water holding capacity.
"""
function compute_water_runoff(S::FT, q_l::FT, T::FT, parameters) where {FT}
    _ρ_l = FT(LP.ρ_cloud_liq(parameters.earth_param_set))
    ρ_snow::FT = parameters.ρ_snow
    Ksat::FT = parameters.Ksat
    Δt::FT = parameters.Δt
    depth = snow_depth(S, ρ_snow, _ρ_l)
    τ = runoff_timescale(depth, Ksat, Δt)
    q_l_max::FT = maximum_liquid_mass_fraction(T, ρ_snow, parameters)
    return -(q_l - q_l_max) * heaviside(q_l - q_l_max) / τ * S /
           max(1 - q_l, FT(0.01))
end


"""
    energy_from_q_l_and_swe(S::FT, q_l::FT, parameters) where {FT}

A helper function for compute the snow energy per unit area, given snow 
water equivalent S, liquid fraction q_l,  and snow model parameters.

Note that liquid water can only exist at the freezing point in this model,
so temperature is not required as an input.
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

If T = T_freeze, we return the energy as if q_l = 0.

"""
function energy_from_T_and_swe(S::FT, T::FT, parameters) where {FT}
    _ρ_i = FT(LP.ρ_cloud_ice(parameters.earth_param_set))
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
        T > _T_freeze
        return (_ρ_l * S * _cp_l + _ρcD_g) * (T - _T_ref)
    end

end
