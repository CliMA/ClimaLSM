export plant_absorbed_ppfd,
    extinction_coeff,
    arrhenius_function,
    intercellular_co2,
    co2_compensation,
    rubisco_assimilation,
    light_assimilation,
    max_electron_transport,
    electron_transport,
    net_photosynthesis,
    moisture_stress,
    dark_respiration,
    compute_GPP,
    MM_Kc,
    MM_Ko,
    compute_Vcmax,
    medlyn_term,
    medlyn_conductance,
    canopy_surface_fluxes,
    upscale_leaf_conductance,
    penman_monteith

# 1. Radiative transfer

"""
    plant_absorbed_ppfd(PAR::FT,
                       ρ_leaf::FT,
                       K::FT,
                       LAI::FT,
                       Ω::FT) where {FT}

Computes the absorbed photosynthetically active radiation in terms 
of mol photons per m^2 per second (`APAR`).

This assumes the Beer-Lambert law, which is a function of photosynthetically
active radiation (`PAR`; moles of photons/m^2/),
PAR canopy reflectance (`ρ_leaf`), the extinction
coefficient (`K`), leaf area index (`LAI`) and the clumping index (`Ω`).
"""
function plant_absorbed_ppfd(
    PAR::FT,
    ρ_leaf::FT,
    K::FT,
    LAI::FT,
    Ω::FT,
) where {FT}
    APAR = PAR * (1 - ρ_leaf) * (1 - exp(-K * LAI * Ω))
    return APAR
end

"""
    extinction_coeff(ld::FT,
                     θs::FT) where {FT}

Computes the vegetation extinction coefficient (`K`), as a function
of the sun zenith angle (`θs`), and the leaf angle distribution (`ld`).
"""
function extinction_coeff(ld::FT, θs::FT) where {FT}
    K = ld / max(cos(θs), eps(FT))
    return K
end

# 2. Photosynthesis, Farquhar model


"""
    intercellular_co2(ca::FT, Γstar::FT, medlyn_factor::FT) where{FT}

Computes the intercellular CO2 concentration (mol/mol) given the
atmospheric concentration (`ca`, mol/mol), the CO2 compensation (`Γstar`, 
 mol/mol), and the Medlyn factor (unitless).
"""
function intercellular_co2(ca::FT, Γstar::FT, medlyn_term::FT) where {FT}
    c_i = max(ca * (1 - 1 / medlyn_term), Γstar)
    return c_i
end

"""
    co2_compensation(Γstar25::FT,
                     ΔHΓstar::FT,
                     T::FT,
                     To::FT,
                     R::FT) where {FT}

Computes the CO2 compensation point (`Γstar`),
in units of mol/mol,
as a function of its value at 25 °C (`Γstar25`),
a constant energy of activation (`ΔHΓstar`), a standard temperature (`To`),
the unversal gas constant (`R`), and the temperature (`T`).

See Table 11.5 of G. Bonan's textbook, Climate Change and Terrestrial Ecosystem Modeling (2019).
"""
function co2_compensation(
    Γstar25::FT,
    ΔHΓstar::FT,
    T::FT,
    To::FT,
    R::FT,
) where {FT}
    Γstar = Γstar25 * arrhenius_function(T, To, R, ΔHΓstar)
    return Γstar
end

"""
    rubisco_assimilation(::C3,
                         Vcmax::FT,
                         ci::FT,
                         Γstar::FT,
                         Kc::FT,
                         Ko::FT,
                         oi::FT) where {FT}

Computes the Rubisco limiting rate of photosynthesis for C3 plants (`Ac`),
in units of moles CO2/m^2/s,
as a function of the maximum rate of carboxylation of Rubisco (`Vcmax`), 
the leaf internal carbon dioxide partial pressure (`ci`), 
the CO2 compensation point (`Γstar`), and Michaelis-Menten parameters
for CO2 and O2, respectively, (`Kc`) and (`Ko`).

The empirical parameter oi is equal to 0.209 (mol/mol).
See Table 11.5 of G. Bonan's textbook, Climate Change and Terrestrial Ecosystem Modeling (2019).
"""
function rubisco_assimilation(
    ::C3,
    Vcmax::FT,
    ci::FT,
    Γstar::FT,
    Kc::FT,
    Ko::FT,
    oi::FT,
) where {FT}
    Ac = Vcmax * (ci - Γstar) / (ci + Kc * (1 + oi / Ko))
    return Ac
end

"""
    rubisco_assimilation(::C4, Vcmax::FT,_...) where {FT}

Computes the Rubisco limiting rate of photosynthesis for C4 plants (`Ac`)
in units of moles CO2/m^2/s,
as equal to the maximum rate of carboxylation of Rubisco (`Vcmax`).
"""
function rubisco_assimilation(::C4, Vcmax::FT, _...) where {FT}
    Ac = Vcmax
    return Ac
end

"""
    light_assimilation(::C3,
                       J::FT,
                       ci::FT,
                       Γstar::FT) where {FT}

Computes the electron transport limiting rate (`Aj`),
in units of moles CO2/m^2/s, for C3 plants as a function of
the rate of electron transport (`J`), the leaf internal carbon dioxide partial pressure (`ci`),
and the CO2 compensation point (`Γstar`).

See Table 11.5 of G. Bonan's textbook, Climate Change and Terrestrial Ecosystem Modeling (2019).
"""
function light_assimilation(::C3, J::FT, ci::FT, Γstar::FT) where {FT}
    Aj = J * (ci - Γstar) / (4 * (ci + 2 * Γstar))
    return Aj
end

"""
    light_assimilation(::C4, J::FT, _...) where {FT}

Computes the electron transport limiting rate (`Aj`),
in units of moles CO2/m^2/s, for C4 plants, as equal to
the rate of electron transport (`J`).
"""
function light_assimilation(::C4, J::FT, _...) where {FT}
    Aj = J
    return Aj
end

"""
    max_electron_transport(Vcmax::FT) where {FT}

Computes the maximum potential rate of electron transport (`Jmax`),
in units of mol/m^2/s, 
as a function of Vcmax at 25 °C (`Vcmax25`),
a constant (`ΔHJmax`), a standard temperature (`To`),
the unversal gas constant (`R`), and the temperature (`T`).

See Table 11.5 of G. Bonan's textbook, 
Climate Change and Terrestrial Ecosystem Modeling (2019).
"""
function max_electron_transport(
    Vcmax25::FT,
    ΔHJmax::FT,
    T::FT,
    To::FT,
    R::FT,
) where {FT}
    Jmax25 = Vcmax25 * FT(exp(1))
    Jmax = Jmax25 * arrhenius_function(T, To, R, ΔHJmax)
    return Jmax
end

"""
    electron_transport(APAR::FT,
                       Jmax::FT,
                       θj::FT,
                       ϕ::FT) where {FT}

Computes the rate of electron transport (`J`),
in units of mol/m^2/s, as a function of
the maximum potential rate of electron transport (`Jmax`),
absorbed photosynthetically active radiation (`APAR`),
an empirical "curvature parameter" (`θj`; Bonan Eqn 11.21)
and the quantum yield of photosystem II (`ϕ`). 

See Ch 11, G. Bonan's textbook, Climate Change and Terrestrial Ecosystem Modeling (2019).
"""
function electron_transport(APAR::FT, Jmax::FT, θj::FT, ϕ::FT) where {FT}
    # Light utilization of APAR
    IPSII = ϕ * APAR / 2
    # This is a solution to a quadratic equation
    # θj *J^2 - (IPSII+Jmax)*J+IPSII*Jmax = 0, Equation 11.21
    J =
        (IPSII + Jmax - sqrt((IPSII + Jmax)^2 - 4 * θj * IPSII * Jmax)) /
        (2 * θj)
    return J
end


"""
    net_photosynthesis(Ac::FT,
                       Aj::FT,
                       Rd::FT,
                       β::FT) where {FT}

Computes the total net carbon assimilation (`An`),
in units of mol CO2/m^2/s, as a function of 
the Rubisco limiting factor (`Ac`), the electron transport limiting rate (`Aj`),
dark respiration (`Rd`), and the moisture stress factor (`β`). 

See Table 11.5 of G. Bonan's textbook, Climate Change and Terrestrial Ecosystem Modeling (2019).
"""
function net_photosynthesis(Ac::FT, Aj::FT, Rd::FT, β::FT) where {FT}
    An = max(0, min(Ac * β, Aj) - Rd)
    return An
end

"""
    moisture_stress(pl::FT,
                    sc::FT,
                    pc::FT) where {FT}

Computes the moisture stress factor (`β`), which is unitless,
 as a function of
a constant (`sc`, 1/Pa), a reference pressure (`pc`, Pa), and 
the leaf water pressure (`pl`, Pa) . 

See Eqn 12.57 of G. Bonan's textbook, 
Climate Change and Terrestrial Ecosystem Modeling (2019).
"""
function moisture_stress(pl::FT, sc::FT, pc::FT) where {FT}
    β = (1 + exp(sc * pc)) / (1 + exp(sc * (pc - pl)))
    return β
end

"""
    dark_respiration(Vcmax25::FT,
                     β::FT,
                     f::FT,
                     ΔHkc::FT,
                     T::FT,
                     To::FT,
                     R::FT) where {FT}

Computes dark respiration (`Rd`),
in units of mol CO2/m^2/s, as a function of the maximum rate of carboxylation of Rubisco (`Vcmax25`),
and the moisture stress factor (`β`), an empirical factor `f` is equal to 0.015,
a constant (`ΔHRd`), a standard temperature (`To`),
the unversal gas constant (`R`), and the temperature (`T`).

See Table 11.5 of G. Bonan's textbook, 
Climate Change and Terrestrial Ecosystem Modeling (2019).
"""
function dark_respiration(
    Vcmax25::FT,
    β::FT,
    f::FT,
    ΔHRd::FT,
    T::FT,
    To::FT,
    R::FT,
) where {FT}
    Rd = f * Vcmax25 * β * arrhenius_function(T, To, R, ΔHRd)
    return Rd
end

"""
    compute_GPP(An::FT,
             K::FT,
             LAI::FT,
             Ω::FT) where {FT}

Computes the total canopy photosynthesis (`GPP`) as a function of 
the total net carbon assimilation (`An`), the extinction coefficient (`K`),
leaf area index (`LAI`) and the clumping index (`Ω`).
"""
function compute_GPP(An::FT, K::FT, LAI::FT, Ω::FT) where {FT}
    GPP = An * (1 - exp(-K * LAI * Ω)) / K
    return GPP
end


"""
    upscale_leaf_conductance(gs::FT, LAI::FT, T::FT, R::FT, P::FT) where {FT}

This currently takes a leaf conductance (moles per leaf area per second)
and (1) converts it to m/s, (2) upscales to the entire canopy, by assuming
the leaves in the canopy are in parallel and hence multiplying
by LAI.

TODO: Check what CLM does, and check if we can use the same function 
for GPP from An, and make more general.
"""
function upscale_leaf_conductance(
    gs::FT,
    LAI::FT,
    T::FT,
    R::FT,
    P::FT,
) where {FT}
    canopy_conductance = gs * LAI * (R * T) / P # convert to m s-1
    return canopy_conductance
end

"""
    arrhenius_function(T::FT, To::FT, R::FT, ΔH::FT)

Computes the Arrhenius function at temperature `T` given
the reference temperature `To=298.15K`, the universal 
gas constant `R`, and the energy activation `ΔH`.

See Table 11.5 of G. Bonan's textbook, 
Climate Change and Terrestrial Ecosystem Modeling (2019).
"""
function arrhenius_function(T::FT, To::FT, R::FT, ΔH::FT) where {FT}
    return exp(ΔH * (T - To) / (To * R * T))
end

"""
    MM_Kc(Kc25::FT,
          ΔHkc::FT,
          T::FT,
          To::FT,
          R::FT) where {FT}

Computes the Michaelis-Menten coefficient for CO2 (`Kc`),
in units of mol/mol,
as a function of its value at 25 °C (`Kc25`),
a constant (`ΔHkc`), a standard temperature (`To`),
the unversal gas constant (`R`), and the temperature (`T`).

See Table 11.5 of G. Bonan's textbook, 
Climate Change and Terrestrial Ecosystem Modeling (2019).
"""
function MM_Kc(Kc25::FT, ΔHkc::FT, T::FT, To::FT, R::FT) where {FT}
    Kc = Kc25 * arrhenius_function(T, To, R, ΔHkc)
    return Kc
end

"""
    MM_Ko(Ko25::FT,
          ΔHko::FT,
          T::FT,
          To::FT,
          R::FT) where {FT}

Computes the Michaelis-Menten coefficient for O2 (`Ko`),
in units of mol/mol,
as a function of its value at 25 °C (`Ko25`),
a constant (`ΔHko`), a standard temperature (`To`),
the universal gas constant (`R`), and the temperature (`T`).

See Table 11.5 of G. Bonan's textbook, 
Climate Change and Terrestrial Ecosystem Modeling (2019).
"""
function MM_Ko(Ko25::FT, ΔHko::FT, T::FT, To::FT, R::FT) where {FT}
    Ko = Ko25 * arrhenius_function(T, To, R, ΔHko)
    return Ko
end

"""
    compute_Vcmax(Vcmax25::FT,
           T::FT,
           To::FT,
           R::FT,
           ep5::FT) where {FT}

Computes the maximum rate of carboxylation of Rubisco (`Vcmax`),
in units of mol/m^2/s, 
as a function of temperature (`T`), Vcmax at the reference temperature 25 °C (`Vcmax25`),
the universal gas constant (`R`), and the reference temperature (`To`).

See Table 11.5 of G. Bonan's textbook, 
Climate Change and Terrestrial Ecosystem Modeling (2019).
"""
function compute_Vcmax(
    Vcmax25::FT,
    T::FT,
    To::FT,
    R::FT,
    ΔHVcmax::FT,
) where {FT}
    Vcmax = Vcmax25 * arrhenius_function(T, To, R, ΔHVcmax)
    return Vcmax
end

# 3. Stomatal conductance model
"""
    medlyn_term(g1::FT, T_air::FT, P_air::FT, q_air::FT, thermo_params) where {FT}

Computes the Medlyn term, equal to `1+g1/sqrt(VPD)`,
by first computing the `VPD`,
where `VPD` is the vapor pressure deficit in the atmosphere
(Pa), and `g_1` is a constant with units of `sqrt(Pa)`.

`thermo_params` is the Thermodynamics.jl parameter set.
"""
function medlyn_term(
    g1::FT,
    T_air::FT,
    P_air::FT,
    q_air::FT,
    thermo_params,
) where {FT}
    VPD = ClimaLSM.vapor_pressure_deficit(T_air, P_air, q_air, thermo_params)
    return 1 + g1 / sqrt(VPD)
end


"""
    medlyn_conductance(g0::FT,
                       Drel::FT,
                       medlyn_term::FT,
                       An::FT,
                       ca::FT) where {FT}

Computes the stomatal conductance according to Medlyn, as a function of 
the minimum stomatal conductance (`g0`), 
the relative diffusivity of water vapor with respect to CO2 (`Drel`),
the Medlyn term (unitless), the biochemical demand for CO2 (`An`), and the
atmospheric concentration of CO2 (`ca`).

This returns the conductance in units of mol/m^2/s. It must be converted to 
m/s using the molar density of water prior to use in SurfaceFluxes.jl.
"""
function medlyn_conductance(
    g0::FT,
    Drel::FT,
    medlyn_term::FT,
    An::FT,
    ca::FT,
) where {FT}
    gs = g0 + Drel * medlyn_term * (An / ca)
    return gs
end

"""
    penman_monteith(
        Δ::FT, # Rate of change of saturation vapor pressure with air temperature. (Pa K−1)  
        Rn::FT, # Net irradiance (W m−2)
        G::FT, # Ground heat flux (W m−2)
        ρa::FT, # Dry air density (kg m−3)
        cp::FT, # Specific heat capacity of air (J kg−1 K−1) 
        VPD::FT, # vapor pressure deficit (Pa)
        ga::FT, # atmospheric conductance (m s−1)
        γ::FT, # Psychrometric constant (γ ≈ 66 Pa K−1)
        gs::FT, # surface or stomatal conductance (m s−1)
        Lv::FT, # Volumetric latent heat of vaporization (J m-3)
        ) where {FT}

Computes the evapotranspiration in m/s using the Penman-Monteith equation.       
"""
function penman_monteith(
    Δ::FT,
    Rn::FT,
    G::FT,
    ρa::FT,
    cp::FT,
    VPD::FT,
    ga::FT,
    γ::FT,
    gs::FT,
    Lv::FT,
) where {FT}
    ET = (Δ * (Rn - G) + ρa * cp * VPD * ga) / ((Δ + γ * (1 + ga / gs)) * Lv)
    return ET
end
