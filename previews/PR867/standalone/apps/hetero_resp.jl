using Unitful: K, °C, mol, μmol, m, s, kg, J

function ParamViz.parameterisation(
    Tₛ,
    θ, # drivers
    ν,
    α_sx,
    Ea_sx,
    kM_sx,
    kM_O₂,
    O₂_a,
    p_sx,
    Csom, # parameters
    θ_a100,
    D_liq,
    D_oa,
) # constants

    params = SoilCO2ModelParameters(FT; ν)
    # θ has to be lower than porosity
    if θ > ν
        θ = ν
    end

    Rh = microbe_source(Tₛ, θ, Csom, params)
    return Rh
end

function Rh_app_f()
    drivers = Drivers(
        ("Tₛ (°C)", "θ (m³ m⁻³)"), # drivers.names
        (FT.([273, 323]), FT.([0.0, 1.0])), # drivers.ranges 
        ((K, °C), (m^3 * m^-3, m^3 * m^-3)),
    )

    parameters = Parameters(
        (# names   
            "Soil porosity, ν (m³ m⁻³)",
            "Pre-exponential factor, α_sx (kg C m⁻³ s⁻¹)",
            "Activation energy, Ea_sx (J mol⁻¹)",
            "Michaelis constant for soil, kM_sx (kg C m⁻³)",
            "Michaelis constant for O₂, kM_O₂ (m³ m⁻³)",
            "Volumetric fraction of O₂ in the soil air, O₂_a (dimensionless)",
            "Fraction of soil carbon that is considered soluble, p_sx (dimensionless)",
            "Soil organic C, Csom (kg C m⁻³)",
        ),
        (# ranges
            FT.([0.0, 1.0]), # porosity
            FT.([100e3, 300e3]), # α_sx
            FT.([50e3, 70e3]), # Ea_sx
            FT.([1e-10, 0.1]), # kM_sx
            FT.([1e-10, 0.1]), # kM_o2
            FT.([0.005, 0.5]), # O2_a
            FT.([0.005, 0.5]), # p_sx
            FT.([1.0, 10.0]), # Csom
        ),
        (
            (m^3 * m^-3, m^3 * m^-3), # porosity
            (kg * m^-3 * s^-1, kg * m^-3 * s^-1), # α_sx
            (J * mol^-1, J * mol^-1), # Ea_sx
            (kg * m^-3, kg * m^-3), # kM_sx
            (m^3 * m^-3, m^3 * m^-3), # kM_O2
            (m, m), # O2_a
            (m, m), # p_sx
            (kg * m^-3, kg * m^-3), # Csom
        ),
    )

    constants = Constants(
        (# names
            "Air-filled porosity at soil water potential of -100 cm H₂O (~ 10 Pa)",
            "Diffusivity of soil C substrate in liquid (unitless)",
            "Diffusion coefficient of oxygen in air, dimensionless",
        ),
        (# values
            FT(0.1816), # θ_a100
            FT(3.17), # D_liq
            FT(1.67), # D_oa
        ),
    )

    inputs = Inputs(drivers, parameters, constants)

    output = Output(
        "Rh (mg C m⁻³ s⁻¹)", # name #NEED TO CONVERT THE UNIT
        [0, 20 * 1e-6], # range
        (mol * m^-2 * s^-1, μmol * m^-2 * s^-1), # unit from, unit to --> actually need to fix this, should be g to mg C m-3 s-1
    )

    Rh_app = webapp(ParamViz.parameterisation, inputs, output)
    return Rh_app
end
