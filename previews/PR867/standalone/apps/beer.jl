using Unitful: m, s, mol, μmol

function ParamViz.parameterisation(PAR, LAI, ρ_leaf, K, Ω, a, b)
    # APAR = plant_absorbed_ppfd(PAR, ρ_leaf, K, LAI, Ω) 
    APAR = PAR * (1 - ρ_leaf) * (1 - exp(-K * LAI * Ω)) # not sure how to call new plant_absorbed_pfd
    return APAR
end

function Beer_app_f()
    drivers = Drivers(
        ("PAR (μmol m⁻² s⁻¹)", "LAI (m² m⁻²)"),
        (FT.([0, 1500 * 1e-6]), FT.([0, 10])),
        ((mol * m^-2 * s^-1, μmol * m^-2 * s^-1), (m^2 * m^-2, m^2 * m^-2)),
    )

    parameters = Parameters(
        (
            "canopy reflectance, ρ_leaf",
            "extinction coefficient, K",
            "clumping index, Ω",
        ),
        (FT.([0, 1]), FT.([0, 1]), FT.([0, 1])),
        ((m, m), (m, m), (m, m)), # dummy units, no conversion
    )

    constants = Constants(("a", "b"), (FT(1), FT(2))) # dummy constants
    inputs = Inputs(drivers, parameters, constants)
    output = Output(
        "APAR (μmol m⁻² s⁻¹)",
        [0, 1500 * 1e-6],
        (mol * m^-2 * s^-1, μmol * m^-2 * s^-1),
    )
    beer_app = webapp(ParamViz.parameterisation, inputs, output)
    return beer_app
end
