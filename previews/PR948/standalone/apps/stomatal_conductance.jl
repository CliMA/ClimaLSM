using Unitful: Pa, m, μmol, mol, s

function ParamViz.parameterisation(VPD, ca, An, g0, g1)
    Drel = 1.6
    medyn_term_val = 1 + g1 / sqrt(VPD)
    gs = medlyn_conductance(g0, Drel, medlyn_term_val, An, ca)
    return gs
end

function Medlyn_app_f()
    drivers = Drivers(
                      ("VPD (kPa)", "ca (μmol m⁻² s⁻¹)"),
                      (FT.([0, 12]), FT.([200, 800])),
                      ((Pa, kPa), (, )),
                     )

    parameters = Parameters(
        (
            "leaf photosynthesis, An",
            "",
            "",
        ),
        (
         FT.([0, 20]),
         FT.([]),
         FT.([]),
         FT.([]),
        ),
        (
         (μmol * m^-2 * s^-1, μmol * m^-2 * s^-1),
         (),
         (),
         (),
        ), # dummy units, no conversion
    )

    constants = Constants(("a", "b"), (FT(1), FT(2))) # dummy constants
    inputs = Inputs(drivers, parameters, constants)
    output = Output(
        "gs (m s⁻¹)",
        [0, ],
        (m * s^-1, m * s^-1),
    )
    medlyn_app = webapp(ParamViz.parameterisation, inputs, output)
    return medlyn_app
end
