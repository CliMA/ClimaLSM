using Unitful: K, °C, mol, μmol, m, s, Pa, kPa

function ParamViz.parameterisation(
    PAR,
    T,
    β,
    LAI,
    ca,
    VPD,
    θs,
    ld,
    ρ_leaf,
    Ω,
    Γstar25,
    ΔHΓstar,
    To,
    R,
    Vcmax25,
    ΔHJmax,
    θj,
    ϕ,
    ΔHVcmax,
    g1,
    Kc25,
    ΔHkc,
    Ko25,
    ΔHko,
    oi,
    f,
    ΔHRd,
)
    K = extinction_coeff(RTparams.G_Function, θs)
    # APAR = plant_absorbed_ppfd(PAR, ρ_leaf, K, LAI, Ω)
    APAR = PAR * (1 - ρ_leaf) * (1 - exp(-K * LAI * Ω)) # not sure how to call new plant_absorbed_pfd
    Jmax = max_electron_transport(Vcmax25, ΔHJmax, T, To, R)
    J = electron_transport(APAR, Jmax, θj, ϕ)
    Vcmax = compute_Vcmax(Vcmax25, T, To, R, ΔHVcmax)
    Γstar = co2_compensation(Γstar25, ΔHΓstar, T, To, R)
    medlynt = 1 + g1 / sqrt(VPD)
    ci = intercellular_co2(ca, Γstar, medlynt)
    Aj = light_assimilation(Canopy.C3(), J, ci, Γstar)
    Kc = MM_Kc(Kc25, ΔHkc, T, To, R)
    Ko = MM_Ko(Ko25, ΔHko, T, To, R)
    Ac = rubisco_assimilation(Canopy.C3(), Vcmax, ci, Γstar, Kc, Ko, oi)
    Rd = dark_respiration(Vcmax25, β, f, ΔHRd, T, To, R)
    An = net_photosynthesis(Ac, Aj, Rd, β)
    return An
end

function An_app_f()
    drivers = Drivers(
        ("PAR (μmol m⁻² s⁻¹)", "T (°C)"),
        (FT.([0, 1500 * 1e-6]), FT.([273, 323])),
        ((mol * m^-2 * s^-1, μmol * m^-2 * s^-1), (K, °C)),
    )

    parameters = Parameters(
        (
            "Moisture stress, β",
            "Leaf area index, LAI (m² m⁻²)",
            "CO2 concentration, ca (ppm)",
            "Vapor pressure deficit, VPD (Pa)",
        ),
        (
            FT.([0, 1]), # β
            FT.([1, 10]), # LAI, m2 m-2
            FT.([300 * 1e-6, 500 * 1e-6]), # ca
            FT.([500, 10000]), # VPD, Pa
        ),
        (
            (m, m), # dummy units, no conversion
            (m^2 * m^-2, m^2 * m^-2),
            (mol / mol, μmol / mol),
            (Pa, kPa),
        ),
    )

    constants = Constants(
        (
            "θs", # Sun zenith angle
            "ld", # Leaf angle distribution
            "ρ_leaf", # PAR canopy reflectance
            "Ω", # Clumping index
            "Γstar25", # co2 compensation at 25c
            "ΔHΓstar", # a constant energy of activation
            "To", # a standard temperature 
            "R", # the universal gas constant
            "Vcmax25", # the maximum rate of carboxylation of Rubisco
            "ΔHJmax", # a constant
            "θj", # an empirical "curvature parameter"
            "ϕ", # the quantum yield of photosystem II
            "ΔHVcmax", # a constant
            "g1", # a constant
            "Kc25", # a constant
            "ΔHkc", # a constant
            "Ko25", # a constant
            "ΔHko", # a constant
            "oi", # an empirical parameter
            "f", # an empirical factor
            "ΔHRd",
        ),
        (
            FT(0.6), # θs - is that a good value? in radian, right? --------
            FT(0.5), # ld - ozark val
            FT(0.1), # ρ_leaf - ozark val
            FT(0.69), # Ω - ozark val
            FT(4.275e-5), # Γstar25 - ozark val
            FT(37830), # ΔHΓstar - ozark val
            FT(298.15), # To - ozark vak
            FT(8.314), # R, J/mol - FT(LSMP.gas_constant(earth_param_set))
            FT(5e-5), # Vcmax25 - ozark model
            FT(43540), # ΔHJmax - ozark model
            FT(0.9), # θj - ozark model
            FT(0.6), # ϕ - ozark model
            FT(58520), # ΔHVcmax - ozark model
            FT(141), # g1 - ozark model
            FT(4.049e-4), # Kc25 - ozark model
            FT(79430), # ΔHkc - ozark model
            FT(0.2874), # Ko25 - ozark model
            FT(36380), # ΔHko - ozark model
            FT(0.209), # oi - ozark model
            FT(0.015), # f - ozark model
            FT(43390), # ΔHRd - ozark model
        ),
    )

    inputs = Inputs(drivers, parameters, constants)

    output = Output(
        "An (μmol m⁻² s⁻¹)",
        [0, 20 * 1e-6],
        (mol * m^-2 * s^-1, μmol * m^-2 * s^-1),
    )

    An_app = webapp(ParamViz.parameterisation, inputs, output)
    return An_app
end
