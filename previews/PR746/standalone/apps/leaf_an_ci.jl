using Unitful: K, °C, mol, μmol, m, s

function ParamViz.parameterisation(
    ci,
    T,
    β,
    LAI,
    PAR,
    Vcmax25,
    θs,
    ld,
    ρ_leaf,
    Ω,
    Γstar25,
    ΔHΓstar,
    To,
    R,
    ΔHJmax,
    θj,
    ϕ,
    ΔHVcmax,
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
    Kc = MM_Kc(Kc25, ΔHkc, T, To, R)
    Ko = MM_Ko(Ko25, ΔHko, T, To, R)
    is_c3 = 1.0
    Ac = rubisco_assimilation(is_c3, Vcmax, ci, Γstar, Kc, Ko, oi)
    Aj = light_assimilation(is_c3, J, ci, Γstar)
    Rd = dark_respiration(Vcmax25, β, f, ΔHRd, T, To, R)
    An = net_photosynthesis(Ac, Aj, Rd, β)
    return An
end

function An_ci_app_f()
    drivers = Drivers(
        ("ci (ppm)", "T (°C)"),  # names
        (FT.([0.0001, 0.001]), FT.([273, 323])), # ranges
        ((mol / mol, μmol / mol), (K, °C)), # units from, unit to
    )

    parameters = Parameters(
        (
            "Moisture stress, β",
            "Leaf area index, LAI (m² m⁻²)",
            "Photosynthetic radiation, PAR (μmol m⁻² s⁻¹)",
            "Vcmax25, (μmol m⁻² s⁻¹)",
        ),
        (
            FT.([0, 1]), # β
            FT.([1, 10]), # LAI, m2 m-2
            FT.([0, 1500 * 1e-6]), # PAR
            FT.([1e-5, 1e-4]), # Vcmax25
        ),
        (
            (m, m), # dummy unit, no conversion
            (m^2 * m^-2, m^2 * m^-2),
            (mol * m^-2 * s^-1, μmol * m^-2 * s^-1),
            (mol * m^-2 * s^-1, μmol * m^-2 * s^-1),
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
            "ΔHJmax", # a constant
            "θj", # an empirical "curvature parameter"
            "ϕ", # the quantum yield of photosystem II
            "ΔHVcmax", # a constant
            "Kc25", # a constant
            "ΔHkc", # a constant
            "Ko25", # a constant
            "ΔHko", # a constant
            "oi", # an empirical parameter
            "f", # an empirical factor
            "ΔHRd",# a constant
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
            FT(43540), # ΔHJmax - ozark model
            FT(0.9), # θj - ozark model
            FT(0.6), # ϕ - ozark model
            FT(58520), # ΔHVcmax - ozark model
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
    An_ci_app = webapp(ParamViz.parameterisation, inputs, output)
    return An_ci_app
end
