using Test
import CLIMAParameters as CP
using ClimaLSM.Canopy

import ClimaLSM
import ClimaLSM.Parameters as LSMP

include(joinpath(pkgdir(ClimaLSM), "parameters", "create_parameters.jl"))
@testset "Big Leaf Parameterizations" begin
    FT = Float32
    earth_param_set = create_lsm_parameters(FT)
    # Test with defaults
    RTparams = BeerLambertParameters{FT}()
    photosynthesisparams = FarquharParameters{FT}(C3();)
    stomatal_g_params = MedlynConductanceParameters{FT}()

    LAI = FT(5.0) # m2 (leaf) m-2 (ground)
    thermo_params = LSMP.thermodynamic_parameters(earth_param_set)
    c = FT(LSMP.light_speed(earth_param_set))
    h = FT(LSMP.planck_constant(earth_param_set))
    N_a = FT(LSMP.avogadro_constant(earth_param_set))
    λ = FT(5e-7) # m (500 nm)
    energy_per_photon = h * c / λ

    # Drivers
    T = FT(290) # K
    P = FT(101250) #Pa
    q = FT(0.02)
    VPD = ClimaLSM.vapor_pressure_deficit(T, P, q, thermo_params)#Pa
    p_l = FT(-2e6) # Pa
    ca = FT(4.11e-4) # mol/mol
    R = FT(LSMP.gas_constant(earth_param_set))
    θs = FT.(Array(0:0.1:(π / 2)))
    SW(θs) = cos.(θs) * FT.(500) # W/m^2
    PAR = SW(θs) ./ (energy_per_photon * N_a) # convert 500 W/m^2 to mol of photons per m^2/s
    K_c = extinction_coeff.(RTparams.ld, θs)
    @test all(K_c .≈ RTparams.ld ./ cos.(θs))
    APAR = plant_absorbed_ppfd.(PAR, RTparams.ρ_leaf, K_c, LAI, RTparams.Ω)
    @test all(
        APAR .≈
        PAR .* (1 - RTparams.ρ_leaf) .* (1 .- exp.(-K_c .* LAI .* RTparams.Ω)),
    )
    To = photosynthesisparams.To
    Vcmax =
        photosynthesisparams.Vcmax25 *
        arrhenius_function(T, To, R, photosynthesisparams.ΔHVcmax)
    Kc = MM_Kc(photosynthesisparams.Kc25, photosynthesisparams.ΔHkc, T, To, R)
    Ko = MM_Ko(photosynthesisparams.Ko25, photosynthesisparams.ΔHko, T, To, R)
    Γstar = co2_compensation(
        photosynthesisparams.Γstar25,
        photosynthesisparams.ΔHΓstar,
        T,
        To,
        R,
    )

    @test photosynthesisparams.Kc25 *
          arrhenius_function(T, To, R, photosynthesisparams.ΔHkc) ≈ Kc
    @test photosynthesisparams.Ko25 *
          arrhenius_function(T, To, R, photosynthesisparams.ΔHko) ≈ Ko
    @test photosynthesisparams.Γstar25 *
          arrhenius_function(T, To, R, photosynthesisparams.ΔHΓstar) ≈ Γstar
    @test photosynthesisparams.Vcmax25 *
          arrhenius_function(T, To, R, photosynthesisparams.ΔHVcmax) ≈ Vcmax

    m_t = medlyn_term(stomatal_g_params.g1, T, P, q, thermo_params)

    @test m_t == 1 + stomatal_g_params.g1 / sqrt(VPD)
    ci = intercellular_co2(ca, Γstar, m_t)
    @test ci == ca * (1 - 1 / m_t)
    @test intercellular_co2(ca, FT(1), m_t) == FT(1)

    Ac = rubisco_assimilation(
        photosynthesisparams.mechanism,
        Vcmax,
        ci,
        Γstar,
        Kc,
        Ko,
        photosynthesisparams.oi,
    )
    @test Ac ==
          Vcmax * (ci - Γstar) / (ci + Kc * (1 + photosynthesisparams.oi / Ko))
    Jmax = max_electron_transport(
        photosynthesisparams.Vcmax25,
        photosynthesisparams.ΔHJmax,
        T,
        To,
        R,
    )
    @test Jmax ==
          photosynthesisparams.Vcmax25 *
          FT(exp(1)) *
          arrhenius_function(T, To, R, photosynthesisparams.ΔHJmax)
    J =
        electron_transport.(
            APAR,
            Jmax,
            photosynthesisparams.θj,
            photosynthesisparams.ϕ,
        ) # mol m-2 s-1
    @test all(
        @.(
            photosynthesisparams.θj * J^2 -
            (photosynthesisparams.ϕ * APAR / 2 + Jmax) * J +
            photosynthesisparams.ϕ * APAR / 2 * Jmax < eps(FT)
        )
    )

    Aj = light_assimilation.(Ref(photosynthesisparams.mechanism), J, ci, Γstar)
    @test all(@.(Aj == J * (ci - Γstar) / (4 * (ci + 2 * Γstar))))
    β = moisture_stress(p_l, photosynthesisparams.sc, photosynthesisparams.pc)
    @test β ==
          (1 + exp(photosynthesisparams.sc * photosynthesisparams.pc)) /
          (1 + exp(photosynthesisparams.sc * (p_l - photosynthesisparams.pc)))
    #    C4 tests
    @test rubisco_assimilation(
        C4(),
        Vcmax,
        ci,
        Γstar,
        Kc,
        Ko,
        photosynthesisparams.oi,
    ) == Vcmax
    @test light_assimilation(C4(), J, ci, Γstar) == J

    Rd = dark_respiration(
        photosynthesisparams.Vcmax25,
        β,
        photosynthesisparams.f,
        photosynthesisparams.ΔHRd,
        T,
        To,
        R,
    )
    @test Rd ≈
          photosynthesisparams.Vcmax25 *
          β *
          photosynthesisparams.f *
          arrhenius_function(T, To, R, photosynthesisparams.ΔHRd)
    An = net_photosynthesis.(Ac, Aj, Rd, β)
    stomatal_conductance =
        medlyn_conductance.(
            stomatal_g_params.g0,
            stomatal_g_params.Drel,
            m_t,
            An,
            ca,
        )
    @test all(
        @.(
            stomatal_conductance ≈
            stomatal_g_params.g0 + stomatal_g_params.Drel * m_t * (An / ca)
        )
    )
    GPP = compute_GPP.(An, K_c, LAI, RTparams.Ω) # mol m-2 s-1
    @test all(@.(GPP ≈ An * (1 - exp(-K_c * LAI * RTparams.Ω)) / K_c))

    @test all(
        @.(
            upscale_leaf_conductance(stomatal_conductance, LAI, T, R, P) ≈
            stomatal_conductance * LAI * R * T / P
        )
    )
end
