using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaParams as CP
using ClimaLand.Soil
import ClimaLand
import ClimaLand.Parameters as LP

for FT in (Float32, Float64)
    @testset "integrated Energy and Hydrology Parameterizations, FT = $FT" begin
        param_set = LP.LandParameters(FT)

        # Density of liquid water (kg/m``^3``)
        _ρ_l = FT(LP.ρ_cloud_liq(param_set))
        # Density of ice water (kg/m``^3``)
        _ρ_i = FT(LP.ρ_cloud_ice(param_set))
        # Volum. isobaric heat capacity liquid water (J/m3/K)
        _ρcp_l = FT(LP.cp_l(param_set) * _ρ_l)
        # Volumetric isobaric heat capacity ice (J/m3/K)
        _ρcp_i = FT(LP.cp_i(param_set) * _ρ_i)
        # Reference temperature (K)
        _T_ref = FT(LP.T_0(param_set))
        # Latent heat of fusion at ``T_0`` (J/kg)
        _LH_f0 = FT(LP.LH_f0(param_set))

        ν = FT(0.2)
        S_s = FT(1e-3)
        θ_r = FT(0.1)
        vg_α = FT(2.0)
        vg_n = FT(1.4)
        hcm = vanGenuchten{FT}(; α = vg_α, n = vg_n)
        K_sat = FT(1e-5)
        ν_ss_om = FT(0.1)
        ν_ss_gravel = FT(0.1)
        ν_ss_quartz = FT(0.1)

        # Constants from Ballard and Arp paper
        κ_minerals = FT(2.5)
        κ_om = FT(0.25)
        κ_quartz = FT(8.0)
        #κ_gravel = κ_minerals implicitly in equation for κ_solid

        ρp_quartz = FT(2.66e3)
        ρp_minerals = FT(2.65e3)
        ρp_om = FT(1.3e3)
        ρp_gravel = ρp_minerals

        ρc_quartz = FT(2.01e6)
        ρc_om = FT(2.51e6)
        ρc_minerals = FT(2.01e6)
        ρc_gravel = ρc_minerals

        κ_air = FT(LP.K_therm(param_set))
        κ_ice = FT(2.21)
        κ_liq = FT(0.57)

        parameters = EnergyHydrologyParameters(
            FT;
            ν,
            ν_ss_om,
            ν_ss_quartz,
            ν_ss_gravel,
            hydrology_cm = hcm,
            K_sat,
            S_s,
            θ_r,
        )
        # Test that the preset params are set properly
        (; α, β, Ω, hydrology_cm, γ, γT_ref) = parameters
        @test α == FT(0.24)
        @test β == FT(18.3)
        @test γ == FT(2.64e-2)
        @test Ω == FT(7)
        @test γT_ref == FT(288)
        # Test derived parameters
        κ_solid_soil =
            Soil.κ_solid.(ν_ss_om, ν_ss_quartz, κ_om, κ_quartz, κ_minerals)
        ρp = (
            ν_ss_om * ρp_om +
            ν_ss_quartz * ρp_quartz +
            ν_ss_gravel * ρp_gravel +
            (1 - ν_ss_om - ν_ss_quartz - ν_ss_gravel) * ρp_minerals
        )
        ρc_ss = (
            ν_ss_om * ρc_om +
            ν_ss_quartz * ρc_quartz +
            ν_ss_gravel * ρc_gravel +
            (1 - ν_ss_om - ν_ss_quartz - ν_ss_gravel) * ρc_minerals
        )
        # Volumetric heat capacity of dry soil - per unit volume of soil
        ρc_ds = (1 - ν) * ρc_ss
        @test parameters.κ_dry == Soil.κ_dry.(ρp, ν, κ_solid_soil, κ_air)
        @test parameters.κ_sat_frozen ==
              Soil.κ_sat_frozen.(κ_solid_soil, ν, κ_ice)
        @test parameters.κ_sat_unfrozen ==
              Soil.κ_sat_unfrozen.(κ_solid_soil, ν, κ_liq)
        @test parameters.ρc_ds == ρc_ds
        @test hydrology_cm.m == FT(1.0 - 1.0 / vg_n)
        @test hydrology_cm.S_c ==
              FT(1 + ((vg_n - 1) / vg_n)^(1 - 2 * vg_n))^(-hydrology_cm.m)
        @test typeof.([α, β, γ, Ω, γT_ref]) == [FT, FT, FT, FT, FT]

        @test temperature_from_ρe_int(
            FT(5.4e7),
            FT(0.05),
            FT(2.1415e6),
            param_set,
        ) == FT(_T_ref + (5.4e7 + 0.05 * _ρ_i * _LH_f0) / 2.1415e6)

        @test volumetric_heat_capacity(
            FT(0.25),
            FT(0.05),
            FT(ρc_ds),
            param_set,
        ) == FT(ρc_ds + 0.25 * _ρcp_l + 0.05 * _ρcp_i)

        @test volumetric_internal_energy(
            FT(0.05),
            FT(2.1415e6),
            FT(300),
            param_set,
        ) == FT(2.1415e6 * (300.0 - _T_ref) - 0.05 * _ρ_i * _LH_f0)

        @test κ_sat(FT(0.25), FT(0.05), FT(0.57), FT(2.29)) ≈
              FT(0.57^(0.25 / (0.05 + 0.25)) * 2.29^(0.05 / (0.05 + 0.25)))

        @test κ_sat(FT(0), FT(0), FT(0.57), FT(2.29)) == FT(0.5 * (0.57 + 2.29))

        @test relative_saturation(FT(0.25), FT(0.05), FT(0.4)) ==
              FT((0.25 + 0.05) / 0.4)

        # ice fraction = 0
        @test kersten_number(
            FT(0),
            FT(0.75),
            parameters.α,
            parameters.β,
            ν_ss_om,
            ν_ss_quartz,
            ν_ss_gravel,
        ) ≈ FT(
            0.75^((FT(1) + 0.1 - 0.24 * 0.1 - 0.1) / FT(2)) *
            (
                (FT(1) + exp(-18.3 * 0.75))^(-FT(3)) -
                ((FT(1) - 0.75) / FT(2))^FT(3)
            )^(FT(1) - 0.1),
        )

        # ice fraction ~= 0
        @test kersten_number(
            FT(0.05),
            FT(0.75),
            parameters.α,
            parameters.β,
            ν_ss_om,
            ν_ss_quartz,
            ν_ss_gravel,
        ) == FT(0.75^(FT(1) + 0.1))

        @test thermal_conductivity(FT(1.5), FT(0.7287), FT(0.7187)) ==
              FT(0.7287 * 0.7187 + (FT(1) - 0.7287) * 1.5)

        @test volumetric_internal_energy_liq(FT(300), param_set) ==
              FT(_ρcp_l * (300.0 - _T_ref))

        @test Soil.κ_solid(FT(0.5), FT(0.25), FT(2.0), FT(3.0), FT(2.0)) ≈
              FT(2)^FT(0.5) * FT(2)^FT(0.25) * FT(3.0)^FT(0.25)

        @test Soil.κ_sat_frozen(FT(0.5), FT(0.1), FT(0.4)) ==
              FT(0.5)^FT(0.9) * FT(0.4)^FT(0.1)

        @test Soil.κ_sat_unfrozen(FT(0.5), FT(0.1), FT(0.4)) ==
              FT(0.5)^FT(0.9) * FT(0.4)^FT(0.1)

        ρb = (1 - ν) * ρp
        @test Soil.κ_dry(ρp, ν, κ_solid_soil, κ_air) ==
              ((FT(0.053) * FT(κ_solid_soil) - κ_air) * FT(ρb) + κ_air * ρp) /
              (ρp - (FT(1.0) - FT(0.053)) * ρb)
        # Impedance factor
        @test impedance_factor(FT(1.0), parameters.Ω) ≈ 1e-7

        # Viscosity Factor
        T = FT.([278.0, 288.0, 298.0])
        @test viscosity_factor.(T, parameters.γ, parameters.γT_ref) ≈
              exp.(parameters.γ .* (T .- parameters.γT_ref))

    end

    @testset "Brooks and Corey closure, FT = $FT" begin
        hcm = BrooksCorey{FT}(; ψb = FT(-0.09), c = FT(0.228))
        # Test derived parameter
        @test hcm.S_c == (1 + 1 / FT(0.228))^(-FT(0.228))

        S = FT.([0.5, 1.0, 1.5])
        θ = FT.([0.3, 0.4, 0.5])
        θ_r = FT(0.2)
        ν = FT(0.4)
        K_sat = FT(2.9e-7)
        S_s = FT(1e-2)

        # Matric Potential and inverse
        va = S[1]^(-1 / hcm.c) * hcm.ψb
        ψ = matric_potential.(hcm, S[1:2])
        @test inverse_matric_potential.(hcm, ψ) ≈ S[1:2]
        @test ψ ≈ [va, hcm.ψb]
        @test eltype(ψ) == FT

        #Pressure head
        p = pressure_head.(hcm, θ_r, θ, ν, S_s)
        @test p ≈ push!(ψ, FT(0.1 / 1e-2 + hcm.ψb))
        @test eltype(p) == FT
        # Derivative
        dψ = dψdϑ.(hcm, θ, ν, θ_r, S_s)
        va = @. -hcm.ψb / (hcm.c * (ν - θ_r)) * S^(-(1 + 1 / hcm.c))
        @test dψ ≈ [va[1], 1 / S_s, 1 / S_s]

        # Hydraulic K
        k = @. Soil.hydraulic_conductivity.(hcm, K_sat, S)
        va = S[1]^(2 / hcm.c + 3) * K_sat
        @test k ≈ [va, K_sat, K_sat]
        @test eltype(k) == FT
    end

    @testset "Richards equation and van Genuchten closures, FT = $FT" begin
        θ_r = FT(0.2)
        ν = FT(0.4)
        S_s = FT(1e-2)
        vg_α = FT(3.6)
        vg_n = FT(1.56)
        hcm = vanGenuchten{FT}(; α = vg_α, n = vg_n)
        K_sat = FT(2.9e-7)
        vg_m = FT(1.0 - 1.0 / vg_n)
        richards_parameters = RichardsParameters(;
            ν = ν,
            hydrology_cm = hcm,
            K_sat = K_sat,
            S_s = S_s,
            θ_r = θ_r,
        )
        # Effective _saturation
        θ = FT.([0.3, 0.4, 0.5])
        S = Soil.effective_saturation.(ν, θ, θ_r)
        @test S ≈ [0.5, 1.0, 1.5]
        @test eltype(S) == FT

        # Matric Potential and inverse
        va = -((S[1]^(-FT(1) / vg_m) - FT(1)) * vg_α^(-vg_n))^(FT(1) / vg_n)
        ψ = matric_potential.(hcm, S[1:2])
        @test inverse_matric_potential.(hcm, ψ) ≈ S[1:2]
        @test ψ ≈ [va, 0.0]
        @test eltype(ψ) == FT
        # Derivative
        dψ = dψdϑ.(hcm, θ, ν, θ_r, S_s)
        va = FT(
            1.0 / (vg_α * vg_m * vg_n) / (ν - θ_r) *
            (S[1]^(-1 / vg_m) - 1)^(1 / vg_n - 1) *
            S[1]^(-1 / vg_m - 1),
        )
        @test dψ ≈ [va, 1 / S_s, 1 / S_s]
        #Pressure head
        p = pressure_head.(hcm, θ_r, θ, ν, S_s)
        @test p ≈ push!(ψ, FT(10.0))
        @test eltype(p) == FT

        # Hydraulic K
        k = Soil.hydraulic_conductivity.(hcm, K_sat, S)
        va =
            (sqrt(S[1]) * (FT(1) - (FT(1) - S[1]^(FT(1) / vg_m))^vg_m)^FT(2)) *
            K_sat
        @test k ≈ [va, K_sat, K_sat]
        @test eltype(k) == FT

        # Volumetric Liquid Fraction
        vlf =
            volumetric_liquid_fraction.(
                FT.([0.25, 0.5, 0.75]),
                FT(0.5),
                FT(0.0),
            )
        @test vlf ≈ FT.([0.25, 0.5, 0.5])
    end

    @testset "Freezing and Thawing, FT = $FT" begin
        param_set = LP.LandParameters(FT)

        # Density of liquid water (kg/m``^3``)
        _ρ_l = FT(LP.ρ_cloud_liq(param_set))
        # Density of ice water (kg/m``^3``)
        _ρ_i = FT(LP.ρ_cloud_ice(param_set))
        # Volum. isobaric heat capacity liquid water (J/m3/K)
        _ρcp_l = FT(LP.cp_l(param_set) * _ρ_l)
        # Volumetric isobaric heat capacity ice (J/m3/K)
        _ρcp_i = FT(LP.cp_i(param_set) * _ρ_i)
        # Reference temperature (K)
        _T_ref = FT(LP.T_0(param_set))
        # Latent heat of fusion at ``T_0`` (J/kg)
        _LH_f0 = FT(LP.LH_f0(param_set))
        # Gravitational acceleration
        _grav = FT(LP.grav(param_set))
        # T freeze
        _T_freeze = FT(LP.T_freeze(param_set))

        ν = FT(0.2)
        S_s = FT(1e-3)
        θ_r = FT(0.1)
        vg_α = FT(2.0)
        vg_n = FT(1.4)
        hcm = vanGenuchten{FT}(; α = vg_α, n = vg_n)

        K_sat = FT(1e-5)
        ν_ss_om = FT(0.1)
        ν_ss_gravel = FT(0.1)
        ν_ss_quartz = FT(0.1)
        parameters = EnergyHydrologyParameters(
            FT;
            ν,
            ν_ss_om,
            ν_ss_quartz,
            ν_ss_gravel,
            hydrology_cm = hcm,
            K_sat,
            S_s,
            θ_r,
        )

        Δz = FT(1.0)
        τ = thermal_time(parameters.ρc_ds, Δz, parameters.κ_dry)
        @test τ == parameters.ρc_ds * Δz^2 / parameters.κ_dry
        θ_l = FT.([0.11, 0.15, ν])
        θ_i = FT(0.0)
        T = FT(270)
        θtot = @.(_ρ_i / _ρ_l * θ_i + θ_l)
        ψ0 = @. matric_potential(hcm, Soil.effective_saturation(ν, θtot, θ_r))
        Tf_depressed = _T_freeze * exp.(_grav * ψ0 / _LH_f0)
        ψT = @. _LH_f0 / _grav *
           log(T / Tf_depressed) *
           ClimaLand.heaviside(Tf_depressed - T)
        θ_star = @. inverse_matric_potential(hcm, ψ0 + ψT) * (ν - θ_r) + θ_r
        ρc_s = volumetric_heat_capacity.(θ_l, θ_i, parameters.ρc_ds, param_set)
        τ = thermal_time.(ρc_s, Δz, parameters.κ_dry)
        @test (
            phase_change_source.(θ_l, θ_i, T, τ, ν, θ_r, hcm, parameters) ≈
            (θ_l .- θ_star) ./ τ
        )
        @test sum(
            phase_change_source.(θ_l, θ_i, T, τ, ν, θ_r, hcm, parameters) .>
            0.0,
        ) == 3

        θ_l = FT.([0.11, 0.15, ν])
        θ_i = FT(0.0)
        T = FT(274)
        θtot = @.(_ρ_i / _ρ_l * θ_i + θ_l)
        ψ0 = @. matric_potential(hcm, Soil.effective_saturation(ν, θtot, θ_r))
        Tf_depressed = _T_freeze * exp.(_grav * ψ0 / _LH_f0)
        ψT = @. _LH_f0 / _grav *
           log(T / Tf_depressed) *
           ClimaLand.heaviside(Tf_depressed - T)
        θ_star = @. inverse_matric_potential(hcm, ψ0 + ψT) * (ν - θ_r) + θ_r
        ρc_s = volumetric_heat_capacity.(θ_l, θ_i, parameters.ρc_ds, param_set)
        τ = thermal_time.(ρc_s, Δz, parameters.κ_dry)
        @test (
            phase_change_source.(θ_l, θ_i, T, τ, ν, θ_r, hcm, parameters) ≈
            zeros(FT, 3)
        )
        @test (θ_star ≈ θ_l)


        θ_l = FT(0.11)
        θ_i = FT.([0.05, 0.08])
        T = FT(274)
        θtot = @.(_ρ_i / _ρ_l * θ_i + θ_l)
        ψ0 = @. matric_potential(hcm, Soil.effective_saturation(ν, θtot, θ_r))
        Tf_depressed = _T_freeze * exp.(_grav * ψ0 / _LH_f0)
        ψT = @. _LH_f0 / _grav *
           log(T / Tf_depressed) *
           ClimaLand.heaviside(Tf_depressed - T)
        θ_star = @. inverse_matric_potential(hcm, ψ0 + ψT) * (ν - θ_r) + θ_r
        ρc_s = volumetric_heat_capacity.(θ_l, θ_i, parameters.ρc_ds, param_set)
        τ = thermal_time.(ρc_s, Δz, parameters.κ_dry)
        @test (
            phase_change_source.(θ_l, θ_i, T, τ, ν, θ_r, hcm, parameters) ≈
            (θ_l .- θ_star) ./ τ
        )
        @test sum(
            phase_change_source.(θ_l, θ_i, T, τ, ν, θ_r, hcm, parameters) .<
            0.0,
        ) == 2


        θ_l = FT(0.11)
        θ_i = FT.([0.05, 0.08])
        T = FT(260)
        θtot = @.(_ρ_i / _ρ_l * θ_i + θ_l)
        ψ0 = @. matric_potential(hcm, Soil.effective_saturation(ν, θtot, θ_r))
        Tf_depressed = _T_freeze * exp.(_grav * ψ0 / _LH_f0)
        ψT = @. _LH_f0 / _grav *
           log(T / Tf_depressed) *
           ClimaLand.heaviside(Tf_depressed - T)
        θ_star = @. inverse_matric_potential(hcm, ψ0 + ψT) * (ν - θ_r) + θ_r
        ρc_s = volumetric_heat_capacity.(θ_l, θ_i, parameters.ρc_ds, param_set)
        τ = thermal_time.(ρc_s, Δz, parameters.κ_dry)
        @test (
            phase_change_source.(θ_l, θ_i, T, τ, ν, θ_r, hcm, parameters) ≈
            (θ_l .- θ_star) ./ τ
        )
        @test sum(
            phase_change_source.(θ_l, θ_i, T, τ, ν, θ_r, hcm, parameters) .>
            0.0,
        ) == 2
    end
end
