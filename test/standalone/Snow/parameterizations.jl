using Test
import ClimaParams as CP
using ClimaLand.Snow
import ClimaLand
import ClimaLand.Parameters as LP
import Random

Random.seed!(1234)

for FT in (Float32, Float64)
    @testset "Snow Parameterizations, FT = $FT" begin
        param_set = LP.LandParameters(FT)

        # Density of liquid water (kg/m``^3``)
        _ρ_l = FT(LP.ρ_cloud_liq(param_set))
        # Density of ice water (kg/m``^3``)
        _ρ_i = FT(LP.ρ_cloud_ice(param_set))
        _cp_l = FT(LP.cp_l(param_set))
        _cp_i = FT(LP.cp_i(param_set))
        # Reference temperature (K)
        _T_ref = FT(LP.T_0(param_set))
        # Freezing temperature (K)
        _T_freeze = FT(LP.T_freeze(param_set))
        # Latent heat of fusion at ``T_0`` (J/kg)
        _LH_f0 = FT(LP.LH_f0(param_set))
        # Thermal conductivity of dry air
        κ_air = FT(LP.K_therm(param_set))

        ρ_snow = FT(200)
        densitymodel = Snow.ConstantDensityModel(ρ_snow)
        z_0m = FT(0.0024)
        α_snow = FT(0.8)
        # These values should match ClimaParams
        ϵ_snow = FT(0.99)
        z_0b = FT(0.00024)
        θ_r = FT(0.0)
        Ksat = FT(1e-3)
        κ_ice = FT(2.21)
        ρcD_g = FT(1700 * 2.09e3 * 0.1)
        Δt = Float64(180.0)
        parameters = SnowParameters{FT}(
            Δt,
            density = densitymodel,
            earth_param_set = param_set,
        )
        @test parameters.density.ρ_snow == ρ_snow
        @test typeof(parameters.density.ρ_snow) == FT
        @test parameters.ρcD_g == ρcD_g
        @test typeof(parameters.ρcD_g) == FT
        @test parameters.z_0m == z_0m
        @test typeof(parameters.z_0m) == FT
        @test parameters.z_0b == z_0b
        @test typeof(parameters.z_0b) == FT
        @test parameters.α_snow == α_snow
        @test typeof(parameters.α_snow) == FT
        @test parameters.ϵ_snow == ϵ_snow
        @test typeof(parameters.ϵ_snow) == FT
        @test parameters.Ksat == Ksat
        @test typeof(parameters.Ksat) == FT
        @test parameters.θ_r == θ_r
        @test typeof(parameters.θ_r) == FT
        @test parameters.κ_ice == κ_ice
        @test typeof(parameters.κ_ice) == FT


        T = FT.([275.0, 272, _T_freeze])
        @test snow_surface_temperature.(T) ≈ T
        @test specific_heat_capacity(FT(1.0), parameters) == _cp_l
        @test specific_heat_capacity(FT(0.0), parameters) == _cp_i
        @test snow_thermal_conductivity(ρ_snow, parameters) ==
              κ_air +
              (FT(0.07) * (ρ_snow / _ρ_i) + FT(0.93) * (ρ_snow / _ρ_i)^2) *
              (κ_ice - κ_air)
        @test all(
            maximum_liquid_mass_fraction.(ρ_snow, Ref(parameters)) .-
            θ_r * _ρ_l / ρ_snow .== 0,
        )

        SWE = cat(FT.(rand(10)), FT(0), dims = 1)
        z = SWE * _ρ_l ./ ρ_snow
        @test runoff_timescale.(z, Ksat, FT(Δt)) ≈ max.(Δt, z ./ Ksat)
        ρ_calc = snow_bulk_density.(SWE, z, parameters)
        @test all(ρ_calc[1:(end - 1)] .≈ ρ_snow)
        @test ρ_calc[end] == _ρ_l
        @test snow_bulk_density(eps(FT(0)), 2 * eps(FT(0)), parameters) == _ρ_l
    end
end
