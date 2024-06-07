using Test
import ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
using Statistics
using NLsolve
using ClimaCore
import ClimaParams
using ClimaLand
using ClimaLand.Domains: Point, Plane
using ClimaLand.Canopy
using ClimaLand.Canopy.PlantHydraulics
import ClimaLand
import ClimaLand.Parameters as LP
import Insolation
using Dates


for FT in (Float32, Float64)
    @testset "LAI assertions, FT = $FT" begin
        LAI = FT.([0, 1])
        SAI = FT.([0, 1])
        RAI = FT.([0, 1])
        h_stem = FT.([0, 1])
        h_leaf = FT.([1])
        for L in LAI
            for S in SAI
                for R in RAI
                    for h_s in h_stem
                        for h_l in h_leaf
                            area_index = (root = R, stem = S, leaf = L)
                            if n_l == 0
                                @test_throws AssertionError ClimaLand.Canopy.PlantHydraulics.lai_consistency_check(
                                    h_s,
                                    h_l,
                                    area_index,
                                )
                            end

                            if (L > eps(FT) || S > eps(FT)) && R < eps(FT)
                                @test_throws AssertionError ClimaLand.Canopy.PlantHydraulics.lai_consistency_check(
                                    h_s,
                                    h_l,
                                    area_index,
                                )
                            end

                            if S > FT(0) && n_s == 0
                                @test_throws AssertionError ClimaLand.Canopy.PlantHydraulics.lai_consistency_check(
                                    h_s,
                                    h_l,
                                    area_index,
                                )
                            end

                            if S < eps(FT) && n_s > 0
                                @test_throws AssertionError ClimaLand.Canopy.PlantHydraulics.lai_consistency_check(
                                    h_s,
                                    h_l,
                                    area_index,
                                )
                            end
                        end
                    end
                end
            end
        end
    end

    @testset "Plant hydraulics parameterizations, FT = $FT" begin
        ν = FT(0.5)
        S_s = FT(1e-2)
        K_sat = FT(1.8e-8)
        ψ63 = FT(-4 / 0.0098)
        Weibull_param = FT(4)
        a = FT(0.05 * 0.0098)
        conductivity_model =
            PlantHydraulics.Weibull{FT}(K_sat, ψ63, Weibull_param)
        retention_model = PlantHydraulics.LinearRetentionCurve{FT}(a)
        S_l = FT.([0.7, 0.9, 1.0, 1.1])
        @test all(
            @. PlantHydraulics.inverse_water_retention_curve(
                retention_model,
                PlantHydraulics.water_retention_curve(
                    retention_model,
                    S_l,
                    ν,
                    S_s,
                ),
                ν,
                S_s,
            ) == S_l
        )
        ψ = PlantHydraulics.water_retention_curve.(retention_model, S_l, ν, S_s)
        # @show @. abs(PlantHydraulics.hydraulic_conductivity(conductivity_model, ψ) -
        # min(K_sat * exp(-(ψ / ψ63)^Weibull_param), K_sat))
        @test all(
            @. abs(
                PlantHydraulics.hydraulic_conductivity(conductivity_model, ψ) -
                min(K_sat * exp(-(ψ / ψ63)^Weibull_param), K_sat),
            ) < 2 * eps(FT)
        )
    end

    @testset "No plant, FT = $FT" begin
        domain = Point(; z_sfc = FT(0.0))

        AR_params = AutotrophicRespirationParameters(FT)
        RTparams = BeerLambertParameters(FT)
        photosynthesis_params = FarquharParameters(FT, C3())
        stomatal_g_params = MedlynConductanceParameters(FT)

        AR_model = AutotrophicRespirationModel{FT}(AR_params)
        stomatal_model = MedlynConductanceModel{FT}(stomatal_g_params)
        photosynthesis_model = FarquharModel{FT}(photosynthesis_params)
        rt_model = BeerLambertModel{FT}(RTparams)

        earth_param_set = LP.LandParameters(FT)
        LAI = FT(0.0) # m2 [leaf] m-2 [ground]
        z_0m = FT(2.0) # m, Roughness length for momentum
        z_0b = FT(0.1) # m, Roughness length for scalars
        h_canopy = FT(0.0) # m, canopy height
        h_sfc = FT(0.0) # m
        h_int = FT(30.0) # m, "where measurements would be taken at a typical flux tower of a 20m canopy"
        shared_params = SharedCanopyParameters{FT, typeof(earth_param_set)}(
            z_0m,
            z_0b,
            earth_param_set,
        )
        lat = FT(0.0) # degree
        long = FT(-180) # degree

        function zenith_angle(
            t,
            ref_time;
            latitude = lat,
            longitude = long,
            insol_params = earth_param_set.insol_params,
        )
            current_datetime = ref_time + Dates.Second(round(t))
            d, δ, η_UTC =
                FT.(
                    Insolation.helper_instantaneous_zenith_angle(
                        current_datetime,
                        ref_time,
                        insol_params,
                    )
                )
            return Insolation.instantaneous_zenith_angle(
                d,
                δ,
                η_UTC,
                longitude,
                latitude,
            )[1]
        end

        function shortwave_radiation(
            t;
            latitude = lat,
            longitude = long,
            insol_params = earth_param_set.insol_params,
        )
            return 1000 # W/m^2
        end

        function longwave_radiation(t)
            return 200 # W/m^2
        end

        u_atmos = t -> 10 #m.s-1

        liquid_precip = (t) -> 0 # m
        snow_precip = (t) -> 0 # m
        T_atmos = t -> 290 # Kelvin
        q_atmos = t -> 0.001 # kg/kg
        P_atmos = t -> 1e5 # Pa
        h_atmos = h_int # m
        c_atmos = (t) -> 4.11e-4 # mol/mol
        ref_time = DateTime(2005)
        atmos = PrescribedAtmosphere(
            TimeVaryingInput(liquid_precip),
            TimeVaryingInput(snow_precip),
            TimeVaryingInput(T_atmos),
            TimeVaryingInput(u_atmos),
            TimeVaryingInput(q_atmos),
            TimeVaryingInput(P_atmos),
            ref_time,
            h_atmos,
            earth_param_set;
            c_co2 = TimeVaryingInput(c_atmos),
        )
        radiation = PrescribedRadiativeFluxes(
            FT,
            TimeVaryingInput(shortwave_radiation),
            TimeVaryingInput(longwave_radiation),
            ref_time;
            θs = zenith_angle,
        )

        h_stem = FT(0.0) # m
        h_leaf = FT(0.0) # m
        SAI = FT(0) # m2/m2
        RAI = FT(0) # m2/m2
        lai_fun = t -> 0
        ai_parameterization = PlantHydraulics.PrescribedSiteAreaIndex{FT}(
            TimeVaryingInput(lai_fun),
            SAI,
            RAI,
        )
        K_sat_plant = 0 # m/s.
        ψ63 = FT(-4 / 0.0098) # / MPa to m
        Weibull_param = FT(4) # unitless
        a = FT(0.05 * 0.0098) # 1/m
        plant_ν = FT(0.1) # m3/m3
        plant_S_s = FT(1e-2)
        conductivity_model =
            PlantHydraulics.Weibull{FT}(K_sat_plant, ψ63, Weibull_param)
        retention_model = PlantHydraulics.LinearRetentionCurve{FT}(a)
        root_depths = [FT(0.0)]# 1st element is the deepest root depth
        function root_distribution(z::T) where {T}
            return T(0) # (1/m)
        end

        param_set = PlantHydraulics.PlantHydraulicsParameters(;
            ai_parameterization = ai_parameterization,
            ν = plant_ν,
            S_s = plant_S_s,
            root_distribution = root_distribution,
            conductivity_model = conductivity_model,
            retention_model = retention_model,
        )

        transpiration = DiagnosticTranspiration{FT}()
        soil_driver = PrescribedSoil(FT)
        plant_hydraulics = PlantHydraulics.BigLeafHydraulicsModel{FT}(;
            parameters = param_set,
            transpiration = transpiration,
            h_stem = h_stem,
            h_leaf = h_leaf,
        )

        autotrophic_parameters = AutotrophicRespirationParameters(FT)
        autotrophic_respiration_model =
            AutotrophicRespirationModel(autotrophic_parameters)

        model = ClimaLand.Canopy.CanopyModel{FT}(;
            parameters = shared_params,
            domain = domain,
            autotrophic_respiration = autotrophic_respiration_model,
            radiative_transfer = rt_model,
            photosynthesis = photosynthesis_model,
            conductance = stomatal_model,
            hydraulics = plant_hydraulics,
            soil_driver = soil_driver,
            atmos = atmos,
            radiation = radiation,
        )

        Y, p, coords = initialize(model)
        dY = similar(Y)
        for i in 1:(h_stem != 0 + 1)
            Y.canopy.hydraulics.ϑ_l.:($i) .= FT(0.1)
            p.canopy.hydraulics.ψ.:($i) .= NaN
            p.canopy.hydraulics.fa.:($i) .= NaN
            dY.canopy.hydraulics.ϑ_l.:($i) .= NaN
        end
        set_initial_cache! = make_set_initial_cache(model)
        set_initial_cache!(p, Y, FT(0.0))
        @test all(parent(p.canopy.hydraulics.fa) .≈ FT(0.0))
        @test all(parent(p.canopy.hydraulics.fa_roots) .≈ FT(0.0))
        @test all(parent(p.canopy.conductance.transpiration) .≈ FT(0.0))
        @test all(parent(p.canopy.radiative_transfer.apar) .≈ FT(0.0))
        exp_tend! = make_exp_tendency(model)
        exp_tend!(dY, Y, p, FT(0))
        @test all(parent(dY.canopy.hydraulics.ϑ_l.:1) .≈ FT(0.0))
    end
end
