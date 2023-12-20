using Test
import CLIMAParameters as CP
using ClimaCore
using Thermodynamics
using Dates
using StaticArrays
using ClimaLSM
using ClimaLSM: PrescribedAtmosphere, PrescribedRadiativeFluxes
using ClimaLSM.Canopy
using ClimaLSM.Canopy.PlantHydraulics
using ClimaLSM.Domains: Point
import Insolation

import ClimaLSM
import ClimaLSM.Parameters as LSMP

include(joinpath(pkgdir(ClimaLSM), "parameters", "create_parameters.jl"))
for FT in (Float32, Float64)
    @testset "Canopy software pipes, FT = $FT" begin
        domain = Point(; z_sfc = FT(0.0))

        AR_params = AutotrophicRespirationParameters{FT}()
        RTparams = BeerLambertParameters{FT}()
        photosynthesis_params = FarquharParameters{FT}(C3();)
        stomatal_g_params = MedlynConductanceParameters{FT}()

        AR_model = AutotrophicRespirationModel{FT}(AR_params)
        stomatal_model = MedlynConductanceModel{FT}(stomatal_g_params)
        photosynthesis_model = FarquharModel{FT}(photosynthesis_params)
        rt_model = BeerLambertModel{FT}(RTparams)

        earth_param_set = create_lsm_parameters(FT)
        LAI = FT(8.0) # m2 [leaf] m-2 [ground]
        z_0m = FT(2.0) # m, Roughness length for momentum - value from tall forest ChatGPT
        z_0b = FT(0.1) # m, Roughness length for scalars - value from tall forest ChatGPT
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
            liquid_precip,
            snow_precip,
            T_atmos,
            u_atmos,
            q_atmos,
            P_atmos,
            ref_time,
            h_atmos;
            c_co2 = c_atmos,
        )
        radiation = PrescribedRadiativeFluxes(
            FT,
            shortwave_radiation,
            longwave_radiation,
            ref_time;
            θs = zenith_angle,
        )

        # Plant Hydraulics
        RAI = FT(1)
        SAI = FT(0)
        lai_fun = t -> LAI
        ai_parameterization =
            PlantHydraulics.PrescribedSiteAreaIndex{FT}(lai_fun, SAI, RAI)
        K_sat_plant = FT(1.8e-8) # m/s
        ψ63 = FT(-4 / 0.0098) # / MPa to m, Holtzman's original parameter value
        Weibull_param = FT(4) # unitless, Holtzman's original c param value
        a = FT(0.05 * 0.0098) # Holtzman's original parameter for the bulk modulus of elasticity
        plant_ν = FT(0.7) # m3/m3
        plant_S_s = FT(1e-2 * 0.0098) # m3/m3/MPa to m3/m3/m
        conductivity_model =
            PlantHydraulics.Weibull{FT}(K_sat_plant, ψ63, Weibull_param)
        retention_model = PlantHydraulics.LinearRetentionCurve{FT}(a)
        root_depths = SVector{10, FT}(-(10:-1:1.0) ./ 10.0 * 2.0 .+ 0.2 / 2.0) # 1st element is the deepest root depth
        function root_distribution(z::T) where {T}
            return T(1.0 / 0.5) * exp(z / T(0.5)) # (1/m)
        end
        param_set = PlantHydraulics.PlantHydraulicsParameters(;
            ai_parameterization = ai_parameterization,
            ν = plant_ν,
            S_s = plant_S_s,
            root_distribution = root_distribution,
            conductivity_model = conductivity_model,
            retention_model = retention_model,
        )
        Δz = FT(1.0) # height of compartments
        n_stem = Int64(0) # number of stem elements
        n_leaf = Int64(1) # number of leaf elements
        compartment_centers =
            FT.(
                Vector(
                    range(
                        start = Δz / 2,
                        step = Δz,
                        stop = Δz * (n_stem + n_leaf) - (Δz / 2),
                    ),
                ),
            )
        compartment_faces =
            FT.(
                Vector(
                    range(
                        start = 0.0,
                        step = Δz,
                        stop = Δz * (n_stem + n_leaf),
                    ),
                )
            )

        ψ_soil0 = FT(0.0)
        soil_driver = PrescribedSoil(FT)
        plant_hydraulics = PlantHydraulics.PlantHydraulicsModel{FT}(;
            parameters = param_set,
            n_stem = n_stem,
            n_leaf = n_leaf,
            compartment_surfaces = compartment_faces,
            compartment_midpoints = compartment_centers,
        )
        canopy = ClimaLSM.Canopy.CanopyModel{FT}(;
            parameters = shared_params,
            domain = domain,
            autotrophic_respiration = AR_model,
            radiative_transfer = rt_model,
            photosynthesis = photosynthesis_model,
            conductance = stomatal_model,
            hydraulics = plant_hydraulics,
            soil_driver = soil_driver,
            atmos = atmos,
            radiation = radiation,
        )
        Y, p, coords = ClimaLSM.initialize(canopy)

        # Check that structure of Y is value (will error if not)
        @test !isnothing(zero(Y))
        @test typeof(canopy.energy) == PrescribedCanopyTempModel{FT}
        @test propertynames(p) == (:canopy,)
        for component in ClimaLSM.Canopy.canopy_components(canopy)
            # Only hydraulics has a prognostic variable
            if component == :hydraulics
                @test propertynames(getproperty(Y.canopy, component)) ==
                      ClimaLSM.prognostic_vars(getproperty(canopy, component))
            end
            @test propertynames(getproperty(p.canopy, component)) ==
                  ClimaLSM.auxiliary_vars(getproperty(canopy, component))
            @test getproperty(auxiliary_types(canopy), component) ==
                  auxiliary_types(getproperty(canopy, component))
            @test getproperty(auxiliary_vars(canopy), component) ==
                  auxiliary_vars(getproperty(canopy, component))
            @test getproperty(prognostic_types(canopy), component) ==
                  prognostic_types(getproperty(canopy, component))
            @test getproperty(prognostic_types(canopy), component) ==
                  prognostic_types(getproperty(canopy, component))
        end
        parent(Y.canopy.hydraulics.ϑ_l) .= plant_ν
        set_initial_cache! = make_set_initial_cache(canopy)
        exp_tendency! = make_exp_tendency(canopy)
        t0 = FT(0.0)
        dY = similar(Y)
        set_initial_cache!(p, Y, t0)
        # check that this is updated correctly:
        # @test p.canopy.autotrophic_respiration.Ra ==
        exp_tendency!(dY, Y, p, t0)
        (evapotranspiration, shf, lhf) =
            ClimaLSM.Canopy.canopy_turbulent_fluxes(
                canopy.atmos,
                canopy,
                Y,
                p,
                t0,
            )

        @test p.canopy.hydraulics.fa.:1 == evapotranspiration
        @test p.canopy.energy.shf == shf
        @test p.canopy.energy.lhf == lhf
        @test p.canopy.conductance.transpiration == evapotranspiration
        c = FT(LSMP.light_speed(earth_param_set))
        h = FT(LSMP.planck_constant(earth_param_set))
        N_a = FT(LSMP.avogadro_constant(earth_param_set))
        _σ = FT(LSMP.Stefan(earth_param_set))
        (; α_PAR_leaf, λ_γ_PAR, λ_γ_NIR, ϵ_canopy) =
            canopy.radiative_transfer.parameters
        APAR = p.canopy.radiative_transfer.apar
        ANIR = p.canopy.radiative_transfer.anir
        energy_per_photon_PAR = h * c / λ_γ_PAR
        energy_per_photon_NIR = h * c / λ_γ_NIR

        @test p.canopy.radiative_transfer.SW_n ==
              @. (energy_per_photon_PAR * N_a * APAR) +
                 (energy_per_photon_NIR * N_a * ANIR)

        T_canopy = FT.(atmos.T(t0))
        T_soil = FT.(soil_driver.T(t0))
        ϵ_soil = FT.(soil_driver.ϵ)
        LW_d = FT.(radiation.LW_d(t0))
        LW_d_canopy = (1 - ϵ_canopy) * LW_d + ϵ_canopy * _σ * T_canopy^4
        LW_u_soil = ϵ_soil * _σ * T_soil^4 + (1 - ϵ_soil) * LW_d_canopy
        @test Array(parent(p.canopy.radiative_transfer.LW_n))[1] ≈
              ϵ_canopy * LW_d - 2 * ϵ_canopy * _σ * T_canopy^4 +
              ϵ_canopy * LW_u_soil


        # Penman-monteith
        Δ = FT(100 * (0.444017302 + (290 - 273.15) * 0.0286064092))
        Rn = FT(shortwave_radiation(t0))
        G = FT(0.0)
        thermo_params = canopy.parameters.earth_param_set.thermo_params
        ts_in = construct_atmos_ts(atmos, t0, thermo_params)
        ρa = Thermodynamics.air_density(thermo_params, ts_in)
        cp =
            FT(cp_m(thermo_params, Thermodynamics.PhasePartition.(q_atmos(t0))))

        es =
            Thermodynamics.saturation_vapor_pressure.(
                Ref(thermo_params),
                FT.(T_atmos(t0)),
                Ref(Thermodynamics.Liquid()),
            )
        ea =
            Thermodynamics.partial_pressure_vapor.(
                Ref(thermo_params),
                FT.(P_atmos(t0)),
                Thermodynamics.PhasePartition.(FT.(q_atmos(t0))),
            )

        VPD = es .- ea

        conditions = turbulent_fluxes(atmos, canopy, Y, p, t0) #Per unit m^2 of leaf
        r_ae = Array(parent(conditions.r_ae))[1] # s/m
        ga = 1 / r_ae
        γ = FT(66)
        R = FT(LSMP.gas_constant(earth_param_set))
        gs = Array(
            parent(
                ClimaLSM.Canopy.upscale_leaf_conductance.(
                    p.canopy.conductance.gs,
                    LAI,
                    FT.(T_atmos(t0)),
                    R,
                    FT.(P_atmos(t0)),
                ),
            ),
        )[1]
        Lv = FT(2453e6) #J/m^3

        ET = penman_monteith(
            Δ, # Rate of change of saturation specific humidity with air temperature. (Pa K−1)
            Rn, # Net irradiance (W m−2), the external source of energy flux
            G, # Ground heat flux (W m−2)
            ρa, # Dry air density (kg m−3)
            cp, # Specific heat capacity of air (J kg−1 K−1)
            VPD, # vapor pressure deficit (Pa)
            ga, # Conductivity of air, atmospheric conductance (m s−1)
            γ, # Psychrometric constant (γ ≈ 66 Pa K−1)
            gs, # Conductivity of stoma, surface or stomatal conductance (m s−1)
            Lv, # Volumetric latent heat of vaporization. Energy required per water volume vaporized. (Lv = 2453 MJ m−3)
        )

        @test abs(
            (Array(parent(evapotranspiration))[1] - ET) /
            Array(parent(evapotranspiration))[1],
        ) < 0.5

        @test ClimaLSM.surface_evaporative_scaling(canopy, Y, p) == FT(1.0)
        @test ClimaLSM.surface_height(canopy, Y, p) == compartment_faces[1]
        T_sfc = FT.(canopy.atmos.T(t0))
        @test ClimaLSM.surface_temperature(canopy, Y, p, t0) == T_sfc
        @test ClimaLSM.Canopy.canopy_temperature(
            canopy.energy,
            canopy,
            Y,
            p,
            t0,
        ) == T_sfc
        ρ_sfc =
            ClimaLSM.surface_air_density(canopy.atmos, canopy, Y, p, t0, T_sfc)
        @test ClimaLSM.surface_specific_humidity(canopy, Y, p, T_sfc, ρ_sfc) ==
              Thermodynamics.q_vap_saturation_generic.(
            Ref(thermo_params),
            T_sfc,
            ρ_sfc,
            Ref(Thermodynamics.Liquid()),
        )
        @test ρ_sfc == compute_ρ_sfc.(Ref(thermo_params), Ref(ts_in), T_sfc)
    end


    @testset "Component prescribed fields, FT = $FT" begin
        FT = Float32
        # Plant Hydraulics
        LAI = FT(2)
        RAI = FT(1)
        SAI = FT(1)
        lai_fun = t -> LAI * sin(t * 2π / 365)
        ai_parameterization =
            PlantHydraulics.PrescribedSiteAreaIndex{FT}(lai_fun, SAI, RAI)
        K_sat_plant = FT(1.8e-8) # m/s
        ψ63 = FT(-4 / 0.0098) # / MPa to m, Holtzman's original parameter value
        Weibull_param = FT(4) # unitless, Holtzman's original c param value
        a = FT(0.05 * 0.0098) # Holtzman's original parameter for the bulk modulus of elasticity
        plant_ν = FT(0.7) # m3/m3
        plant_S_s = FT(1e-2 * 0.0098) # m3/m3/MPa to m3/m3/m
        conductivity_model =
            PlantHydraulics.Weibull{FT}(K_sat_plant, ψ63, Weibull_param)
        retention_model = PlantHydraulics.LinearRetentionCurve{FT}(a)
        root_depths = SVector{10, FT}(-(10:-1:1.0) ./ 10.0 * 2.0 .+ 0.2 / 2.0) # 1st element is the deepest root depth
        function root_distribution(z::T) where {T}
            return T(1.0 / 0.5) * exp(z / T(0.5)) # (1/m)
        end
        param_set = PlantHydraulics.PlantHydraulicsParameters(;
            ai_parameterization = ai_parameterization,
            ν = plant_ν,
            S_s = plant_S_s,
            root_distribution = root_distribution,
            conductivity_model = conductivity_model,
            retention_model = retention_model,
        )
        Δz = FT(1.0) # height of compartments
        n_stem = Int64(0) # number of stem elements
        n_leaf = Int64(1) # number of leaf elements
        compartment_centers =
            FT.(
                Vector(
                    range(
                        start = Δz / 2,
                        step = Δz,
                        stop = Δz * (n_stem + n_leaf) - (Δz / 2),
                    ),
                ),
            )
        compartment_faces =
            FT.(
                Vector(
                    range(
                        start = 0.0,
                        step = Δz,
                        stop = Δz * (n_stem + n_leaf),
                    ),
                )
            )

        plant_hydraulics = PlantHydraulics.PlantHydraulicsModel{FT}(;
            parameters = param_set,
            n_stem = n_stem,
            n_leaf = n_leaf,
            compartment_surfaces = compartment_faces,
            compartment_midpoints = compartment_centers,
        )

        t0 = FT(100)
        domain = Point(; z_sfc = FT(0.0))
        p = ClimaCore.fill(
            (;
                canopy = (;
                    hydraulics = (;
                        area_index = (
                            leaf = FT(0.0),
                            root = FT(0.0),
                            stem = FT(0.0),
                        )
                    )
                )
            ),
            domain.space.surface,
        )
        # Test that they are set properly
        set_canopy_prescribed_field!(plant_hydraulics, p, t0)
        @test all(
            Array(parent(p.canopy.hydraulics.area_index.leaf)) .==
            FT(LAI * sin(t0 * 2π / 365)),
        )
        @test all(
            Array(parent(p.canopy.hydraulics.area_index.stem)) .== FT(1.0),
        )
        @test all(
            Array(parent(p.canopy.hydraulics.area_index.root)) .== FT(1.0),
        )

        # Test that LAI is updated
        update_canopy_prescribed_field!(plant_hydraulics, p, FT(200))
        @test all(
            Array(parent(p.canopy.hydraulics.area_index.leaf)) .==
            FT(LAI * sin(200 * 2π / 365)),
        )

        struct Default{FT} <: ClimaLSM.Canopy.AbstractCanopyComponent{FT} end
        set_canopy_prescribed_field!(Default{FT}(), p, t0)
        update_canopy_prescribed_field!(Default{FT}(), p, t0)
        # Test that they are unchanged
        @test all(
            parent(p.canopy.hydraulics.area_index.leaf) .==
            FT(LAI * sin(200 * 2π / 365)),
        )
        @test all(
            Array(parent(p.canopy.hydraulics.area_index.stem)) .== FT(1.0),
        )
        @test all(
            Array(parent(p.canopy.hydraulics.area_index.root)) .== FT(1.0),
        )
    end

    @testset "PrescribedSoil, FT = $FT" begin
        soil_driver = PrescribedSoil(FT)
        @test ground_albedo_PAR(soil_driver, nothing, nothing, nothing) ==
              FT(0.2)
        @test ground_albedo_NIR(soil_driver, nothing, nothing, nothing) ==
              FT(0.4)
        @test soil_driver.root_depths ==
              SVector{10, FT}(-(10:-1:1.0) ./ 10.0 * 2.0 .+ 0.2 / 2.0)
        @test FT.(soil_driver.ψ(2.0)) == FT.(0.0)
        @test FT.(soil_driver.T(2.0)) == FT.(298.0)
    end

    @testset "Canopy software pipes with energy model, FT = $FT" begin
        domain = Point(; z_sfc = FT(0.0))

        RTparams = BeerLambertParameters{FT}()
        photosynthesis_params = FarquharParameters{FT}(C3();)
        stomatal_g_params = MedlynConductanceParameters{FT}()

        stomatal_model = MedlynConductanceModel{FT}(stomatal_g_params)
        photosynthesis_model = FarquharModel{FT}(photosynthesis_params)
        rt_model = BeerLambertModel{FT}(RTparams)
        energy_model = BigLeafEnergyModel{FT}(BigLeafEnergyParameters{FT}())
        earth_param_set = create_lsm_parameters(FT)
        LAI = FT(8.0) # m2 [leaf] m-2 [ground]
        z_0m = FT(2.0) # m, Roughness length for momentum - value from tall forest ChatGPT
        z_0b = FT(0.1) # m, Roughness length for scalars - value from tall forest ChatGPT
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
            liquid_precip,
            snow_precip,
            T_atmos,
            u_atmos,
            q_atmos,
            P_atmos,
            ref_time,
            h_atmos;
            c_co2 = c_atmos,
        )
        radiation = PrescribedRadiativeFluxes(
            FT,
            shortwave_radiation,
            longwave_radiation,
            ref_time;
            θs = zenith_angle,
        )

        # Plant Hydraulics
        RAI = FT(1)
        SAI = FT(0)
        lai_fun = t -> LAI
        ai_parameterization =
            PlantHydraulics.PrescribedSiteAreaIndex{FT}(lai_fun, SAI, RAI)
        K_sat_plant = FT(1.8e-8) # m/s
        ψ63 = FT(-4 / 0.0098) # / MPa to m, Holtzman's original parameter value
        Weibull_param = FT(4) # unitless, Holtzman's original c param value
        a = FT(0.05 * 0.0098) # Holtzman's original parameter for the bulk modulus of elasticity
        plant_ν = FT(0.7) # m3/m3
        plant_S_s = FT(1e-2 * 0.0098) # m3/m3/MPa to m3/m3/m
        conductivity_model =
            PlantHydraulics.Weibull{FT}(K_sat_plant, ψ63, Weibull_param)
        retention_model = PlantHydraulics.LinearRetentionCurve{FT}(a)
        root_depths = SVector{10, FT}(-(10:-1:1.0) ./ 10.0 * 2.0 .+ 0.2 / 2.0) # 1st element is the deepest root depth
        function root_distribution(z::T) where {T}
            return T(1.0 / 0.5) * exp(z / T(0.5)) # (1/m)
        end
        param_set = PlantHydraulics.PlantHydraulicsParameters(;
            ai_parameterization = ai_parameterization,
            ν = plant_ν,
            S_s = plant_S_s,
            root_distribution = root_distribution,
            conductivity_model = conductivity_model,
            retention_model = retention_model,
        )
        Δz = FT(1.0) # height of compartments
        n_stem = Int64(0) # number of stem elements
        n_leaf = Int64(1) # number of leaf elements
        compartment_centers =
            FT.(
                Vector(
                    range(
                        start = Δz / 2,
                        step = Δz,
                        stop = Δz * (n_stem + n_leaf) - (Δz / 2),
                    ),
                ),
            )
        compartment_faces =
            FT.(
                Vector(
                    range(
                        start = 0.0,
                        step = Δz,
                        stop = Δz * (n_stem + n_leaf),
                    ),
                )
            )

        ψ_soil0 = FT(0.0)
        T_soil0 = FT(290)
        soil_driver = PrescribedSoil(
            root_depths,
            (t) -> ψ_soil0,
            (t) -> T_soil0,
            FT(0.2),
            FT(0.4),
            FT(0.98),
        )

        plant_hydraulics = PlantHydraulics.PlantHydraulicsModel{FT}(;
            parameters = param_set,
            n_stem = n_stem,
            n_leaf = n_leaf,
            compartment_surfaces = compartment_faces,
            compartment_midpoints = compartment_centers,
        )
        autotrophic_parameters =
            ClimaLSM.Canopy.AutotrophicRespirationParameters{FT}()
        autotrophic_respiration_model =
            ClimaLSM.Canopy.AutotrophicRespirationModel{FT}(
                autotrophic_parameters,
            )

        canopy = ClimaLSM.Canopy.CanopyModel{FT}(;
            parameters = shared_params,
            domain = domain,
            radiative_transfer = rt_model,
            photosynthesis = photosynthesis_model,
            conductance = stomatal_model,
            autotrophic_respiration = autotrophic_respiration_model,
            energy = energy_model,
            hydraulics = plant_hydraulics,
            soil_driver = soil_driver,
            atmos = atmos,
            radiation = radiation,
        )
        @test canopy.radiative_transfer.parameters.ϵ_canopy == FT(0.98)
        @test canopy.energy.parameters.ac_canopy == FT(2.0e3)
        Y, p, coords = ClimaLSM.initialize(canopy)

        # Check that structure of Y is value (will error if not)
        @test !isnothing(zero(Y))
        @test propertynames(p) == (:canopy,)
        for component in ClimaLSM.Canopy.canopy_components(canopy)
            # Only hydraulics has a prognostic variable
            if component == :hydraulics
                @test propertynames(getproperty(Y.canopy, component)) ==
                      ClimaLSM.prognostic_vars(getproperty(canopy, component))
            end
            @test propertynames(getproperty(p.canopy, component)) ==
                  ClimaLSM.auxiliary_vars(getproperty(canopy, component))

            @test getproperty(auxiliary_types(canopy), component) ==
                  auxiliary_types(getproperty(canopy, component))
            @test getproperty(auxiliary_vars(canopy), component) ==
                  auxiliary_vars(getproperty(canopy, component))
            @test getproperty(prognostic_types(canopy), component) ==
                  prognostic_types(getproperty(canopy, component))
            @test getproperty(prognostic_types(canopy), component) ==
                  prognostic_types(getproperty(canopy, component))
        end
        Y.canopy.hydraulics .= plant_ν
        Y.canopy.energy.T = FT(289)

        set_initial_cache! = make_set_initial_cache(canopy)
        t0 = FT(0.0)
        set_initial_cache!(p, Y, t0)
        exp_tendency! = make_exp_tendency(canopy)
        dY = similar(Y)
        exp_tendency!(dY, Y, p, t0)
        (evapotranspiration, shf, lhf) =
            canopy_turbulent_fluxes(canopy.atmos, canopy, Y, p, t0)
        @test p.canopy.hydraulics.fa.:1 == evapotranspiration
        @test p.canopy.energy.lhf == lhf
        @test p.canopy.energy.shf == shf
        @test all(Array(parent(p.canopy.energy.fa_energy_roots)) .== FT(0))

        @test all(
            Array(parent(ClimaLSM.surface_temperature(canopy, Y, p, t0))) .==
            FT(289),
        )
        @test all(
            Array(
                parent(
                    ClimaLSM.Canopy.canopy_temperature(
                        canopy.energy,
                        canopy,
                        Y,
                        p,
                        t0,
                    ),
                ),
            ) .== FT(289),
        )
    end
end
