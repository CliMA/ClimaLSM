using Test
import ClimaComms
ClimaComms.@import_required_backends

using Statistics

using Dates
import ClimaComms
using ClimaCore
using ClimaLand.Bucket:
    BucketModel,
    BucketModelParameters,
    PrescribedBaregroundAlbedo,
    partition_snow_surface_fluxes
using ClimaLand.Domains: coordinates, Column, HybridBox, SphericalShell
using ClimaLand:
    initialize,
    make_exp_tendency,
    make_set_initial_cache,
    PrescribedAtmosphere,
    PrescribedRadiativeFluxes,
    TimeVaryingInput

# Bucket model parameters
import ClimaLand
import ClimaLand.Parameters as LP
import ClimaParams

for FT in (Float32, Float64)
    earth_param_set = LP.LandParameters(FT)
    α_bareground_func = (coordinate_point) -> 0.2 # surface albedo, spatially constant
    α_snow = FT(0.8) # snow albedo
    σS_c = FT(0.2)
    W_f = FT(0.15)
    z_0m = FT(1e-2)
    z_0b = FT(1e-3)
    κ_soil = FT(1.5)
    ρc_soil = FT(2e6)

    # Model domain
    bucket_domains = [
        Column(; zlim = (FT(-100), FT(0)), nelements = 10),
        HybridBox(;
            xlim = (FT(-1), FT(0)),
            ylim = (FT(-1), FT(0)),
            zlim = (FT(-100), FT(0)),
            nelements = (2, 2, 10),
            npolynomial = 1,
            periodic = (true, true),
        ),
        SphericalShell(;
            radius = FT(100),
            depth = FT(3.5),
            nelements = (1, 10),
            npolynomial = 1,
        ),
    ]
    init_temp(z::FT, value::FT) where {FT} = FT(value)

    @testset "Small negative values FT = $FT " begin
        d = bucket_domains[1]
        start_date = DateTime(2005)
        SW_d = (t) -> 20
        LW_d = (t) -> 20
        bucket_rad = PrescribedRadiativeFluxes(
            FT,
            TimeVaryingInput(SW_d),
            TimeVaryingInput(LW_d),
            start_date,
        )
        # Atmos
        liquid_precip = (t) -> 0.0 # precipitation
        snow_precip = (t) -> 0.0 # precipitation

        T_atmos = (t) -> 280.0
        u_atmos = (t) -> 10.0
        q_atmos = (t) -> 0.03
        h_atmos = FT(3)
        P_atmos = (t) -> 101325 # Pa
        bucket_atmos = PrescribedAtmosphere(
            TimeVaryingInput(liquid_precip),
            TimeVaryingInput(snow_precip),
            TimeVaryingInput(T_atmos),
            TimeVaryingInput(u_atmos),
            TimeVaryingInput(q_atmos),
            TimeVaryingInput(P_atmos),
            start_date,
            h_atmos,
            earth_param_set,
        )

        τc = FT(10.0)
        surface_space = d.space.surface
        albedo = PrescribedBaregroundAlbedo{FT}(
            α_snow,
            α_bareground_func,
            surface_space,
        )
        bucket_parameters = BucketModelParameters(FT; albedo, z_0m, z_0b, τc)
        model = BucketModel(
            parameters = bucket_parameters,
            domain = d,
            atmosphere = bucket_atmos,
            radiation = bucket_rad,
        )
        drivers = ClimaLand.get_drivers(model)
        @test drivers == (bucket_atmos, bucket_rad)
        _LH_f0 = LP.LH_f0(model.parameters.earth_param_set)
        _ρ_liq = LP.ρ_cloud_liq(model.parameters.earth_param_set)
        _ρLH_f0 = _ρ_liq * _LH_f0 # Latent heat per unit volume
        _T_freeze = LP.T_freeze(model.parameters.earth_param_set)

        # run for some time
        Y, p, coords = initialize(model)
        Y.bucket.T .= init_temp.(coords.subsurface.z, FT(274.0))
        Y.bucket.W .= -0.1 # small negative moisture
        Y.bucket.Ws .= 0.0 # no runoff
        Y.bucket.σS .= -0.1 # small negative snow
        t0 = FT(0.0)
        dY = similar(Y)

        exp_tendency! = make_exp_tendency(model)
        set_initial_cache! = make_set_initial_cache(model)
        set_initial_cache!(p, Y, t0)
        # Test that the albedo is the bareground albedo (negative snow treated as zero snow)
        @test p.bucket.α_sfc ==
              FT.(
            α_bareground_func.(
                ClimaCore.Fields.coordinate_field(surface_space)
            )
        )
        # Test that evaporation is zero
        @test all(parent(p.bucket.turbulent_fluxes.vapor_flux) .== FT(0.0))
        # Test that q_sfc is computed as if there was zero snow
        @test p.bucket.q_sfc ==
              ClimaLand.Bucket.saturation_specific_humidity.(
            p.bucket.T_sfc,
            p.bucket.ρ_sfc,
            earth_param_set.thermo_params,
        )
        # Test that beta factor is zero despite negative water content
        @test all(
            parent(
                ClimaLand.Bucket.beta_factor.(
                    Y.bucket.W,
                    Y.bucket.σS,
                    model.parameters.W_f * model.parameters.f_bucket,
                    model.parameters.σS_c * model.parameters.f_snow,
                    model.parameters.p,
                ),
            ) .== FT(0.0),
        )
        F_sfc =
            @. p.bucket.turbulent_fluxes.lhf + p.bucket.turbulent_fluxes.shf -
               p.bucket.R_n
        partitioned_fluxes = @. ClimaLand.Bucket.partition_snow_surface_fluxes(
            Y.bucket.σS,
            p.bucket.T_sfc,
            model.parameters.τc,
            ClimaLand.heaviside(Y.bucket.σS),
            p.bucket.turbulent_fluxes.vapor_flux,
            F_sfc,
            _ρLH_f0,
            _T_freeze,
        )
        @test partitioned_fluxes.F_melt == ClimaCore.Fields.zeros(surface_space)
        @test partitioned_fluxes.F_into_snow ==
              ClimaCore.Fields.zeros(surface_space)
        @test partitioned_fluxes.G_under_snow == F_sfc
        exp_tendency!(dY, Y, p, t0)
        # Test that all water tendencies are zero (no precip, no snowmelt, no vapor loss)
        @test dY.bucket.W == ClimaCore.Fields.zeros(surface_space)
        @test dY.bucket.Ws == ClimaCore.Fields.zeros(surface_space)
        @test dY.bucket.σS == ClimaCore.Fields.zeros(surface_space)
    end

    for i in 1:3
        @testset "Conservation of water and energy I (snow present), FT = $FT" begin
            # Radiation
            start_date = DateTime(2005)
            SW_d = (t) -> 20
            LW_d = (t) -> 20
            bucket_rad = PrescribedRadiativeFluxes(
                FT,
                TimeVaryingInput(SW_d),
                TimeVaryingInput(LW_d),
                start_date,
            )
            # Atmos
            liquid_precip = (t) -> -1e-8 # precipitation
            snow_precip = (t) -> -1e-7 # precipitation

            T_atmos = (t) -> 280.0
            u_atmos = (t) -> 10.0
            q_atmos = (t) -> 0.03
            h_atmos = FT(3)
            P_atmos = (t) -> 101325 # Pa
            bucket_atmos = PrescribedAtmosphere(
                TimeVaryingInput(liquid_precip),
                TimeVaryingInput(snow_precip),
                TimeVaryingInput(T_atmos),
                TimeVaryingInput(u_atmos),
                TimeVaryingInput(q_atmos),
                TimeVaryingInput(P_atmos),
                start_date,
                h_atmos,
                earth_param_set,
            )

            τc = FT(10.0)
            surface_space = bucket_domains[i].space.surface
            albedo = PrescribedBaregroundAlbedo{FT}(
                α_snow,
                α_bareground_func,
                surface_space,
            )
            bucket_parameters =
                BucketModelParameters(FT; albedo, z_0m, z_0b, τc)
            model = BucketModel(
                parameters = bucket_parameters,
                domain = bucket_domains[i],
                atmosphere = bucket_atmos,
                radiation = bucket_rad,
            )

            # run for some time
            Y, p, coords = initialize(model)
            Y.bucket.T .= init_temp.(coords.subsurface.z, FT(274.0))
            Y.bucket.W .= 0.0 # no moisture
            Y.bucket.Ws .= 0.0 # no runoff
            Y.bucket.σS .= 0.5
            t0 = FT(0.0)
            dY = similar(Y)

            exp_tendency! = make_exp_tendency(model)
            set_initial_cache! = make_set_initial_cache(model)
            set_initial_cache!(p, Y, t0)
            exp_tendency!(dY, Y, p, t0)
            _LH_f0 = LP.LH_f0(model.parameters.earth_param_set)
            _ρ_liq = LP.ρ_cloud_liq(model.parameters.earth_param_set)
            _ρLH_f0 = _ρ_liq * _LH_f0 # Latent heat per unit volume
            _T_freeze = LP.T_freeze(model.parameters.earth_param_set)
            function snow_cover_fraction(σS)
                return σS > eps(typeof(σS)) ? typeof(σS)(1.0) : typeof(σS)(0.0)
            end

            partitioned_fluxes =
                partition_snow_surface_fluxes.(
                    Y.bucket.σS,
                    p.bucket.T_sfc,
                    model.parameters.τc,
                    snow_cover_fraction.(Y.bucket.σS),
                    p.bucket.turbulent_fluxes.vapor_flux,
                    p.bucket.turbulent_fluxes.lhf .+
                    p.bucket.turbulent_fluxes.shf .- p.bucket.R_n,
                    _ρLH_f0,
                    _T_freeze,
                )
            F_melt = partitioned_fluxes.F_melt
            F_into_snow =
                partitioned_fluxes.F_into_snow .- _ρLH_f0 .* FT(snow_precip(t0))
            G = partitioned_fluxes.G_under_snow
            F_sfc =
                p.bucket.turbulent_fluxes.lhf .+
                p.bucket.turbulent_fluxes.shf .- p.bucket.R_n .-
                _ρLH_f0 .* FT(snow_precip(t0))
            F_water_sfc =
                FT(liquid_precip(t0)) + FT(snow_precip(t0)) .+
                p.bucket.turbulent_fluxes.vapor_flux

            if i == 1
                A_point = sum(ones(bucket_domains[i].space.surface))
            else
                A_point = 1
            end

            dIsnow = -_ρLH_f0 .* dY.bucket.σS
            @test sum(dIsnow) / A_point ≈ sum(-1 .* F_into_snow) / A_point

            de_soil = dY.bucket.T .* ρc_soil
            @test sum(de_soil) ≈ sum(-1 .* G) / A_point

            dWL = dY.bucket.W .+ dY.bucket.Ws .+ dY.bucket.σS
            @test sum(dWL) / A_point ≈ -sum(F_water_sfc) / A_point

            dIL = sum(dIsnow) / A_point .+ sum(de_soil)
            @test dIL ≈ sum(-1 .* F_sfc) / A_point
        end
    end

    for i in 1:3
        @testset "Conservation of water and energy II (no snow to start), FT = $FT" begin
            # Radiation
            start_date = DateTime(2005)
            SW_d = (t) -> 20
            LW_d = (t) -> 20
            bucket_rad = PrescribedRadiativeFluxes(
                FT,
                TimeVaryingInput(SW_d),
                TimeVaryingInput(LW_d),
                start_date,
            )
            # Atmos
            liquid_precip = (t) -> -1e-8 # precipitation
            snow_precip = (t) -> -1e-7 # precipitation

            T_atmos = (t) -> 280
            u_atmos = (t) -> 10
            q_atmos = (t) -> 0.03
            h_atmos = FT(3)
            P_atmos = (t) -> 101325 # Pa
            bucket_atmos = PrescribedAtmosphere(
                TimeVaryingInput(liquid_precip),
                TimeVaryingInput(snow_precip),
                TimeVaryingInput(T_atmos),
                TimeVaryingInput(u_atmos),
                TimeVaryingInput(q_atmos),
                TimeVaryingInput(P_atmos),
                start_date,
                h_atmos,
                earth_param_set,
            )

            τc = FT(10.0)
            surface_space = bucket_domains[i].space.surface
            albedo = PrescribedBaregroundAlbedo{FT}(
                α_snow,
                α_bareground_func,
                surface_space,
            )
            bucket_parameters =
                BucketModelParameters(FT; albedo, z_0m, z_0b, τc)
            model = BucketModel(
                parameters = bucket_parameters,
                domain = bucket_domains[i],
                atmosphere = bucket_atmos,
                radiation = bucket_rad,
            )

            # run for some time
            Y, p, coords = initialize(model)
            Y.bucket.T .= init_temp.(coords.subsurface.z, FT(274.0))
            Y.bucket.W .= 0.0 # no moisture
            Y.bucket.Ws .= 0.0 # no runoff
            Y.bucket.σS .= 0.0
            t0 = FT(0.0)
            dY = similar(Y)

            exp_tendency! = make_exp_tendency(model)
            set_initial_cache! = make_set_initial_cache(model)
            set_initial_cache!(p, Y, t0)
            exp_tendency!(dY, Y, p, t0)
            _LH_f0 = LP.LH_f0(model.parameters.earth_param_set)
            _ρ_liq = LP.ρ_cloud_liq(model.parameters.earth_param_set)
            _ρLH_f0 = _ρ_liq * _LH_f0 # Latent heat per unit volume
            _T_freeze = LP.T_freeze(model.parameters.earth_param_set)
            function snow_cover_fraction(σS)
                return σS > eps(typeof(σS)) ? typeof(σS)(1.0) : typeof(σS)(0.0)
            end

            partitioned_fluxes =
                partition_snow_surface_fluxes.(
                    Y.bucket.σS,
                    p.bucket.T_sfc,
                    model.parameters.τc,
                    snow_cover_fraction.(Y.bucket.σS),
                    p.bucket.turbulent_fluxes.vapor_flux,
                    p.bucket.turbulent_fluxes.lhf .+
                    p.bucket.turbulent_fluxes.shf .- p.bucket.R_n,
                    _ρLH_f0,
                    _T_freeze,
                )
            F_melt = partitioned_fluxes.F_melt
            F_into_snow =
                partitioned_fluxes.F_into_snow .- _ρLH_f0 .* FT(snow_precip(t0))
            G = partitioned_fluxes.G_under_snow
            F_sfc =
                p.bucket.turbulent_fluxes.lhf .+
                p.bucket.turbulent_fluxes.shf .- p.bucket.R_n .-
                _ρLH_f0 .* FT(snow_precip(t0))
            F_water_sfc =
                FT(liquid_precip(t0)) + FT(snow_precip(t0)) .+
                p.bucket.turbulent_fluxes.vapor_flux

            if i == 1
                A_point = sum(ones(bucket_domains[i].space.surface))
            else
                A_point = 1
            end

            dIsnow = -_ρLH_f0 .* dY.bucket.σS
            @test sum(dIsnow) / A_point ≈ sum(-1 .* F_into_snow) / A_point

            de_soil = dY.bucket.T .* ρc_soil
            @test sum(de_soil) ≈ sum(-1 .* G) / A_point

            dWL = dY.bucket.W .+ dY.bucket.Ws .+ dY.bucket.σS
            @test sum(dWL) / A_point ≈ -sum(F_water_sfc) / A_point

            dIL = sum(dIsnow) / A_point .+ sum(de_soil)
            @test dIL ≈ sum(-1 .* F_sfc) / A_point
        end
    end

    @testset "Conservation of water and energy - nonuniform evaporation, FT = $FT" begin
        i = 3
        # Radiation
        start_date = DateTime(2005)
        SW_d = (t) -> 20
        LW_d = (t) -> 20
        bucket_rad = PrescribedRadiativeFluxes(
            FT,
            TimeVaryingInput(SW_d),
            TimeVaryingInput(LW_d),
            start_date,
        )
        # Atmos
        liquid_precip = (t) -> -1e-8 # precipitation
        snow_precip = (t) -> -1e-7 # precipitation

        T_atmos = (t) -> 280
        u_atmos = (t) -> 10
        q_atmos = (t) -> 0.03
        h_atmos = FT(3)
        P_atmos = (t) -> 101325 # Pa
        bucket_atmos = PrescribedAtmosphere(
            TimeVaryingInput(liquid_precip),
            TimeVaryingInput(snow_precip),
            TimeVaryingInput(T_atmos),
            TimeVaryingInput(u_atmos),
            TimeVaryingInput(q_atmos),
            TimeVaryingInput(P_atmos),
            start_date,
            h_atmos,
            earth_param_set,
        )

        τc = FT(10.0)
        surface_space = bucket_domains[i].space.surface
        albedo = PrescribedBaregroundAlbedo{FT}(
            α_snow,
            α_bareground_func,
            surface_space,
        )
        bucket_parameters = BucketModelParameters(FT; albedo, z_0m, z_0b, τc)
        model = BucketModel(
            parameters = bucket_parameters,
            domain = bucket_domains[i],
            atmosphere = bucket_atmos,
            radiation = bucket_rad,
        )

        # run for some time
        Y, p, coords = initialize(model)
        Y.bucket.T .= init_temp.(coords.subsurface.z, FT(274.0))
        Y.bucket.W .= 0.0 # no moisture
        Y.bucket.Ws .= 0.0 # no runoff
        Y.bucket.σS .= 0.0
        t0 = FT(0.0)
        dY = similar(Y)

        compute_exp_tendency! = ClimaLand.make_compute_exp_tendency(model)
        set_initial_cache! = make_set_initial_cache(model)
        set_initial_cache!(p, Y, t0)
        random = zeros(bucket_domains[i].space.surface)
        ArrayType = ClimaComms.array_type(Y)
        parent(random) .= ArrayType(rand(FT, size(parent(random))))
        p.bucket.turbulent_fluxes.vapor_flux .= random .* 1e-7
        compute_exp_tendency!(dY, Y, p, t0)
        _LH_f0 = LP.LH_f0(model.parameters.earth_param_set)
        _ρ_liq = LP.ρ_cloud_liq(model.parameters.earth_param_set)
        _ρLH_f0 = _ρ_liq * _LH_f0 # Latent heat per unit volume
        _T_freeze = LP.T_freeze(model.parameters.earth_param_set)
        function snow_cover_fraction(σS)
            return σS > eps(typeof(σS)) ? typeof(σS)(1.0) : typeof(σS)(0.0)
        end
        σ_snow = snow_cover_fraction.(Y.bucket.σS)
        F_sfc =
            @. p.bucket.turbulent_fluxes.lhf + p.bucket.turbulent_fluxes.shf -
               p.bucket.R_n
        partitioned_fluxes =
            partition_snow_surface_fluxes.(
                Y.bucket.σS,
                p.bucket.T_sfc,
                model.parameters.τc,
                σ_snow,
                p.bucket.turbulent_fluxes.vapor_flux,
                F_sfc,
                _ρLH_f0,
                _T_freeze,
            )
        F_melt = partitioned_fluxes.F_melt
        F_into_snow = @. partitioned_fluxes.F_into_snow * σ_snow -
           _ρLH_f0 * FT(snow_precip(t0))
        G = @. partitioned_fluxes.G_under_snow * σ_snow + (1 - σ_snow) * F_sfc
        F_water_sfc =
            FT(liquid_precip(t0)) + FT(snow_precip(t0)) .+
            p.bucket.turbulent_fluxes.vapor_flux

        dIsnow = -_ρLH_f0 .* dY.bucket.σS
        @test sum(dIsnow) ≈ sum(-1 .* F_into_snow)

        de_soil = dY.bucket.T .* ρc_soil
        @test sum(de_soil) ≈ sum(-1 .* G)

        dWL = dY.bucket.W .+ dY.bucket.Ws .+ dY.bucket.σS
        @test sum(dWL) ≈ -sum(F_water_sfc)

        dIL = sum(dIsnow) .+ sum(de_soil)
        @test dIL ≈ sum(-1 .* F_sfc .- _ρLH_f0 .* dY.bucket.σS)
    end
end
