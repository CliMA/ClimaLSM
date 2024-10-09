using Test
import ClimaComms
ClimaComms.@import_required_backends

using Dates
using Statistics

using ClimaCore
import ClimaParams as CP
using ClimaLand.Bucket:
    BucketModel, BucketModelParameters, PrescribedBaregroundAlbedo
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
    for bucket_domain in bucket_domains
        surface_space = bucket_domain.space.surface
        albedo = PrescribedBaregroundAlbedo{FT}(
            α_snow,
            α_bareground_func,
            surface_space,
        )

        @testset "Zero flux tendency, FT = $FT" begin
            # Radiation
            start_date = DateTime(2005)
            SW_d = (t) -> 0
            LW_d = (t) -> 5.67e-8 * 280.0^4.0
            bucket_rad = PrescribedRadiativeFluxes(
                FT,
                TimeVaryingInput(SW_d),
                TimeVaryingInput(LW_d),
                start_date,
            )
            # Atmos
            precip = (t) -> 0 # no precipitation
            T_atmos = (t) -> 280.0
            u_atmos = (t) -> 1.0
            q_atmos = (t) -> 0.0 # no atmos water
            h_atmos = FT(1e-8)
            P_atmos = (t) -> 101325
            bucket_atmos = PrescribedAtmosphere(
                TimeVaryingInput(precip),
                TimeVaryingInput(precip),
                TimeVaryingInput(T_atmos),
                TimeVaryingInput(u_atmos),
                TimeVaryingInput(q_atmos),
                TimeVaryingInput(P_atmos),
                start_date,
                h_atmos,
                earth_param_set,
            )
            τc = FT(1.0)
            bucket_parameters =
                BucketModelParameters(FT; albedo, z_0m, z_0b, τc)

            model = BucketModel(
                parameters = bucket_parameters,
                domain = bucket_domain,
                atmosphere = bucket_atmos,
                radiation = bucket_rad,
            )
            # Initial conditions with no moisture
            Y, p, coords = initialize(model)
            @test propertynames(p.drivers) == (
                :P_liq,
                :P_snow,
                :T,
                :P,
                :u,
                :q,
                :c_co2,
                :thermal_state,
                :SW_d,
                :LW_d,
                :θs,
            )
            # test if the correct dss buffers were added to aux.
            # We only need to add a dss buffer when there is a horizontal
            # space (spectral element space). So, we check that it is added
            # in the case of the HybridBox and SphericalShell domains,
            # and check that it is not added in the single column case.
            if typeof(model.domain) <: Union{
                ClimaLand.Domains.HybridBox,
                ClimaLand.Domains.SphericalShell,
            }
                @test typeof(p.dss_buffer_3d) == typeof(
                    ClimaCore.Spaces.create_dss_buffer(
                        ClimaCore.Fields.zeros(bucket_domain.space.subsurface),
                    ),
                )
                @test typeof(p.dss_buffer_2d) == typeof(
                    ClimaCore.Spaces.create_dss_buffer(
                        ClimaCore.Fields.zeros(bucket_domain.space.surface),
                    ),
                )
            else
                @test propertynames(p) == (:bucket, :drivers)
            end


            Y.bucket.T .= init_temp.(coords.subsurface.z, FT(280))
            Y.bucket.W .= 0.0 # no moisture
            Y.bucket.Ws .= 0.0 # no runoff
            Y.bucket.σS .= 0.0

            exp_tendency! = make_exp_tendency(model)
            set_initial_cache! = make_set_initial_cache(model)
            set_initial_cache!(p, Y, 0.0)
            dY = similar(Y)
            # init to nonzero numbers
            dY.bucket.T .= init_temp.(coords.subsurface.z, FT(1.0))
            dY.bucket.W .= 1.0
            dY.bucket.Ws .= 1.0
            dY.bucket.σS .= 0.0
            exp_tendency!(dY, Y, p, 0.0)
            @test mean(Array(parent(dY.bucket.T))) < eps(FT)
            @test mean(Array(parent(dY.bucket.W))) < eps(FT)
            @test mean(Array(parent(dY.bucket.Ws))) < eps(FT)
            @test mean(Array(parent(dY.bucket.σS))) < eps(FT)
        end


        @testset "Energy + Moisture Conservation, FT = $FT" begin
            # Radiation
            start_date = DateTime(2005)
            SW_d = (t) -> 10
            LW_d = (t) -> 300
            bucket_rad = PrescribedRadiativeFluxes(
                FT,
                TimeVaryingInput(SW_d),
                TimeVaryingInput(LW_d),
                start_date,
            )
            # Atmos
            precip = (t) -> -1e-6
            precip_snow = (t) -> 0
            T_atmos = (t) -> 298
            u_atmos = (t) -> 4
            q_atmos = (t) -> 0.01
            h_atmos = FT(30)
            P_atmos = (t) -> 101325
            bucket_atmos = PrescribedAtmosphere(
                TimeVaryingInput(precip),
                TimeVaryingInput(precip_snow),
                TimeVaryingInput(T_atmos),
                TimeVaryingInput(u_atmos),
                TimeVaryingInput(q_atmos),
                TimeVaryingInput(P_atmos),
                start_date,
                h_atmos,
                earth_param_set,
            )
            τc = FT(100.0)
            bucket_parameters =
                BucketModelParameters(FT; albedo, z_0m, z_0b, τc)
            model = BucketModel(
                parameters = bucket_parameters,
                domain = bucket_domain,
                atmosphere = bucket_atmos,
                radiation = bucket_rad,
            )

            # Initial conditions with no moisture
            Y, p, coords = initialize(model)
            Y.bucket.T .= init_temp.(coords.subsurface.z, FT(280.0))
            Y.bucket.W .= 0.149
            Y.bucket.Ws .= 0.0
            Y.bucket.σS .= 0.0
            exp_tendency! = make_exp_tendency(model)
            t0 = FT(0.0)
            set_initial_cache! = make_set_initial_cache(model)
            set_initial_cache!(p, Y, t0)
            dY = similar(Y)
            exp_tendency!(dY, Y, p, t0)
            F_water_sfc = FT(precip(t0)) .+ p.bucket.turbulent_fluxes.vapor_flux
            F_sfc =
                p.bucket.turbulent_fluxes.lhf .+
                p.bucket.turbulent_fluxes.shf .- p.bucket.R_n
            surface_space = model.domain.space.surface
            A_sfc = sum(ones(surface_space))

            _T_freeze = LP.T_freeze(earth_param_set)
            _LH_f0 = LP.LH_f0(earth_param_set)
            _ρ_liq = LP.ρ_cloud_liq(earth_param_set)
            _ρLH_f0 = _ρ_liq * _LH_f0

            @test p.bucket.snow_cover_fraction ==
                  ClimaLand.heaviside.(Y.bucket.σS)
            @test p.bucket.F_sfc == F_sfc
            @test p.bucket.partitioned_fluxes ==
                  ClimaLand.Bucket.partition_snow_surface_fluxes.(
                Y.bucket.σS,
                p.bucket.T_sfc,
                model.parameters.τc,
                p.bucket.snow_cover_fraction,
                p.bucket.turbulent_fluxes.vapor_flux,
                p.bucket.F_sfc,
                _ρLH_f0,
                _T_freeze,
            )
            @test p.bucket.G ==
                  F_sfc .* (1 .- p.bucket.snow_cover_fraction) .+
                  p.bucket.partitioned_fluxes.G_under_snow .*
                  p.bucket.snow_cover_fraction
            @test p.bucket.snow_melt ==
                  p.bucket.partitioned_fluxes.F_melt ./ _ρLH_f0
            liquid_precip = p.drivers.P_liq

            @test p.bucket.infiltration ==
                  ClimaLand.Bucket.infiltration_at_point.(
                Y.bucket.W,
                p.bucket.snow_cover_fraction .* p.bucket.snow_melt,
                liquid_precip,
                (1 .- p.bucket.snow_cover_fraction) .*
                p.bucket.turbulent_fluxes.vapor_flux,
                model.parameters.W_f,
            )

            # For the point space, we actually want the flux itself, since our variables are per unit area.
            # Divide by A_sfc
            if typeof(model.domain.space.surface) <: ClimaCore.Spaces.PointSpace
                F_sfc .= F_sfc ./ A_sfc
            end

            @test -sum(F_water_sfc) ≈ sum(dY.bucket.W .+ dY.bucket.Ws)
            @test -sum(F_sfc) ≈ sum(dY.bucket.T .* ρc_soil)
        end
    end
end
