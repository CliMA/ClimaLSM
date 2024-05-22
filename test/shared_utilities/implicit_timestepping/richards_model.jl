using Test
import ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
using LinearAlgebra
using ClimaCore
import ClimaParams as CP
using ClimaLand
using ClimaLand.Domains: Column, HybridBox
using ClimaLand.Soil

import ClimaLand
import ClimaLand.Parameters as LP


for FT in (Float32, Float64)
    @testset "Richards Jacobian entries, Moisture BC, FT = $FT" begin
        ν = FT(0.495)
        K_sat = FT(0.0443 / 3600 / 100) # m/s
        S_s = FT(1e-3) #inverse meters
        vg_n = FT(1.43)
        vg_α = FT(2.6) # inverse meters
        hcm = vanGenuchten{FT}(; α = vg_α, n = vg_n)
        θ_r = FT(0.124)
        zmax = FT(0)
        zmin = FT(-1.5)
        nelems = 150
        soil_domains = [
            Column(; zlim = (zmin, zmax), nelements = nelems),
            HybridBox(;
                xlim = FT.((0, 1)),
                ylim = FT.((0, 1)),
                zlim = (zmin, zmax),
                nelements = (1, 1, nelems),
                npolynomial = 3,
            ),
        ]
        top_state_bc = MoistureStateBC((p, t) -> ν - 1e-3)
        bot_flux_bc = FreeDrainage()
        sources = ()
        boundary_states = (; top = top_state_bc, bottom = bot_flux_bc)
        params = Soil.RichardsParameters(ν, hcm, K_sat, S_s, θ_r)

        for domain in soil_domains
            soil = Soil.RichardsModel{FT}(;
                parameters = params,
                domain = domain,
                boundary_conditions = boundary_states,
                sources = sources,
            )

            Y, p, coords = initialize(soil)
            Y.soil.ϑ_l .= FT(0.24)
            # We do not set the initial aux state here because
            # we want to test that it is updated correctly in
            # the jacobian correctly.
            W = RichardsTridiagonalW(Y)
            Wfact! = make_tendency_jacobian(soil)
            dtγ = FT(1.0)
            Wfact!(W, Y, p, dtγ, FT(0.0))

            K_ic = hydraulic_conductivity(
                hcm,
                K_sat,
                effective_saturation(ν, FT(0.24), θ_r),
            )
            dz = FT(0.01)
            dψdϑ_ic = dψdϑ(hcm, FT(0.24), ν, θ_r, S_s)

            @test all(parent(W.temp2) .== FT(0.0))
            @test all(parent(W.temp2) .== FT(0.0))
            @test W.transform == false
            @test typeof(W.W_column_arrays) <:
                  Vector{LinearAlgebra.Tridiagonal{FT, Vector{FT}}}
            @test length(W.W_column_arrays) == 1
            if typeof(domain) <: Column
                @test all(
                    parent(W.∂ϑₜ∂ϑ.coefs.:1)[2:end] .≈
                    parent(W.∂ϑₜ∂ϑ.coefs.:3)[1:(end - 1)],
                )
                @test Array(parent(W.∂ϑₜ∂ϑ.coefs.:1))[1] == FT(0)
                @test Array(parent(W.∂ϑₜ∂ϑ.coefs.:3))[end] == FT(0)
                @test all(
                    Array(parent(W.∂ϑₜ∂ϑ.coefs.:1))[2:end] .≈
                    K_ic / dz^2 * dψdϑ_ic,
                )
                @test Array(parent(W.∂ϑₜ∂ϑ.coefs.:2))[1] .≈
                      -K_ic / dz^2 * dψdϑ_ic
                @test all(
                    Array(parent(W.∂ϑₜ∂ϑ.coefs.:2))[2:(end - 1)] .≈
                    -2 * K_ic / dz^2 * dψdϑ_ic,
                )
                @test Array(parent(W.∂ϑₜ∂ϑ.coefs.:2))[end] .≈
                      -K_ic / dz^2 * dψdϑ_ic - K_ic / (dz * dz / 2) * dψdϑ_ic
            elseif typeof(domain) <: HybridBox
                @test all(
                    Array(parent(W.∂ϑₜ∂ϑ.coefs.:1))[2:end, :, 1, 1, 1] .≈
                    Array(parent(W.∂ϑₜ∂ϑ.coefs.:3))[1:(end - 1), :, 1, 1, 1],
                )
                @test all(
                    Array(parent(W.∂ϑₜ∂ϑ.coefs.:1))[1, :, 1, 1, 1] .== FT(0),
                )
                @test all(
                    Array(parent(W.∂ϑₜ∂ϑ.coefs.:3))[end, :, 1, 1, 1] .== FT(0),
                )
                @test all(
                    Array(parent(W.∂ϑₜ∂ϑ.coefs.:1))[2:end, :, 1, 1, 1] .≈
                    K_ic / dz^2 * dψdϑ_ic,
                )
                @test all(
                    Array(parent(W.∂ϑₜ∂ϑ.coefs.:2))[1, :, 1, 1, 1] .≈
                    -K_ic / dz^2 * dψdϑ_ic,
                )
                @test all(
                    Array(parent(W.∂ϑₜ∂ϑ.coefs.:2))[2:(end - 1), :, 1, 1, 1] .≈
                    -2 * K_ic / dz^2 * dψdϑ_ic,
                )
                @test all(
                    Array(parent(W.∂ϑₜ∂ϑ.coefs.:2))[end, :, 1, 1, 1] .≈
                    -K_ic / dz^2 * dψdϑ_ic - K_ic / (dz * dz / 2) * dψdϑ_ic,
                )
            end

        end
    end

    @testset "Richards Jacobian entries, Flux BC, FT = $FT" begin
        ν = FT(0.495)
        K_sat = FT(0.0443 / 3600 / 100) # m/s
        S_s = FT(1e-3) #inverse meters
        vg_n = FT(1.43)
        vg_α = FT(2.6) # inverse meters
        hcm = vanGenuchten{FT}(; α = vg_α, n = vg_n)
        θ_r = FT(0.124)
        zmax = FT(0)
        zmin = FT(-1.5)
        nelems = 150
        soil_domains = [
            Column(; zlim = (zmin, zmax), nelements = nelems),
            HybridBox(;
                xlim = FT.((0, 1)),
                ylim = FT.((0, 1)),
                zlim = (zmin, zmax),
                nelements = (1, 1, nelems),
                npolynomial = 3,
            ),
        ]
        top_flux_bc = WaterFluxBC((p, t) -> -K_sat)
        bot_flux_bc = FreeDrainage()
        sources = ()
        boundary_states = (; top = top_flux_bc, bottom = bot_flux_bc)
        params = Soil.RichardsParameters(ν, hcm, K_sat, S_s, θ_r)
        for domain in soil_domains
            soil = Soil.RichardsModel{FT}(;
                parameters = params,
                domain = domain,
                boundary_conditions = boundary_states,
                sources = sources,
            )

            Y, p, coords = initialize(soil)
            Y.soil.ϑ_l .= FT(0.24)
            # We do not set the initial aux state here because
            # we want to test that it is updated correctly in
            # the jacobian correctly.
            W = RichardsTridiagonalW(Y)
            Wfact! = make_tendency_jacobian(soil)
            dtγ = FT(1.0)
            Wfact!(W, Y, p, dtγ, FT(0.0))

            K_ic = hydraulic_conductivity(
                hcm,
                K_sat,
                effective_saturation(ν, FT(0.24), θ_r),
            )
            dz = FT(0.01)
            dψdϑ_ic = dψdϑ(hcm, FT(0.24), ν, θ_r, S_s)
            if typeof(domain) <: Column
                @test Array(parent(W.∂ϑₜ∂ϑ.coefs.:2))[1] .≈
                      -K_ic / dz^2 * dψdϑ_ic
                @test all(
                    Array(parent(W.∂ϑₜ∂ϑ.coefs.:2))[2:(end - 1)] .≈
                    -2 * K_ic / dz^2 * dψdϑ_ic,
                )
                @test Array(parent(W.∂ϑₜ∂ϑ.coefs.:2))[end] .≈
                      -K_ic / dz^2 * dψdϑ_ic
            elseif typeof(domain) <: HybridBox
                @test all(
                    Array(parent(W.∂ϑₜ∂ϑ.coefs.:2))[1, :, 1, 1, 1] .≈
                    -K_ic / dz^2 * dψdϑ_ic,
                )
                @test all(
                    Array(parent(W.∂ϑₜ∂ϑ.coefs.:2))[2:(end - 1), :, 1, 1, 1] .≈
                    -2 * K_ic / dz^2 * dψdϑ_ic,
                )
                @test all(
                    Array(parent(W.∂ϑₜ∂ϑ.coefs.:2))[end, :, 1, 1, 1] .≈
                    -K_ic / dz^2 * dψdϑ_ic,
                )
            end
        end
    end
end
