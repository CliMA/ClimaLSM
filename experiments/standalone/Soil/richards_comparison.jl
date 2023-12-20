using Test
using Plots
using DelimitedFiles
using Statistics
using ArtifactWrappers
import SciMLBase
import ClimaTimeSteppers as CTS
using ClimaCore
import CLIMAParameters as CP
using ClimaLSM
using ClimaLSM.Domains: Column
using ClimaLSM.Soil

import ClimaLSM
import ClimaLSM.Parameters as LSMP

include(joinpath(pkgdir(ClimaLSM), "parameters", "create_parameters.jl"))

# Read in reference solutions from artifacts
bonan_clay_dataset = ArtifactWrapper(
    @__DIR__,
    "richards_clay",
    ArtifactFile[ArtifactFile(
        url = "https://caltech.box.com/shared/static/nk89znth59gcsdb65lnywnzjnuno3h6k.txt",
        filename = "clay_bonan_sp801_22323.txt",
    ),],
)
bonan_sand_dataset = ArtifactWrapper(
    @__DIR__,
    "richards_sand",
    ArtifactFile[ArtifactFile(
        url = "https://caltech.box.com/shared/static/2vk7bvyjah8xd5b7wxcqy72yfd2myjss.csv",
        filename = "sand_bonan_sp801.csv",
    ),],
)

@testset "Richards comparison to Bonan; clay" begin
    # Define simulation times
    t0 = Float64(0)
    dt = Float64(1e3)
    tf = Float64(1e6)

    for (FT, tf) in ((Float32, 2 * dt), (Float64, tf))
        ν = FT(0.495)
        K_sat = FT(0.0443 / 3600 / 100) # m/s
        S_s = FT(1e-3) #inverse meters
        vg_n = FT(1.43)
        vg_α = FT(2.6) # inverse meters
        hcm = vanGenuchten(; α = vg_α, n = vg_n)
        θ_r = FT(0.124)
        zmax = FT(0)
        zmin = FT(-1.5)
        nelems = 150
        soil_domain = Column(; zlim = (zmin, zmax), nelements = nelems)
        z = ClimaCore.Fields.coordinate_field(soil_domain.space.subsurface).z

        top_state_bc = MoistureStateBC((p, t) -> ν - 1e-3)
        bot_flux_bc = FreeDrainage()
        sources = ()
        boundary_states =
            (; top = (water = top_state_bc,), bottom = (water = bot_flux_bc,))
        params =
            Soil.RichardsParameters{FT, typeof(hcm)}(ν, hcm, K_sat, S_s, θ_r)

        soil = Soil.RichardsModel{FT}(;
            parameters = params,
            domain = soil_domain,
            boundary_conditions = boundary_states,
            sources = sources,
        )
        set_initial_cache! = make_set_initial_cache(soil)

        Y, p, coords = initialize(soil)

        # specify ICs
        Y.soil.ϑ_l .= FT(0.24)
        exp_tendency! = make_exp_tendency(soil)
        imp_tendency! = ClimaLSM.make_imp_tendency(soil)
        update_jacobian! = ClimaLSM.make_update_jacobian(soil)
        set_initial_cache!(p, Y, t0)

        stepper = CTS.ARS111()
        norm_condition = CTS.MaximumError(FT(1e-8))
        conv_checker = CTS.ConvergenceChecker(; norm_condition = norm_condition)
        ode_algo = CTS.IMEXAlgorithm(
            stepper,
            CTS.NewtonsMethod(
                max_iters = 50,
                update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
                convergence_checker = conv_checker,
            ),
        )

        # set up jacobian info
        jac_kwargs = (;
            jac_prototype = RichardsTridiagonalW(Y),
            Wfact = update_jacobian!,
        )

        prob = SciMLBase.ODEProblem(
            CTS.ClimaODEFunction(
                T_exp! = exp_tendency!,
                T_imp! = SciMLBase.ODEFunction(imp_tendency!; jac_kwargs...),
                dss! = ClimaLSM.dss!,
            ),
            Y,
            (t0, tf),
            p,
        )
        sol = SciMLBase.solve(prob, ode_algo; dt = dt, saveat = 10000)

        # Check that simulation still has correct float type
        @assert eltype(sol.u[end].soil) == FT

        # Check results and plot for Float64 simulation
        if FT == Float64
            N = length(sol.t)
            ϑ_l = parent(sol.u[N].soil.ϑ_l)
            datapath = get_data_folder(bonan_clay_dataset)
            data = joinpath(datapath, "clay_bonan_sp801_22323.txt")
            ds_bonan = readdlm(data)
            bonan_moisture = reverse(ds_bonan[:, 1])
            bonan_z = reverse(ds_bonan[:, 2]) ./ 100.0
            @test sqrt.(mean((bonan_moisture .- ϑ_l) .^ 2.0)) < FT(1e-3)

            plot(ϑ_l, parent(z), label = "Clima")
            plot!(bonan_moisture, bonan_z, label = "Bonan's Matlab code")
            savefig(
                "./experiments/standalone/Soil/comparison_clay_bonan_matlab.png",
            )
        end
    end
end


@testset "Richards comparison to Bonan; sand" begin
    # Define simulation times
    # Note, we can use a bigger step and still conserve mass.
    t0 = Float64(0)
    dt = Float64(1)
    tf = Float64(60 * 60 * 0.8)

    for (FT, tf) in ((Float32, 2 * dt), (Float64, tf))
        ν = FT(0.287)
        K_sat = FT(34 / 3600 / 100) # m/s
        S_s = FT(1e-3) #inverse meters
        vg_n = FT(3.96)
        vg_α = FT(2.7) # inverse meters
        hcm = vanGenuchten(; α = vg_α, n = vg_n)
        θ_r = FT(0.075)
        zmax = FT(0)
        zmin = FT(-1.5)
        nelems = 150
        soil_domain = Column(; zlim = (zmin, zmax), nelements = nelems)
        z = ClimaCore.Fields.coordinate_field(soil_domain.space.subsurface).z

        top_state_bc = MoistureStateBC((p, t) -> 0.267)
        bot_flux_bc = FreeDrainage()
        sources = ()
        boundary_states =
            (; top = (water = top_state_bc,), bottom = (water = bot_flux_bc,))

        params =
            Soil.RichardsParameters{FT, typeof(hcm)}(ν, hcm, K_sat, S_s, θ_r)

        soil = Soil.RichardsModel{FT}(;
            parameters = params,
            domain = soil_domain,
            boundary_conditions = boundary_states,
            sources = sources,
        )

        Y, p, coords = initialize(soil)
        set_initial_cache! = make_set_initial_cache(soil)

        # specify ICs
        Y.soil.ϑ_l .= FT(0.1)
        exp_tendency! = make_exp_tendency(soil)
        imp_tendency! = ClimaLSM.make_imp_tendency(soil)
        update_jacobian! = ClimaLSM.make_update_jacobian(soil)
        set_initial_cache!(p, Y, t0)

        stepper = CTS.ARS111()
        norm_condition = CTS.MaximumError(FT(1e-8))
        conv_checker = CTS.ConvergenceChecker(; norm_condition = norm_condition)
        ode_algo = CTS.IMEXAlgorithm(
            stepper,
            CTS.NewtonsMethod(
                max_iters = 50,
                update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
                convergence_checker = conv_checker,
            ),
        )
        # set up jacobian info
        jac_kwargs = (;
            jac_prototype = RichardsTridiagonalW(Y),
            Wfact = update_jacobian!,
        )

        prob = SciMLBase.ODEProblem(
            CTS.ClimaODEFunction(
                T_exp! = exp_tendency!,
                T_imp! = SciMLBase.ODEFunction(imp_tendency!; jac_kwargs...),
                dss! = ClimaLSM.dss!,
            ),
            Y,
            (t0, tf),
            p,
        )
        sol = SciMLBase.solve(prob, ode_algo; dt = dt, saveat = 60 * dt)

        # Check that simulation still has correct float type
        @assert eltype(sol.u[end].soil) == FT

        # Check results and plot for Float64 simulation
        if FT == Float64
            N = length(sol.t)
            ϑ_l = parent(sol.u[N].soil.ϑ_l)
            datapath = get_data_folder(bonan_sand_dataset)
            data = joinpath(datapath, "sand_bonan_sp801.csv")
            ds_bonan = readdlm(data, ',')
            bonan_moisture = reverse(ds_bonan[:, 1])
            bonan_z = reverse(ds_bonan[:, 2]) ./ 100.0
            @test sqrt.(mean((bonan_moisture .- ϑ_l) .^ 2.0)) < FT(1e-3)

            plot(ϑ_l, parent(z), label = "Clima")
            plot!(bonan_moisture, bonan_z, label = "Bonan's Matlab code")
            savefig(
                "./experiments/standalone/Soil/comparison_sand_bonan_matlab.png",
            )
        end
    end
end
