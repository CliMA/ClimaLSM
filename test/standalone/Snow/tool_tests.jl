using ClimaLand
using Test
using BSON, Dates, HTTP
using DataFrames, CSV, StatsBase, Flux, LinearAlgebra

import ClimaParams as CP
import ClimaUtilities.TimeVaryingInputs: TimeVaryingInput
using Thermodynamics
using SurfaceFluxes
using StaticArrays
using Dates
using ClimaLand.Snow
using ClimaLand.Domains
import ClimaLand.Parameters as LP


try
    import CUDA
    import cuDNN
catch
    nothing
end

DataToolsExt = Base.get_extension(ClimaLand, :NeuralSnowExt)
ModelToolsExt = Base.get_extension(ClimaLand, :NeuralSnowExt)
NeuralSnowExt = Base.get_extension(ClimaLand, :NeuralSnowExt)

if !isnothing(DataToolsExt)
    DataTools = DataToolsExt.DataTools
    ModelTools = ModelToolsExt.ModelTools
    NeuralSnow = NeuralSnowExt.NeuralSnow
    @testset "Testing Data Utilities" begin
        start_date = "2015-01-01"
        end_date = "2015-02-01"
        station_id = 1030
        station_state = "CO"
        real_link = "https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customSingleStationReport/daily/start_of_period/1030:CO:SNTL/2015-01-01,2015-02-01/SNWD::value,"
        @test DataTools.data_url(
            station_id,
            station_state,
            ["SNWD"],
            start_date = start_date,
            end_date = end_date,
        ) == real_link

        colnames = [
            "date",
            "SWE",
            "z",
            "precip",
            "rel_hum_avg",
            "sol_rad_avg",
            "wind_speed_avg",
            "air_temp_avg",
        ]
        # Leave out tests that scrape data, to limit failed builds due to SNOTEL site being down:
        #=
        test_data1 = DataTools.get_data(
            station_id,
            station_state,
            ["SNWD"],
            start = start_date,
            finish = end_date,
        )
        @test typeof(test_data1) == DataFrame
        @test size(test_data1) == (32, 2)
        @test typeof(test_data1[1, 1]) == Date
        @test sum(test_data1[!, 2]) == 1168
        =#

        download_link = "https://caltech.box.com/shared/static/4tih9hiydrc7bcvrpkr2x727b5l54oiq.csv"
        download_data = CSV.read(HTTP.get(download_link).body, DataFrame)
        test_data2 = deepcopy(download_data)
        #=
        test_data2 = DataTools.sitedata_daily(
            station_id,
            station_state,
            start = start_date,
            finish = end_date,
        )
        =#
        @test typeof(test_data2) === DataFrame
        @test size(test_data2) == (32, 8)
        @test DataFrames.names(test_data2) == colnames
        @test typeof(test_data2[1, 1]) == Date
        @test sum(test_data2[!, :z]) == 1168
        #@test isequal(download_data, test_data2)

        download_link_hr = "https://caltech.box.com/shared/static/4q7qtnc2nv8rwh4q8s0afe238ydzfl93.csv"
        download_data = CSV.read(HTTP.get(download_link_hr).body, DataFrame)
        test_data3 = deepcopy(download_data)
        #=
        test_data3 = DataTools.sitedata_hourly(
            station_id,
            station_state,
            start = start_date,
            finish = end_date,
        )
        =#
        test_data3[!, :id] .= station_id
        test_data2[!, :id] .= station_id
        push!(colnames, "id")
        @test typeof(test_data3) === DataFrame
        @test size(test_data3) == (768, 9)
        @test DataFrames.names(test_data3) == colnames
        @test typeof(test_data3[1, 1]) == DateTime
        @test sum(skipmissing(test_data3[!, :z])) == 25063
        @test sum(skipmissing(test_data3[!, :rel_hum_avg])) == 62232
        #@test isequal(download_data, test_data3)

        test_bounds = Dict{Symbol, Tuple{Real, Real}}(
            :z => (0, 37),
            :rel_hum_avg => (84, 100),
        )
        test_data4 = DataTools.apply_bounds(test_data3, test_bounds)
        @test sum(skipmissing(test_data4[!, :z])) == 19007
        @test sum(skipmissing(test_data4[!, :rel_hum_avg])) == 45973

        test_data5 = DataTools.hourly2daily(test_data3)
        @test sum(test_data5[!, :z]) == 1168
        @test sum(test_data5[!, :rel_hum_avg]) == 2593
        @test sum(describe(test_data5, :nmissing)[!, 2]) == 0

        test_data6 = DataTools.rectify_daily_hourly(test_data2, test_data5)
        @test sum(test_data6[!, :z]) == 1168
        @test sum(test_data6[!, :rel_hum_avg]) == 2593
        inch2meter = 0.0254
        kmphr2mps = 5.0 / 18.0
        scales = Dict{Symbol, Real}(
            :SWE => inch2meter,
            :z => inch2meter,
            :precip => inch2meter,
            :rel_hum_avg => 0.01,
            :wind_speed_avg => kmphr2mps,
        )
        test_data7 = DataTools.scale_cols(test_data6, scales)
        @test sum(test_data7[!, :rel_hum_avg]) ≈ 25.93 atol = 1e-3
        test_data8 = DataTools.makediffs(test_data7, Day(1))
        @test size(test_data8) == (31, 12)
        @test mean(test_data8[!, :dzdt]) ≈ 0 atol = 1e-7
        @test mean(test_data8[!, :dSWEdt]) ≈ 0 atol = 1e-7
        @test minimum(test_data8[!, :dprecipdt]) == 0

        test_data9 = DataTools.rolldata(test_data8, Day(1), 7)
        @test size(test_data9) == (25, 12)
        @test mean(test_data9[!, :dzdt]) ≈ 0 atol = 1e-8
        @test mean(test_data9[!, :dSWEdt]) ≈ 0 atol = 1e-8
        @test minimum(test_data9[!, :dprecipdt]) == 0
        @test minimum(test_data9[!, :z]) ≈ 0.8636 atol = 1e-4

        test_data10 = DataTools.prep_data(
            test_data8,
            extract_vars = [:dprecipdt_rain, :dprecipdt_snow],
        )
        x_train, y_train = DataTools.make_data(
            test_data10,
            [:dprecipdt_rain],
            :dprecipdt_snow,
            1.0,
        )
        @test test_data10[!, 1] .+ test_data10[!, 2] ==
              test_data8[!, :dprecipdt]
        @test size(x_train) == (1, 31)
        @test size(y_train) == (1, 31)

        @test DataTools.fix_temp([10.0], alaska = true)[1] ≈ 9.968 atol = 1e-3

        x = ones(366, 7)
        x = Float32.(x)
        x = DataFrame(
            x,
            [
                :SWE,
                :z,
                :precip,
                :rel_hum_avg,
                :sol_rad_avg,
                :wind_speed_avg,
                :air_temp_avg,
            ],
        )
        x[!, :id] .= station_id
        x[!, :date] .= collect(Date("2015-10-01"):Day(1):Date("2016-09-30"))
        x[200, :precip] = 2.0
        x[49, :precip] = 2.0
        x[49, :SWE] = 3.0
        allowmissing!(x)
        x[[3, 4, 10], :sol_rad_avg] .= missing
        x[100, :wind_speed_avg] = 1000.0
        #x1 = DataTools.serreze_qc(x, station_id, station_state)
        #@test size(x1) == (365, 9)
        #@test count(ismissing, x1[!, :air_temp_avg]) == 365
        #@test sum(x1[!, :precip]) == 365
        x1, n1 = DataTools.yan_qc(x)
        @test count(ismissing, x1[!, :SWE]) == 366
        @test n1 == 1.5
        x1 = DataTools.livneh_bc(x)
        @test sum(x1[!, :precip]) == 852
        @test sum(x1[!, :SWE]) == 368
        x1 = DataTools.z_qc(x)
        @test sum(x1) == 153
        x1, x2, x3 = DataTools.manual_filter(x)
        @test sum(x1) == 0
        @test sum(x2) == 30
        @test sum(x3) == 0
        x1 = DataTools.impute_data(x, t1 = 1, t2 = 2, dt = Day(1))
        @test sum(x1) == 366
        x1 = DataTools.qc_filter(x)
        @test sum(x1) == 1
    end

    @testset "Testing Model Utilities" begin
        nfeatures = 7
        n = 5
        pred_vars = [
            :z,
            :SWE,
            :rel_hum_avg,
            :sol_rad_avg,
            :wind_speed_avg,
            :air_temp_avg,
            :dprecipdt_snow,
        ]
        z_idx = 1
        swe_idx = 2
        p_idx = 7
        model = ModelTools.make_model(nfeatures, n, z_idx, p_idx)
        ps = ModelTools.get_model_ps(model)
        for item in ps
            item[:] .= Float32(1.0)
        end

        test_input = Matrix{Float32}(ones(nfeatures, 8))
        @test model(test_input)[1] == 1968
        @test size(model(test_input)) == (1, 8)
        @test sum(length, ps) == nfeatures * (n * (2 * nfeatures + 1) + 2) + 1
        ModelTools.setoutscale!(model, 0.5)
        @test model[:final_scale].weight[3, 3] == 0.5
        @test model(test_input)[1] == 984
        ModelTools.setoutscale!(model, 2.0)
        @test model(test_input)[1] == 1968 #this tests clipping
        @test model(-test_input)[1] == 0.0
        ModelTools.settimescale!(model, 1968)
        @test model[:final_scale].weight[2, 2] == Float32(1 / 1968)
        @test ModelTools.evaluate(model, Vector{Float32}(ones(nfeatures)))[1] ==
              1968

        test_constants = [1.0, 2.0, 3.0, 4.0, 5.0]
        x = zeros(6, 5)
        x[1:5, 1:5] = diagm([1, 2, 3, 4, 5])
        x[6, :] = [1, 1, 1, 1, 1]
        y = [1.0, 4.0, 9.0, 16.0, 25.0, 15.0]
        x_dataframe = DataFrame(x, :auto)
        x_dataframe[!, :y] = y
        answer = [1.0, 2.0, 3.0, 4.0, 5.0, 0.0]
        model2 =
            ModelTools.LinearModel(x_dataframe, [:x1, :x2, :x3, :x4, :x5], :y)
        model3 = ModelTools.LinearModel(x, y)
        @test model2 ≈ answer
        @test model3 ≈ answer
        @test ModelTools.evaluate(model2, Vector{Float32}(ones(5)))[1] == 15
        @test typeof(
            ModelTools.evaluate(model2, Vector{Float32}(ones(5)))[1],
        ) == Float32

        temp(x) = x
        test_loss_check(x, y) = ModelTools.custom_loss(x, y, temp, 2, 1)
        input_x = Vector{Float32}([1, 2, 3, 4, 5])
        input_y = Vector{Float32}([2, 3, 5, 2, 3])
        @test test_loss_check(input_x, input_y) == 11.8f0
        @test typeof(test_loss_check(input_x, input_y)) == Float32

        data_download_link = "https://caltech.box.com/shared/static/1gfyh71c44ljzb9xbnza3lbzj6p9723x.csv"
        modelz_download_link = "https://caltech.box.com/shared/static/ay7cv0rhuiytrqbongpeq2y7m3cimhm4.bson"
        modelswe_download_link = "https://caltech.box.com/shared/static/fe3cghffl0vz3xlzi3jsg3sb2ptyxlai.bson"
        data = CSV.read(HTTP.get(data_download_link).body, DataFrame)
        data = data[data[!, :id] .== 1286, :]
        data = DataTools.prep_data(data)
        zmodel = ModelTools.make_model(nfeatures, 4, z_idx, p_idx)
        swemodel = ModelTools.make_model(nfeatures, 5, swe_idx, p_idx)
        zmodel_state =
            BSON.load(IOBuffer(HTTP.get(modelz_download_link).body))[:zstate]
        swemodel_state =
            BSON.load(IOBuffer(HTTP.get(modelswe_download_link).body))[:swestate]
        Flux.loadmodel!(zmodel, zmodel_state)
        Flux.loadmodel!(swemodel, swemodel_state)
        ModelTools.settimescale!(zmodel, 86400.0)
        ModelTools.settimescale!(swemodel, 86400.0)
        pred_series, _, _ = ModelTools.make_timeseries(zmodel, data, Day(1))
        true_series = data[!, :z]
        test_loss(x, y) = ModelTools.custom_loss(x, y, zmodel, 2, 1)
        series_err =
            sqrt(sum((pred_series .- true_series) .^ 2) ./ length(pred_series))
        direct_err = test_loss(
            Matrix{Float32}(select(data, pred_vars))',
            data[!, :dzdt]',
        )
        @test series_err ≈ 0.15 atol = 0.05
        @test direct_err ≈ 0 atol = 1e-12

        zseries, sweseries, _, _ =
            ModelTools.paired_timeseries(zmodel, swemodel, data, Day(1))
        true_swes = data[!, :SWE]
        zerr = sqrt(sum((zseries .- true_series) .^ 2) ./ length(true_series))
        sweerr = sqrt(sum((sweseries .- true_swes) .^ 2) ./ length(true_series))
        @test zerr ≈ 0.1 atol = 0.05
        @test sweerr ≈ 0.05 atol = 0.03

        out_scale = maximum(abs.(data[!, :dzdt]))
        x_train, y_train =
            DataTools.make_data(data, pred_vars, :dzdt, out_scale)
        ps = ModelTools.get_model_ps(zmodel)
        ModelTools.settimescale!(zmodel, 86400 * out_scale)
        ModelTools.setoutscale!(zmodel, 1.0)
        callback_check = [0.0]
        function call_check(val = callback_check)
            val[1] += 1
        end
        nepochs = 10
        ModelTools.trainmodel!(
            zmodel,
            ps,
            x_train,
            y_train,
            2,
            1,
            nepochs = nepochs,
            cb = call_check,
        )
        @test callback_check[1] == nepochs
        ModelTools.setoutscale!(zmodel, out_scale)
        ModelTools.settimescale!(zmodel, 86400.0)
        pred_series, _, _ = ModelTools.make_timeseries(zmodel, data, Day(1))
        series_err =
            sqrt(sum((pred_series .- true_series) .^ 2) ./ length(pred_series))
        @test series_err <= 0.2
    end

    @testset "Testing NeuralSnow module" begin
        #Model setup:
        FT = Float32
        earth_param_set = LP.LandParameters(FT)
        start_date = DateTime(2005)
        param_set = LP.LandParameters(FT)
        Δt = FT(180.0)
        domain = Point(; z_sfc = FT(0))
        "Radiation"
        SW_d = TimeVaryingInput((t) -> eltype(t)(20.0))
        LW_d = TimeVaryingInput((t) -> eltype(t)(20.0))
        rad = ClimaLand.PrescribedRadiativeFluxes(FT, SW_d, LW_d, start_date)
        "Atmos"
        precip = TimeVaryingInput((t) -> eltype(t)(0.01 / Δt))
        T_atmos = TimeVaryingInput((t) -> eltype(t)(278.0))
        u_atmos = TimeVaryingInput((t) -> eltype(t)(10.0))
        q_atmos = TimeVaryingInput((t) -> eltype(t)(0.03))
        h_atmos = FT(3)
        P_atmos = TimeVaryingInput((t) -> eltype(t)(101325))
        atmos = ClimaLand.PrescribedAtmosphere(
            precip,
            precip,
            T_atmos,
            u_atmos,
            q_atmos,
            P_atmos,
            start_date,
            h_atmos,
            earth_param_set,
        )

        #Test extension utilities
        z_model = NeuralSnow.get_znetwork()
        @test typeof(z_model) <: Flux.Chain
        z32 = NeuralSnow.converted_model_type(z_model, FT)
        @test eltype(z32[1].layers[1].weight) == FT
        z64 = NeuralSnow.converted_model_type(z_model, Float64)
        @test eltype(z64[1].layers[1].weight) == Float64
        dens_model1 = NeuralSnow.NeuralDepthModel(FT)
        @test prod(
            Flux.trainables(dens_model1.z_model) .== Flux.trainables(z32),
        )
        @test eltype(dens_model1.α) == FT
        test_alph = 3 / 86400
        dens_model2 = NeuralSnow.NeuralDepthModel(FT, α = test_alph, Δt = Δt)
        @test dens_model2.α == FT(test_alph)
        @test dens_model2.z_model[:final_scale].weight[2, 2] == FT(1 / Δt)

        parameters = SnowParameters{FT}(
            Δt;
            earth_param_set = param_set,
            density = dens_model2,
        )
        model = ClimaLand.Snow.SnowModel(
            parameters = parameters,
            domain = domain,
            boundary_conditions = ClimaLand.Snow.AtmosDrivenSnowBC(atmos, rad),
        )

        drivers = ClimaLand.get_drivers(model)
        Y, p, coords = ClimaLand.initialize(model)
        @test (Y.snow |> propertynames) ==
              (:S, :U, :Z, :P_avg, :T_avg, :R_avg, :Qrel_avg, :u_avg)

        Y.snow.S .= FT(0.1)
        Y.snow.U .=
            ClimaLand.Snow.energy_from_T_and_swe.(
                Y.snow.S,
                FT(273.0),
                Ref(model.parameters),
            )
        Y.snow.Z .= FT(0.2)
        set_initial_cache! = ClimaLand.make_set_initial_cache(model)
        t0 = FT(0.0)
        set_initial_cache!(p, Y, t0)
        @test snow_depth(model.parameters.density, Y, p, parameters) == Y.snow.Z

        output1 = NeuralSnow.eval_nn(
            dens_model2.z_model,
            FT.([0, 0, 0, 0, 0, 0, 0])...,
        )

        @test eltype(output1) == FT
        @test output1 == 0.0f0

        zerofield = similar(Y.snow.Z)
        zerofield .= FT(0)
        @test NeuralSnow.dzdt(dens_model2, Y) == zerofield

        Z = FT(0.5)
        S = FT(0.1)
        dzdt = FT(1 / Δt)
        dsdt = FT(1 / Δt)
        @test NeuralSnow.clip_dZdt(S, Z, dsdt, dzdt, Δt) == dzdt

        @test NeuralSnow.clip_dZdt(Z, S, dsdt, dzdt, Δt) ≈ FT(1.4 / Δt)

        @test NeuralSnow.clip_dZdt(S, Z, FT(-S / Δt), dzdt, Δt) ≈ FT(-Z / Δt)

        oldρ = p.snow.ρ_snow
        Snow.update_density!(dens_model2, model.parameters, Y, p)
        @test p.snow.ρ_snow == oldρ

        dY = similar(Y)
        dswe_by_precip = 0.1
        Y.snow.P_avg .= FT(dswe_by_precip / Δt)
        NeuralSnow.update_density_prog!(dens_model2, model, dY, Y, p)
        @test parent(dY.snow.Z)[1] * Δt > dswe_by_precip
        new_dYP = FT(test_alph) .* (p.drivers.P_snow .- Y.snow.P_avg)
        @test dY.snow.P_avg == new_dYP
    end
end
