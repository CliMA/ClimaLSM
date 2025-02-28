module ModelTools
using Flux, LinearAlgebra
using DataFrames, Dates
export make_model,
    get_model_ps,
    settimescale!,
    setoutscale!,
    LinearModel,
    make_timeseries,
    trainmodel!,
    custom_loss,
    evaluate,
    paired_timeseries

"""
    make_model(nfeatures, n, z_idx, p_idx; in_scale, dtype)

Create the neural network to be trained, with initial scaling weights.

# Arguments
- `nfeatures::Int`: indicates number of features.
- `n::Int`: the value of the hyperparameter n.
- `z_idx::Int`: The index of the data vectors pertaining to the depth (z) values.
- `p_idx::Int`: The index of the data vectors pertaining to the precipitation values.
-  `in_scale::Vector{<:Real}`: Optional scaling constants for each input feature.
- `dtype::Type`: Sets type of output model. Default is `Float32`.
"""
function make_model(
    nfeatures::Int,
    n::Int,
    z_idx::Int,
    p_idx::Int;
    in_scale::Union{Vector{<:Real}, Nothing} = nothing,
    dtype::Type = Float32,
)
    in_scales =
        (isnothing(in_scale)) ? Matrix{dtype}(diagm(ones(nfeatures))) :
        Matrix{dtype}(diagm((1.0 ./ in_scale)))
    get_relus = Matrix{dtype}([1 0 0; 0 1 0; 1 0 -1])
    get_min = Matrix{dtype}([1 1 -1; 0 1 0])
    get_max = Matrix{dtype}([1 -1])
    model = Chain(
        pred = SkipConnection(
            Chain(
                scale = Dense(in_scales, false, identity),
                l1 = Dense(nfeatures, n * nfeatures, relu),
                l2 = Dense(n * nfeatures, nfeatures, elu),
                l3 = Dense(nfeatures, 1),
            ),
            vcat, #returns [predicted_value, input...]
        ),
        get_boundaries = Parallel(
            vcat,
            up_bound = x -> relu.(x[1, :])' .* (x[p_idx + 1, :] .> 0)', # = upper = relu(upper)
            low_bound = x -> x[z_idx + 1, :]', # = z = relu(z)
            output_pos = x -> x[1, :]',
        ),
        final_scale = Dense(
            Matrix{dtype}(diagm([1.0, 1.0, 1.0])),
            false,
            identity,
        ),
        #output_no_thresh = x -> x[3, :]'
        apply_relus = Dense(get_relus, false, relu),  #returns relu(upper) = upper, relu(z) = z, relu(upper - pred_z)
        apply_upper = Dense(get_min, false, relu), #returns relu(min(pred_z, upper)+z), relu(z) = z
        apply_lower = Dense(get_max, false, identity), #returns relu(min(pred_z, upper)+z) - relu(z) = max(min(pred_Z, upper), lower)
    )
    return model
end

"""
    get_model_ps(model)

Return the trainable weights for the developed neural model.

# Arguments
- `model::Chain`: the neural model to be used.
"""
function get_model_ps(model::Chain)
    return Flux.trainables(Flux.trainables(model[:pred])[2:7])
end

"""
    settimescale!(model, dt; dtype)

Set the timescale parameter for model usage.

# Arguments
- `model::Chain`: the neural model to be used.
- `dt::Real`: the number of seconds per timestep for usage.
- `dtype::Type`: Sets type, consistent with neural model. Default is `Float32`.
"""
function settimescale!(model, dt::Real; dtype::Type = Float32)
    model[:final_scale].weight[2, 2] = dtype(1.0 / dt)
end

"""
    setoutscale!(model, scale; dtype)

Set the physical scaling parameter for model usage (i.e. rectifying scaling done on model input).

# Arguments
- `model::Chain`: the neural model to be used.
- `scale::Real`: the scaling parameter to return data to applicable units.
- `dtype::Type`: Sets type, consistent with neural model. Default is `Float32`.
"""
function setoutscale!(model, scale::Real; dtype::Type = Float32)
    model[:final_scale].weight[3, 3] = dtype(scale)
end

"""
    LinearModel(data, vars, target; dtype, scale_const)

Create a linear regression model on a data frame for comparison to neural model.
Returns the coefficients for the model.

# Arguments
- `data::DataFrame`: The data set to be utilized.
- `vars::Vector{Symbol}`: The input variables to be used in the model creation.
-  `target::Symbol`: The target variable to be used in the model ceation.
- `dtype::Type`: Sets type, consistent with neural model. Default is `Float32`.
- `scale_const`: Optional scaling constant for model output. Default is 1.0.
"""
function LinearModel(
    data::DataFrame,
    vars::Vector{Symbol},
    target::Symbol;
    dtype::Type = Float32,
    scale_const = 1.0,
)
    X = Matrix{dtype}(select(data, vars))
    y = Vector{dtype}(data[!, target]) ./ dtype(scale_const)
    constants = [X ones(nrow(data))] \ y
    return Vector{dtype}(constants .* scale_const)
end

"""
    LinearModel(x_train, y_train; dtype, scale_const)

Create a linear regression model on a training matrix for comparison to neural model.
Returns the coefficients for the model.
**Note: using the same matrices input to the neural model will require a transpose of `x_train`, `y_train`

# Arguments
- `x_train::Matrix`: The input to be utilized.
- `y_train::Vector`: The output data to be utilized.
- `dtype::Type`: Sets type, consistent with neural model. Default is `Float32`.
- `scale_const`: Optional scaling constant for model output. Default is 1.0.
"""
function LinearModel(
    x_train::Matrix,
    y_train::Vector;
    dtype::Type = Float32,
    scale_const = 1.0,
)
    #using x_train from neural input will require a transpose of x_train, y_train
    return Vector{dtype}(
        ([x_train ones(size(x_train)[1])] \ y_train) .* scale_const,
    )
end

"""
    evaluate(model, input)

Evaluate a created model on a given input vector.

# Arguments
- `model::Chain`: A neural model to be used for prediction.
- `input`: The input data used to generate a prediction.
"""
function evaluate(model::Chain, input)
    return model(input)
end

"""
    evaluate(model, input)

Evaluate a created model on a given input vector.

# Arguments
- `model::Vector{<:Real}`: Linear regression coefficients used for prediction.
- `input`: The input data used to generate a prediction.
"""
function evaluate(model::Vector{<:Real}, input)
    #requires input matrix to be the same orientation as that for the neural model
    return model[1:(end - 1)]' * input .+ model[end]
end


"""
    make_timeseries(model, timeseries, dt; predictvar, timevar, inputvars, dtype, hole_thresh)

Generate a predicted timeseries given forcing data and the timestep present in that data (holes acceptable).

# Arguments
- `model`: The model used for forecasting (can be any model with a defined `evaluate` call).
- `timeseries::DataFrame`: The input data frame used to generate predictions, including a time variable.
- `dt::Period`: The unit timestep present in the dataframe (i.e. daily dataframe, `dt = Day(1)` or `Second(86400)`).
- `predictvar::Symbol`: The variable to predict from the timeseries. Default is `:z`.
- `timevar::Symbol`: The variable giving the time of each forcing. Default is `:date`.
- `inputvars::Vector{Symbol}`: The variables (in order), to extract from the data to use for predictions.
Default is `[:z, :SWE, :rel_hum_avg, :sol_rad_avg, :wind_speed_avg, :air_temp_avg, :dprecipdt_snow]` like the paper.
- `dtype::Type`: The data type required for input to the model. Default is `Float32`.
- `hole_thresh::Int`: The acceptable number of "holes" in the timeseries for the model to skip over. Default is 30.
"""
function make_timeseries(
    model,
    timeseries::DataFrame,
    dt::Period;
    predictvar::Symbol = :z,
    timevar::Symbol = :date,
    inputvars::Vector{Symbol} = [
        :z,
        :SWE,
        :rel_hum_avg,
        :sol_rad_avg,
        :wind_speed_avg,
        :air_temp_avg,
        :dprecipdt_snow,
    ],
    dtype::Type = Float32,
    hole_thresh::Int = 30,
)
    forcings = Matrix{dtype}(select(timeseries, inputvars))
    pred_idx = findfirst(inputvars .== predictvar)
    check_dates =
        (timeseries[2:end, timevar] - timeseries[1:(end - 1), timevar]) ./ dt
    pred_vals = zeros(nrow(timeseries) - 1)
    pred_series = zeros(nrow(timeseries))
    pred_series[1] = timeseries[1, predictvar]
    countresets = 0
    for j in 2:length(pred_series)
        input = forcings[j - 1, :]
        input[pred_idx] = pred_series[j - 1]
        pred = evaluate(model, input)[1]
        pred_vals[j - 1] = pred
        nperiods = check_dates[j - 1]
        new_val = pred_series[j - 1] + nperiods * Dates.value(Second(dt)) * pred
        pred_series[j] =
            (nperiods <= hole_thresh) ? max(0.0, new_val) :
            timeseries[j, predictvar]  #the "max" is only in the case of holes for a non-negative system, and does not generalize
        #pred_series[j] = (nperiods <= hole_thresh) ? new_val : true_series[j]  #for showing no thresholds
        if nperiods > hole_thresh
            countresets += 1
        end
    end
    return pred_series, pred_vals, countresets
end

"""
    paired_timeseries(zmodel, swemodel, timeseries, dt; timevar, z_inputvars, SWE_inputvars, dtype, hole_thresh)

Generate a predicted timeseries given forcing data and the timestep present in that data (holes acceptable), with a coupled
z and SWE model. Used in the paper for combined timeseries.

# Arguments
- `zmodel`: The model used for depth forecasting (can be any model with a defined `evaluate` call).
- `swemodel`: The model used for SWE forecasting (can be any model with a defined `evaluate` call).
- `timeseries::DataFrame`: The input data frame used to generate predictions, including a time variable.
- `dt::Period`: The unit timestep present in the dataframe (i.e. daily dataframe, `dt = Day(1)` or `Second(86400)`).
- `timevar::Symbol`: The variable giving the time of each forcing. Default is `:date`.
- `z_inputvars::Vector{Symbol}`: The variables (in order), to extract from the data to use for depth predictions.
Default is `[:z, :SWE, :rel_hum_avg, :sol_rad_avg, :wind_speed_avg, :air_temp_avg, :dprecipdt_snow]` like the paper.
- `SWE_inputvars::Vector{Symbol}`: The variables (in order), to extract from the data to use for SWE predictions.
Default is `[:z, :SWE, :rel_hum_avg, :sol_rad_avg, :wind_speed_avg, :air_temp_avg, :dprecipdt_snow]` like the paper.
- `dtype::Type`: The data type required for input to the model. Default is `Float32`.
- `hole_thresh::Int`: The acceptable number of "holes" in the timeseries for the model to skip over. Default is 30.
"""
function paired_timeseries(
    zmodel,
    swemodel,
    timeseries::DataFrame,
    dt::Period;
    timevar::Symbol = :date,
    z_inputvars::Vector{Symbol} = [
        :z,
        :SWE,
        :rel_hum_avg,
        :sol_rad_avg,
        :wind_speed_avg,
        :air_temp_avg,
        :dprecipdt_snow,
    ],
    SWE_inputvars::Vector{Symbol} = [
        :z,
        :SWE,
        :rel_hum_avg,
        :sol_rad_avg,
        :wind_speed_avg,
        :air_temp_avg,
        :dprecipdt_snow,
    ],
    dtype::Type = Float32,
    hole_thresh::Int = 30,
)
    zforcings = Matrix{dtype}(select(timeseries, z_inputvars))
    sweforcings = Matrix{dtype}(select(timeseries, SWE_inputvars))
    z_zidx = findfirst(z_inputvars .== :z)
    z_sweidx = findfirst(z_inputvars .== :SWE)
    swe_zidx = findfirst(SWE_inputvars .== :z)
    swe_sweidx = findfirst(SWE_inputvars .== :SWE)
    check_dates =
        (timeseries[2:end, timevar] - timeseries[1:(end - 1), timevar]) ./ dt
    pred_dzdts = zeros(nrow(timeseries) - 1)
    pred_dswedts = zeros(nrow(timeseries) - 1)
    pred_zs = zeros(nrow(timeseries))
    pred_swes = zeros(nrow(timeseries))
    pred_zs[1] = timeseries[1, :z]
    pred_swes[1] = timeseries[1, :SWE]
    countresets = 0
    for j in 2:length(pred_zs)
        zinput = zforcings[j - 1, :]
        sweinput = sweforcings[j - 1, :]
        zinput[z_zidx] = pred_zs[j - 1]
        zinput[z_sweidx] = pred_swes[j - 1]
        sweinput[swe_zidx] = pred_zs[j - 1]
        sweinput[swe_sweidx] = pred_swes[j - 1]
        dzdtpred = evaluate(zmodel, zinput)[1]
        dswedtpred = evaluate(swemodel, sweinput)[1]
        pred_dzdts[j - 1] = dzdtpred
        pred_dswedts[j - 1] = dswedtpred
        nperiods = check_dates[j - 1]
        new_z = pred_zs[j - 1] + nperiods * Dates.value(Second(dt)) * dzdtpred
        new_swe =
            pred_swes[j - 1] + nperiods * Dates.value(Second(dt)) * dswedtpred
        pred_swes[j] =
            (nperiods <= hole_thresh) ? max(0.0, new_swe) : timeseries[j, :SWE]
        pred_zs[j] =
            (nperiods <= hole_thresh) ? max(pred_swes[j], new_z) :
            timeseries[j, :z]
        if nperiods > hole_thresh
            countresets += 1
        end
    end
    return pred_zs, pred_swes, pred_dzdts, pred_dswedts, countresets
end


"""
    custom_loss(x, y, model, n1, n2)

Creates a loss function to be used during training specified by two hyperparameters n1, n2, as outlined in the paper.

# Arguments
- `x`: the input values.
- `y`: output values to compare to.
- `model`: the model over which to evaluate the loss function.
- `n1::Int`: the hyperparameter dictating the scaling of mismatch error.
- `n2::Int`: the hyperparameter dictating the weighting of a given mismatch error by the target magnitude.
"""
function custom_loss(x, y, model, n1, n2)
    sum(abs.((model(x) .- y)) .^ n1 .* (1 .+ abs.(y) .^ n2)) ./ length(y)
end

"""
    trainmodel!(model, ps, x_train, y_train, n1, n2; nepochs, opt, verbose, cb)

A training function for a neural model, permitting usage of a callback function.

# Arguments
- `model`: the model used for training.
- `ps`: the model parameters that will be trained.
- `x_train`: the input training data to be used in training.
- `y_train`: the target data to be used in training.
- `n1`: the scaling hypermarameter used to generate custom loss functions.
- `n2`: the weighting hyperparameter used to generate custom loss functions.
- `nepochs::Int`: the number of epochs. Default is 100.
- `nbatch::Int`: The number of data points to be used per batch. Default is 64.
- `opt`: the Flux optimizer to be used. Default is `RMSProp()`.
- `verbose::Bool`: indicates whether to print the training loss every 10 epochs
- `cb`: Allows utlization of a callback function (must take no required
input arguments, but default optional args are permitted). Default is Nothing.
"""
function trainmodel!(
    model,
    ps,
    x_train,
    y_train,
    n1,
    n2;
    nepochs::Int = 100,
    nbatch = 64,
    opt = Flux.Optimise.RMSProp(),
    verbose = false,
    cb = Nothing,
)
    train_loader = Flux.DataLoader(
        (x_train, y_train),
        batchsize = nbatch,
        partial = false,
        shuffle = true,
    )
    loss(x, y) = custom_loss(x, y, model, n1, n2)
    for epoch in 1:nepochs
        for (x, y) in train_loader
            Flux.train!(loss, ps, [(x, y)], opt)
        end
        if verbose & (epoch % 10 == 0)
            print(
                "Epoch: ",
                epoch,
                " | training loss: ",
                loss(x_train, y_train),
                "\n",
            )
        end
        cb()
    end
end
end
