# # Seasonal Snow Timeseries Generation with a Neural Network

# This tutorial explains how to make use of the code developed
# for forecasting seasonal snow depth evolution, using a neural network with
# structurally-enforced constraints. The following steps through a
# basic use-case of the system on an already-cleaned dataset, though
# exploration of optional keyword arguments in the developed code and
# additional tools for scraping data (explained in the [data tutorial](../data_tutorial/))
# provide for a richer set of functionality.

# The updates of the neural snow model follow the equation

# ``
# \frac{dz}{dt} = \mathcal{M}\left(z, SWE, φ, R, v, T_{air}, P_{snow}\right),
# ``

# where

# - ``t`` is the time (s),

# - ``z`` is the snow depth (m),

# - ``\mathcal{M}`` is the neural network,

# - ``SWE`` is the Snow Water Equivalent, or the height of water if all the snow melted (m),

# - ``φ`` is the relative humidity (0-1),

# - ``R`` is the solar radiation (W/m²).

# - ``v`` is the wind speed (W/m²).

# - ``T_{air}`` is the air temperature (degrees C).

# - ``P_{snow}`` is the water equivalent rate of snow precipitation (m/s).

# The model is a 1D model to permit utilization over any desired grid resolution and shape.

# We will use the forcings and snow depth data as a validation
# of the model, so the initial conditions will be the initial value
# provided in the existing data.

# We begin by importing the developed code to create and run the neural network,
# as well as some preliminary packages:
using ClimaLand
using DataFrames, CSV, HTTP, Dates, Flux, StatsBase, cuDNN, BSON

# The code lives in an extenson that we have to manually load. The extension can
# be loaded only if "DataFrames", "CSV", "HTTP", "Flux", "StatsBase", "cuDNN", "BSON", and "ClimaLand"
# are loaded.
DataTools = Base.get_extension(ClimaLand, :NeuralSnowExt).DataTools
ModelTools = Base.get_extension(ClimaLand, :NeuralSnowExt).ModelTools;

# and also, for this tutorial, some purpose-made functions for displaying the output.
# A similar `analysis_tools.jl` file exists alongside `display_tools.jl` for
# some basic functions for analyzing/scoring the model, if desired.
using ClimaLand
code_dir = joinpath(pkgdir(ClimaLand), "docs/tutorials/standalone/Snow")
include(joinpath(code_dir, "display_tools.jl"));

# Next, we set up values of the network hyperparameters, including the
# width parameter ``n`` as outlined
# in the associated [paper](https://arxiv.org/abs/2412.06819), and the two loss function hyperparameters ``n_1``, ``n_2``.
n = 4
n1 = 2
n2 = 4;

# We next outline which variables in the dataset will be used as predictors,
# calling them by their column name as a `Symbol`. The number and choice of these
# can be changed to reflect any dataset. Another column is specified as the target
# variable, in this case, the ``\frac{dz}{dt}`` column.
pred_vars = [
    :z,
    :SWE,
    :rel_hum_avg,
    :sol_rad_avg,
    :wind_speed_avg,
    :air_temp_avg,
    :dprecipdt_snow,
]
target = :dzdt;

# Specifying the indices of the depth and precipitation variables
# (used in the constraints) and the total number of input features
# will be necessary when creating the model, so we will specify them
# here as well.
nfeatures = length(pred_vars)
z_idx = 1
p_idx = 7;

# We next read in the already-cleaned [training](https://caltech.box.com/v/neuralsnow-training-data)
# and [testing](https://caltech.box.com/v/neuralsnow-testing-data) datasets, though for custom datasets
# there is plenty of functionality provided in the `DataTools` module
# to scrape SNOTEL data directly. We also set the
# unit timestep seen in this data (daily, so 1 day) to be used for
# setting the network's constraints as well as generating timeseries during usage.
# To see the code that generated this data file, check out the [data tutorial](../data_tutorial/).
# We also specify the maximum gap size in the data (in units of Δt) that the network can traverse
# before requiring a reset, via `hole_thresh`.
training_data_download_link =
    ClimaLand.Artifacts.neural_snow_training_data_link()
testing_data_download_link = ClimaLand.Artifacts.neural_snow_testing_data_link()
data_train = CSV.read(HTTP.get(training_data_download_link).body, DataFrame)
valdata = CSV.read(HTTP.get(testing_data_download_link).body, DataFrame)
Δt = Second(86400)
hole_thresh = 5;

# With this, we can begin the actual usage pipeline. First, we split the
# precipitation feature into rain and snow constituents, and apply a set of
# filters before extracting the necessary features with `prep_data` (the split already exists in the testing data):
usedata = DataTools.prep_data(data_train);

# After this, we determine scalings for the input and target data
# that are conducive to beneficial weight updates. In this case, the
# target data during training will be scaled in the -1 to 1 range, and
# the neural network will scale input features according to their
# standard deviations (no shifting is carried out in this case, so that
# the physical meaning of "0" is preserved). This data
# is then converted into matrix form for ease of its conversion into
# a Flux `DataLoader` object, later, during training.
out_scale = maximum(abs.(usedata[!, target]))
in_scales = std.(eachcol(select(usedata, pred_vars)))
x_train, y_train = DataTools.make_data(usedata, pred_vars, target, out_scale);

# We then create the model itself: we can start by specifying upper and lower bounding
# functions that enhance model stability and generalizability. Boundary
# functions only take two inputs, `pred` and `input`, with the following considerations:
# - `input` will be an `AbstractArray{<:AbstractFloat}` type (usually a `Matrix`), of size ``N\times K``, where ``N`` is the number of input features and ``K`` is the number of samples provided
# - `pred` will also be an `AbstractArray{<:AbstractFloat}` type of size ``1 \times K``, where ``K`` is the number of predictions made from the provided inputs/samples
# - the boundary functions must return a row-vector of size ``1\times K``, equal to the values of the boundary values for each input sample, or something that is readily broadcasted to this size (like a single scalar)
# - for each given input, the upper boundary value should be greater than or equal to the lower boundary value
# - anything beyond `pred` or `input` used in the function should be accessibly defined in the scope of the utilizing code
# With this in mind, we pick an upper boundary value that leaves the prediction unchanged
# if snowfall is present, but clamps the prediction to be nonpositive if no snowfall is present.
# We pick a lower boundary boundary defining ``\frac{dz}{dt} \leq -z/Δt`` to prevent the
# snowpack from ever becoming a negative value:
upper_bound(pred, input) = @. (input[p_idx, :]' > 0) * relu(pred)
lower_bound(pred, input) = -input[z_idx, :]' / Dates.value(Δt);

# We then specify `Float32` as the `Float` type for the model (it will run faster
# than `Float64`, and changing the model type is as simple as calling `convert_model!(model, T)`
# for Float type `T`), and make the model using our boundary functions:
FT = Float32
model = ModelTools.make_model(
    nfeatures,
    n,
    upper_bound,
    lower_bound,
    FT,
    in_scale = in_scales,
);

# The above example shows how to build modles for a more general setup depending on your features, data, and model
# needs, but for the same model
# with the same boundary types as given in the paper, a predefined method `make_model_paper()` also exists
# to instead get the same model used in the [paper](https://arxiv.org/abs/2412.06819), with additional speed optimizations and
# the right hyperparameters and scalings already set. For this method, one only has to
# indicate which input features are to be used to determine the boundary constraints on the network
# (the values in this tutorial are the default values of the function, but one could pass alternative
# values if working with different data or building custom models)
model = ModelTools.make_model_paper(
    depth_index = z_idx,
    precipitation_index = p_idx,
);

# As training updates are better with the scaled data, we have to modify
# the timescale and output scaling of the model structure prior to training.
# This step is undone/reset after training is over. Note that the `settimescale!()`
# function only works for models made with `make_model_paper()` and this would have
# to be done manually otherwise. `setoutscale!()` will work for models made
# with either of `make_model()` or `make_model_paper()`:
ModelTools.settimescale!(model, Dates.value(Δt) * out_scale)
ModelTools.setoutscale!(model, 1.0);

# For models made with `make_model()` or `make_model_paper()`, training
# is as simple as calling the `trainmodel!` function, which makes use of 
# our loss-function hyperparameters:
print("\nTraining model!\n")
ModelTools.trainmodel!(model, x_train, y_train, n1, n2, verbose = true);

# To show the model's output on some of our training data in physically meaningful
# units, we first reset the timesacle and output scaling constants. From there,
# all we do is pass the dataframe for a given SNOTEL site and the trained model
# to the `make_timeseries` function, and we can compare the result to the actual data.
ModelTools.setoutscale!(model, out_scale)
ModelTools.settimescale!(model, Dates.value(Δt));

# For instance, let's show the results on SNOTEL site 1286 (Slagamount Lakes site, Montana):

# *Note that gaps in the data are shown as shaded regions on the plotted timeseries*.
site_id = 1286
sitedata = usedata[usedata[!, :id] .== site_id, :]
true_series = sitedata[!, :z]
pred_series, _, _ =
    ModelTools.make_timeseries(model, sitedata, Δt, hole_thresh = hole_thresh)
ptitle = "Slagamount Lakes, Snow Depth (m)"
siteplot(
    ptitle,
    sitedata[!, :date],
    [true_series, pred_series],
    ["Data", "Neural Model"],
    [:black, :red],
    savename = "base_tutorial_plot1.png",
    display_plot = false,
);
# ![](base_tutorial_plot1.png)

# Or, alternatively, SNOTEL site 1070 (Anchorage Hillside, Alaska) from the testing data:
site_id = "1070" #string format for the testing ids is due to non-numerical testing site codes.
sitedata = valdata[valdata[!, :id] .== site_id, :]
true_series = sitedata[!, :z]
pred_series, _, _ =
    ModelTools.make_timeseries(model, sitedata, Δt, hole_thresh = hole_thresh)
ptitle = "Anchorage Hillside, Snow Depth (m)"
siteplot(
    ptitle,
    sitedata[!, :date],
    [true_series, pred_series],
    ["Data", "Neural Model"],
    [:black, :red],
    savename = "base_tutorial_plot2.png",
    display_plot = false,
);
# ![](base_tutorial_plot2.png)

# The `save_predictive_model_weights()` function can be used to write the
# weights of a trained model to a text file. Weights written to such files
# can be loaded into a network by first building the model structure with either
# `make_model()` or `make_model_paper()`, and then loading the weights using the 
# `load_model_weights!()` function, which takes a filepath or a hyperlink. For example, if you wanted to use the exact
# model weights used for the [``z`` network](https://caltech.box.com/v/paper-model-z) or the
# [``SWE`` network](https://caltech.box.com/v/paper-model-swe) in the paper, you can use the following:
z_model_link = ClimaLand.Artifacts.neural_snow_znetwork_link()
swe_model_link = ClimaLand.Artifacts.neural_snow_swenetwork_link()
ModelTools.load_model_weights!(z_model_link, model);
# However, note the SWE model has a different structure with `n=5`, and must
# be built correctly with the correct arguments in `make_model_paper()` in order to 
# correctly load the right weights.

# Additional functionality can be explored through the [optional arguments](https://github.com/CliMA/ClimaLand.jl/blob/main/ext/neural_snow/ModelTools.jl)
# to the developed functions, though creating timeseries for any validation
# dataset can be handled with a similar call to `make_timeseries` (or 
# `paired_timeseries` for a ``z`` and ``SWE`` network). The timestep
# `Δt` (as well as a matching call to the network with `settimescale!`) can
# also be changed to different values to evaluate the network's capability on
# validation data with different temporal resolutions, without the need
# for retraining.
