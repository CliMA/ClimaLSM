import SciMLBase
import ClimaTimeSteppers as CTS
using CairoMakie
import ClimaParams as CP
import ClimaUtilities.TimeVaryingInputs: TimeVaryingInput
import ClimaUtilities.OutputPathGenerator: generate_output_path
import ClimaUtilities.ClimaArtifacts: @clima_artifact
using ClimaCore
using ClimaLand.Snow
using ClimaLand.Domains
using ClimaComms
import ClimaLand
import ClimaLand.Parameters as LP
using DataFrames
using NCDatasets
using Dates
using ClimaLand: PrescribedAtmosphere, PrescribedRadiativeFluxes
using Thermodynamics
using Statistics
using Insolation
using DelimitedFiles

using CSV, HTTP, Flux, cuDNN, BSON, StatsBase
NeuralSnow = Base.get_extension(ClimaLand, :NeuralSnowExt).NeuralSnow;

# Site-specific quantities
# Error if no site argument is provided
#if length(ARGS) < 1
#    @error("Please provide a site name as command line argument")
#else
#    SITE_NAME = ARGS[1]
#end
SITE_NAME = "cdp"
climaland_dir = pkgdir(ClimaLand)

FT = Float32
param_set = LP.LandParameters(FT)
context = ClimaComms.context()
ClimaComms.init(context)

# This reads in the data and sets up the drivers, as well as computes the IC from the data
include(
    joinpath(climaland_dir, "experiments/standalone/Snow/process_snowmip.jl"),
)
savedir = generate_output_path("experiments/standalone/Snow/")
t0 = FT(0.0)
tf = FT(seconds[end])
ndays = (tf - t0) / 3600 / 24
Δt = FT(60 * 60)

domain = ClimaLand.Domains.Point(; z_sfc = FT(0))

#density_model = NeuralSnow.NeuralDepthModel(FT)
density_model = Snow.ConstantDensityModel(ρ)
depths = z[snow_data_avail]

parameters = SnowParameters{FT}(
    Δt;
    Ksat = FT(1e-4),
    α_snow = α,
    density = density_model,
    earth_param_set = param_set,
)
model = ClimaLand.Snow.SnowModel(
    parameters = parameters,
    domain = domain,
    boundary_conditions = ClimaLand.Snow.AtmosDrivenSnowBC(atmos, radiation),
)
Y, p, coords = ClimaLand.initialize(model)

# Set initial conditions
Y.snow.S .= FT(SWE[1]) # first data point
Y.snow.Sl .= 0 # first data point
#Y.snow.Z .= FT(depths[1]) #uncomment if using NeuralDepthModel instead of ConstantDensityModel
Y.snow.U .=
    ClimaLand.Snow.energy_from_q_l_and_swe(FT(SWE[1]), FT(0), parameters) # with q_l = 0

set_initial_cache! = ClimaLand.make_set_initial_cache(model)
set_initial_cache!(p, Y, t0)
exp_tendency! = ClimaLand.make_exp_tendency(model)
# We use ARS111 (equivalent to explicit Euler) here for ease of assessing conservation
timestepper = CTS.ARS111()
ode_algo = CTS.IMEXAlgorithm(
    timestepper,
    CTS.NewtonsMethod(
        max_iters = 1,
        update_j = CTS.UpdateEvery(CTS.NewTimeStep),
    ),
);


prob = SciMLBase.ODEProblem(
    CTS.ClimaODEFunction(
        T_exp! = exp_tendency!,
        T_imp! = nothing,
        dss! = ClimaLand.dss!,
    ),
    Y,
    (t0, tf),
    p,
)
saveat = FT.(collect(t0:Δt:tf))
sv = (;
    t = Array{FT}(undef, length(saveat)),
    saveval = Array{NamedTuple}(undef, length(saveat)),
);
saving_cb = ClimaLand.NonInterpSavingCallback(sv, saveat);
updateat = copy(saveat)
drivers = ClimaLand.get_drivers(model)
updatefunc = ClimaLand.make_update_drivers(drivers)
driver_cb = ClimaLand.DriverUpdateCallback(updateat, updatefunc)
cb = SciMLBase.CallbackSet(driver_cb, saving_cb)

sol = SciMLBase.solve(
    prob,
    ode_algo;
    dt = Δt,
    adaptive = false,
    saveat = saveat,
    callback = cb,
);

# Plotting
q_l = [parent(sv.saveval[k].snow.q_l)[1] for k in 1:length(sol.t)];
T = [parent(sv.saveval[k].snow.T)[1] for k in 1:length(sol.t)];
evaporation = [
    parent(sv.saveval[k].snow.turbulent_fluxes.vapor_flux)[1] for
    k in 1:length(sol.t)
];
R_n = [parent(sv.saveval[k].snow.R_n)[1] for k in 1:length(sol.t)];
water_runoff =
    [parent(sv.saveval[k].snow.water_runoff)[1] for k in 1:length(sol.t)];
rain = [parent(sv.saveval[k].drivers.P_liq)[1] for k in 1:length(sv.t)];
snow = [parent(sv.saveval[k].drivers.P_snow)[1] for k in 1:length(sv.t)];
scf =
    [parent(sv.saveval[k].snow.snow_cover_fraction)[1] for k in 1:length(sol.t)];

S = [parent(sol.u[k].snow.S)[1] for k in 1:length(sol.t)];
U = [parent(sol.u[k].snow.U)[1] for k in 1:length(sol.t)];
t = sol.t;

start_day = 1
days = start_day .+ floor.(t ./ 3600 ./ 24)
doys = days .% 365 # doesn't account for leap year

obs_swes = Vector{Union{Float64, Missing}}(missing, length(doys))
obs_swes[snow_data_avail] .= mass[snow_data_avail] ./ 1000

obs_tsnows = Vector{Union{Float64, Missing}}(missing, length(doys))
obs_tsnows[snow_data_avail] = T_snow .+ 273.15

obs_df = DataFrame(
    doy = doys,
    model_swe = S,
    obs_swe = obs_swes,
    model_tsnow = T,
    obs_tsnow = obs_tsnows,
)
function missingmean(x)
    return mean(skipmissing(x))
end

mean_obs_df = combine(
    groupby(obs_df, :doy),
    [:model_swe, :obs_swe, :model_tsnow, :obs_tsnow] .=> missingmean,
    renamecols = false,
)
daily = t ./ 24 ./ 3600

fig = CairoMakie.Figure(size = (1600, 1200), fontsize = 26)
# set limits
ax1 = CairoMakie.Axis(fig[1, 1], ylabel = "Water Vol/Area ground")

xlims!(ax1, 0, ndays)
CairoMakie.hidexdecorations!(ax1, ticks = false)
CairoMakie.lines!(ax1, daily, S, label = "Model")
CairoMakie.scatter!(
    ax1,
    seconds[snow_data_avail] ./ 24 ./ 3600,
    SWE,
    label = "Data",
    color = :red,
)
CairoMakie.axislegend(ax1, position = :rt)

ax2 = CairoMakie.Axis(
    fig[2, 1],
    ylabel = "Liquid Mass Fraction",
    xlabel = "Time of year (days)",
)
CairoMakie.lines!(ax2, daily, q_l, label = "")
xlims!(ax2, 0, ndays)
ax3 = CairoMakie.Axis(
    fig[3, 1],
    ylabel = "Average Snow Water",
    xlabel = "Day of Year",
)
CairoMakie.lines!(ax3, mean_obs_df.doy, mean_obs_df.model_swe, label = "Model")
CairoMakie.scatter!(
    ax3,
    mean_obs_df.doy,
    mean_obs_df.obs_swe,
    label = "Data",
    color = :red,
)
CairoMakie.save(joinpath(savedir, "snow_water_content_$(SITE_NAME).png"), fig)




fig = CairoMakie.Figure(size = (1600, 1200), fontsize = 26)
# set limits
ax1 = CairoMakie.Axis(fig[1, 1], ylabel = "Temperature")

xlims!(ax1, 0, ndays)
CairoMakie.hidexdecorations!(ax1, ticks = false)
CairoMakie.lines!(ax1, daily, T, label = "Model")
CairoMakie.scatter!(
    ax1,
    seconds[snow_data_avail] ./ 24 ./ 3600,
    T_snow .+ 273.15,
    label = "Snow T",
    color = :red,
)
CairoMakie.scatter!(
    ax1,
    seconds[snow_data_avail] ./ 24 ./ 3600,
    Tair[snow_data_avail],
    label = "Atmosphere T",
    color = :orange,
)
CairoMakie.axislegend(ax1, position = :rt)

ax2 = CairoMakie.Axis(
    fig[2, 1],
    ylabel = "Snow Energy/Ground Area",
    xlabel = "Time of year (days)",
)
CairoMakie.lines!(ax2, daily, U, label = "")
xlims!(ax2, 0, ndays)
ax3 = CairoMakie.Axis(
    fig[3, 1],
    ylabel = "Average Snow Temperature",
    xlabel = "Day of Year",
)
CairoMakie.lines!(
    ax3,
    mean_obs_df.doy,
    mean_obs_df.model_tsnow,
    label = "Model",
)
CairoMakie.scatter!(
    ax3,
    mean_obs_df.doy,
    mean_obs_df.obs_tsnow,
    label = "Data",
    color = :red,
)
CairoMakie.save(joinpath(savedir, "snow_energy_content_$(SITE_NAME).png"), fig)



fig = CairoMakie.Figure(size = (1000, 1000), fontsize = 26)
# set limits
ax1 = CairoMakie.Axis(
    fig[1, 1],
    ylabel = "Cumulative height (m)",
    xlabel = "Time(days)",
)
xlims!(ax1, 0, ndays)
CairoMakie.lines!(ax1, daily, cumsum(snow) .* Δt, label = "Snow", color = :red)
CairoMakie.lines!(
    ax1,
    daily,
    cumsum(rain) .* Δt,
    label = "Rain",
    color = :green,
)
CairoMakie.lines!(
    ax1,
    daily,
    cumsum(evaporation .* scf) .* Δt,
    label = "Evaporation",
    color = :purple,
)
CairoMakie.lines!(
    ax1,
    daily,
    cumsum(water_runoff .* scf) .* Δt,
    label = "Runoff/Melt",
    color = :blue,
)
CairoMakie.axislegend(ax1, position = :lb)

CairoMakie.save(joinpath(savedir, "water_fluxes_$(SITE_NAME).png"), fig)

# Assess conservation
fig = CairoMakie.Figure(size = (1600, 1200), fontsize = 26)
ax1 = CairoMakie.Axis(fig[2, 1], ylabel = "ΔEnergy (J/A)", xlabel = "Days")
ΔE_expected =
    cumsum(
        -1 .* [
            parent(sv.saveval[k].snow.applied_energy_flux)[end] for
            k in 2:1:length(sv.t)
        ],
    ) * (sv.t[2] - sv.t[1])
E_measured = [parent(sol.u[k].snow.U)[end] for k in 1:1:length(sv.t)]
ΔW_expected =
    cumsum(
        -1 .* [
            parent(sv.saveval[k].snow.applied_water_flux)[end] for
            k in 2:1:length(sv.t)
        ],
    ) * (sv.t[2] - sv.t[1])
W_measured = [parent(sol.u[k].snow.S)[end] for k in 1:1:length(sv.t)]
CairoMakie.lines!(
    ax1,
    daily[2:end],
    E_measured[2:end] .- E_measured[1],
    label = "Simulated",
)
CairoMakie.lines!(ax1, daily[2:end], ΔE_expected, label = "Expected")
CairoMakie.axislegend(ax1, position = :rt)

# Temp
ax4 = CairoMakie.Axis(fig[1, 1], ylabel = "ΔWater (m)")
CairoMakie.hidexdecorations!(ax4, ticks = false)
CairoMakie.lines!(
    ax4,
    daily[2:end],
    W_measured[2:end] .- W_measured[1],
    label = "Simulated",
)

CairoMakie.lines!(ax4, daily[2:end], ΔW_expected, label = "Expected")
CairoMakie.axislegend(ax4, position = :rt)


ax3 = CairoMakie.Axis(fig[2, 2], ylabel = "ΔE/E", xlabel = "Days")
CairoMakie.lines!(
    ax3,
    daily[2:end],
    (E_measured[2:end] .- E_measured[1] .- ΔE_expected) ./ mean(E_measured),
)

ax2 = CairoMakie.Axis(fig[1, 2], ylabel = "ΔW/W")
CairoMakie.hidexdecorations!(ax2, ticks = false)
CairoMakie.lines!(
    ax2,
    daily[2:end],
    (W_measured[2:end] .- W_measured[1] .- ΔW_expected) ./ mean(W_measured),
)

CairoMakie.save(joinpath(savedir, "conservation_$(SITE_NAME).png"), fig)
