import SciMLBase
import ClimaTimeSteppers as CTS
import ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
using ClimaCore
using CairoMakie
using Statistics
using Dates
using NCDatasets

using ClimaLand
using ClimaLand.Domains: Column
using ClimaLand.Soil
using ClimaLand.Snow
import ClimaLand
import ClimaLand.Parameters as LP
import ClimaParams

climaland_dir = pkgdir(ClimaLand)

FT = Float64
earth_param_set = LP.LandParameters(FT)
# Utility functions for reading in and filling fluxnet data
include(joinpath(climaland_dir, "experiments/integrated/fluxnet/data_tools.jl"))

# Column dimensions - separation of layers at the top and bottom of the column:
nelements = 10
zmin = FT(-2)
zmax = FT(0)
dz_bottom = FT(1.0)
dz_top = FT(0.04)

land_domain = Column(;
    zlim = (zmin, zmax),
    nelements = nelements,
    dz_tuple = (dz_bottom, dz_top),
)

# TIME STEPS:
t0 = FT(1800)
N_days_spinup = 0
N_days = N_days_spinup + 360
dt = FT(900)
tf = t0 + FT(3600 * 24 * N_days)

# Read in the parameters for the site
include(
    joinpath(
        climaland_dir,
        "experiments/integrated/fluxnet/snow_soil/parameters_fluxnet.jl",
    ),
)
# This reads in the data from the flux tower site and creates
# the atmospheric and radiative driver structs for the model
include(
    joinpath(
        climaland_dir,
        "experiments/integrated/fluxnet/met_drivers_FLUXNET.jl",
    ),
)


# Now we set up the model. For the soil model, we pick
# a model type and model args:
soil_domain = land_domain
soil_ps = Soil.EnergyHydrologyParameters(
    FT;
    ν = soil_ν,
    ν_ss_om,
    ν_ss_quartz,
    ν_ss_gravel,
    hydrology_cm = vanGenuchten{FT}(; α = soil_vg_α, n = soil_vg_n),
    K_sat = soil_K_sat,
    S_s = soil_S_s,
    θ_r,
    z_0m = z_0m_soil,
    z_0b = z_0b_soil,
    emissivity = soil_ϵ,
    PAR_albedo = soil_α_PAR,
    NIR_albedo = soil_α_NIR,
)

soil_args = (parameters = soil_ps,)
soil_model_type = Soil.EnergyHydrology

ρ = 300.0
α = 0.8
snow_parameters = SnowParameters{FT}(
    dt;
    α_snow = α,
    density = Snow.ConstantDensityModel(ρ),
    earth_param_set = earth_param_set,
);
snow_args = (; parameters = snow_parameters);
snow_model_type = Snow.SnowModel
land_input = (
    atmos = atmos,
    radiation = radiation,
    domain = land_domain,
    runoff = ClimaLand.Soil.SurfaceRunoff(),
)
land = ClimaLand.LandHydrologyModel{FT}(;
    land_args = land_input,
    soil_model_type = soil_model_type,
    soil_args = soil_args,
    snow_args = snow_args,
    snow_model_type = snow_model_type,
)
Y, p, cds = initialize(land)
exp_tendency! = make_exp_tendency(land)
imp_tendency! = make_imp_tendency(land)
jacobian! = ClimaLand.make_jacobian(land);
jac_kwargs = (; jac_prototype = ImplicitEquationJacobian(Y), Wfact = jacobian!);
set_initial_cache! = make_set_initial_cache(land)


#Initial conditions
# Find data index corresponds to t0
t0_idx = 1 + Int(round(t0 / DATA_DT))
Y.soil.ϑ_l =
    drivers.SWC.status != absent ? drivers.SWC.values[t0_idx] : soil_ν / 2
Y.soil.θ_i = 0
T_0 =
    drivers.TS.status != absent ? drivers.TS.values[t0_idx] :
    drivers.TA.values[t0_idx]
ρc_s =
    volumetric_heat_capacity.(
        Y.soil.ϑ_l,
        Y.soil.θ_i,
        soil_ps.ρc_ds,
        earth_param_set,
    )
Y.soil.ρe_int =
    volumetric_internal_energy.(Y.soil.θ_i, ρc_s, T_0, earth_param_set)

Y.snow.S .= 0.0
Y.snow.S_l .= 0.0
Y.snow.U .= 0.0
set_initial_cache!(p, Y, t0)

saveat = Array(t0:dt:tf)
sv = (;
    t = Array{Float64}(undef, length(saveat)),
    saveval = Array{NamedTuple}(undef, length(saveat)),
)
saving_cb = ClimaLand.NonInterpSavingCallback(sv, saveat)

updateat = deepcopy(saveat)
model_drivers = ClimaLand.get_drivers(land)
updatefunc = ClimaLand.make_update_drivers(model_drivers)
driver_cb = ClimaLand.DriverUpdateCallback(updateat, updatefunc)
cb = SciMLBase.CallbackSet(driver_cb, saving_cb)
# TIME STEPPING
timestepper = CTS.ARS111();
ode_algo = CTS.IMEXAlgorithm(
    timestepper,
    CTS.NewtonsMethod(
        max_iters = 3,
        update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
    ),
);

# Problem definition and callbacks
prob = SciMLBase.ODEProblem(
    CTS.ClimaODEFunction(
        T_exp! = exp_tendency!,
        T_imp! = SciMLBase.ODEFunction(imp_tendency!; jac_kwargs...),
        dss! = ClimaLand.dss!,
    ),
    Y,
    (t0, tf),
    p,
);
sol = SciMLBase.solve(prob, ode_algo; callback = cb, dt = dt, saveat = t0:dt:tf);

# Plotting
daily = sol.t ./ 3600 ./ 24
savedir = joinpath(climaland_dir, "experiments/integrated/fluxnet/snow_soil")


# Water content

fig = Figure(size = (1600, 1200), fontsize = 26)
ax1 = Axis(fig[2, 2], ylabel = "SWC", xlabel = "Days")
lines!(
    ax1,
    daily,
    [parent(sol.u[k].soil.ϑ_l)[end - 2] for k in 1:1:length(sol.t)],
    label = "10cm",
)
lines!(
    ax1,
    daily,
    [
        parent(sol.u[k].soil.θ_i .+ sol.u[k].soil.ϑ_l)[end - 2] for
        k in 1:1:length(sol.t)
    ],
    label = "10cm, liq+ice",
)

lines!(
    ax1,
    seconds ./ 3600 ./ 24,
    drivers.SWC.values[:],
    label = "Data, Unknown Depth",
)
axislegend(ax1, position = :rt)

ax2 = Axis(fig[1, 2], ylabel = "Precipitation (mm/day)")
ylims!(ax2, [-1300, 0])
hidexdecorations!(ax2, ticks = false)
lines!(ax2, seconds ./ 3600 ./ 24, P_liq .* (1e3 * 24 * 3600), label = "Liquid")
lines!(ax2, seconds ./ 3600 ./ 24, P_snow .* (1e3 * 24 * 3600), label = "Snow")
axislegend(ax2, position = :rb)
ax3 = Axis(fig[2, 1], ylabel = "SWE (m)", xlabel = "Days")
lines!(ax3, daily, [parent(sol.u[k].snow.S)[1] for k in 1:1:length(sol.t)])

# Temp
ax4 = Axis(fig[1, 1], ylabel = "T (K)")
hidexdecorations!(ax4, ticks = false)
lines!(ax4, seconds ./ 3600 ./ 24, drivers.TA.values[:], label = "Data, Air")
lines!(
    ax4,
    sv.t ./ 24 ./ 3600,
    [parent(sv.saveval[k].soil.T)[end - 2] for k in 1:1:length(sv.t)],
    label = "Model 10 cm",
)

lines!(
    ax4,
    sv.t ./ 24 ./ 3600,
    [parent(sv.saveval[k].snow.T)[1] for k in 1:1:length(sv.t)],
    label = "Snow",
)
lines!(
    ax4,
    seconds ./ 3600 ./ 24,
    drivers.TS.values[:],
    label = "Data, Unknown depth",
)
axislegend(ax4, position = :rt)
CairoMakie.save(joinpath(savedir, "results.png"), fig)

# Assess conservation
_ρ_i = FT(LP.ρ_cloud_ice(earth_param_set))
_ρ_l = FT(LP.ρ_cloud_liq(earth_param_set))
fig = Figure(size = (1600, 1200), fontsize = 26)
ax1 = Axis(fig[2, 1], ylabel = "ΔEnergy (J/A)", xlabel = "Days")
ΔE_expected =
    cumsum(
        -1 .* [
            parent(
                sv.saveval[k].atmos_energy_flux .-
                sv.saveval[k].soil.bottom_bc.heat,
            )[end] for k in 1:1:(length(sv.t) - 1)
        ],
    ) * (sv.t[2] - sv.t[1])
E_measured = [
    sum(sol.u[k].soil.ρe_int) + parent(sol.u[k].snow.U)[end] for
    k in 1:1:length(sv.t)
]
ΔW_expected =
    cumsum(
        -1 .* [
            parent(
                sv.saveval[k].atmos_water_flux .-
                sv.saveval[k].soil.bottom_bc.water .+ sv.saveval[k].soil.R_s,
            )[end] for k in 1:1:(length(sv.t) - 1)
        ],
    ) * (sv.t[2] - sv.t[1])
W_measured = [
    sum(sol.u[k].soil.ϑ_l) +
    sum(sol.u[k].soil.θ_i) * _ρ_i / _ρ_l +
    parent(sol.u[k].snow.S)[end] for k in 1:1:length(sv.t)
]
lines!(
    ax1,
    daily[2:end],
    E_measured[2:end] .- E_measured[1],
    label = "Simulated",
)
lines!(ax1, daily[2:end], ΔE_expected, label = "Expected")
axislegend(ax1, position = :rt)

# Temp
ax4 = Axis(fig[1, 1], ylabel = "ΔWater (m)")
hidexdecorations!(ax4, ticks = false)
lines!(
    ax4,
    daily[2:end],
    W_measured[2:end] .- W_measured[1],
    label = "Simulated",
)

lines!(ax4, daily[2:end], ΔW_expected, label = "Expected")
axislegend(ax4, position = :rt)


ax3 = Axis(fig[2, 2], ylabel = "ΔE/E", xlabel = "Days", yscale = log10)
lines!(
    ax3,
    daily[2:end],
    abs.(E_measured[2:end] .- E_measured[1] .- ΔE_expected) ./ mean(E_measured),
)

ax2 = Axis(fig[1, 2], ylabel = "ΔW/W", yscale = log10)
hidexdecorations!(ax2, ticks = false)
lines!(
    ax2,
    daily[2:end],
    abs.(W_measured[2:end] .- W_measured[1] .- ΔW_expected) ./ mean(W_measured),
)

CairoMakie.save(joinpath(savedir, "results_conservation.png"), fig)
