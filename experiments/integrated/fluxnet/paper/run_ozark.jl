import SciMLBase
import ClimaTimeSteppers as CTS
import ClimaComms
ClimaComms.@import_required_backends
using ClimaCore
import ClimaParams as CP
using Statistics
using Dates
using Insolation
using StatsBase

using ClimaLand
using ClimaLand.Domains: Column
using ClimaLand.Snow
using ClimaLand.Soil
using ClimaLand.Soil.Biogeochemistry
using ClimaLand.Canopy
using ClimaLand.Canopy.PlantHydraulics
import ClimaLand
import ClimaLand.Parameters as LP
import ClimaUtilities.OutputPathGenerator: generate_output_path
using ClimaDiagnostics
using ClimaUtilities
const FT = Float64
earth_param_set = LP.LandParameters(FT)
climaland_dir = pkgdir(ClimaLand)

include(joinpath(climaland_dir, "experiments/integrated/fluxnet/data_tools.jl"))
include(joinpath(climaland_dir, "experiments/integrated/fluxnet/plot_utils.jl"))

site_ID = "US-MOz"

# Read all site-specific domain parameters from the simulation file for the site
include(
    joinpath(
        climaland_dir,
        "experiments/integrated/fluxnet/$site_ID/$(site_ID)_simulation.jl",
    ),
)

include(
    joinpath(climaland_dir, "experiments/integrated/fluxnet/fluxnet_domain.jl"),
)

# Read all site-specific parameters from the parameter file for the site
include(
    joinpath(
        climaland_dir,
        "experiments/integrated/fluxnet/$site_ID/$(site_ID)_parameters.jl",
    ),
)

# This reads in the data from the flux tower site and creates
# the atmospheric and radiative driver structs for the model
include(
    joinpath(
        climaland_dir,
        "experiments/integrated/fluxnet/fluxnet_simulation.jl",
    ),
)

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
);

soil_args = (domain = soil_domain, parameters = soil_ps)
soil_model_type = Soil.EnergyHydrology{FT}

# Soil microbes model
soilco2_type = Soil.Biogeochemistry.SoilCO2Model{FT}

soilco2_ps = SoilCO2ModelParameters(FT)

Csom = ClimaLand.PrescribedSoilOrganicCarbon{FT}(TimeVaryingInput((t) -> 5))

# Set the soil CO2 BC to being atmospheric CO2
soilco2_top_bc = Soil.Biogeochemistry.AtmosCO2StateBC()
soilco2_bot_bc = Soil.Biogeochemistry.SoilCO2FluxBC((p, t) -> 0.0) # no flux
soilco2_sources = (MicrobeProduction{FT}(),)

soilco2_boundary_conditions = (; top = soilco2_top_bc, bottom = soilco2_bot_bc)

soilco2_args = (;
    boundary_conditions = soilco2_boundary_conditions,
    sources = soilco2_sources,
    domain = soil_domain,
    parameters = soilco2_ps,
)

# Now we set up the canopy model, which we set up by component:
# Component Types
canopy_component_types = (;
    autotrophic_respiration = Canopy.AutotrophicRespirationModel{FT},
    radiative_transfer = Canopy.TwoStreamModel{FT},
    photosynthesis = Canopy.FarquharModel{FT},
    conductance = Canopy.MedlynConductanceModel{FT},
    hydraulics = Canopy.PlantHydraulicsModel{FT},
    energy = Canopy.BigLeafEnergyModel{FT},
)
# Individual Component arguments
# Set up autotrophic respiration
autotrophic_respiration_args =
    (; parameters = AutotrophicRespirationParameters(FT))
# Set up radiative transfer
radiative_transfer_args = (;
    parameters = TwoStreamParameters(
        FT;
        Ω,
        α_PAR_leaf,
        τ_PAR_leaf,
        α_NIR_leaf,
        τ_NIR_leaf,
        G_Function,
    )
)
# Set up conductance
conductance_args = (; parameters = MedlynConductanceParameters(FT; g1))
# Set up photosynthesis
is_c3 = FT(1) # set the photosynthesis mechanism to C3
photosynthesis_args = (;
    parameters = FarquharParameters(
        FT,
        is_c3;
        Vcmax25 = Vcmax25,
        pc = pc,
        sc = sc,
    )
)
# Set up plant hydraulics
ai_parameterization = PrescribedSiteAreaIndex{FT}(LAIfunction, SAI, RAI)

plant_hydraulics_ps = PlantHydraulics.PlantHydraulicsParameters(;
    ai_parameterization = ai_parameterization,
    ν = plant_ν,
    S_s = plant_S_s,
    rooting_depth = rooting_depth,
    conductivity_model = conductivity_model,
    retention_model = retention_model,
)
plant_hydraulics_args = (
    parameters = plant_hydraulics_ps,
    n_stem = n_stem,
    n_leaf = n_leaf,
    compartment_midpoints = compartment_midpoints,
    compartment_surfaces = compartment_surfaces,
)

energy_args = (parameters = Canopy.BigLeafEnergyParameters{FT}(ac_canopy),)

# Canopy component args
canopy_component_args = (;
    autotrophic_respiration = autotrophic_respiration_args,
    radiative_transfer = radiative_transfer_args,
    photosynthesis = photosynthesis_args,
    conductance = conductance_args,
    hydraulics = plant_hydraulics_args,
    energy = energy_args,
)

# Other info needed
shared_params = SharedCanopyParameters{FT, typeof(earth_param_set)}(
    z0_m,
    z0_b,
    earth_param_set,
)

canopy_model_args = (; parameters = shared_params, domain = canopy_domain)

# Snow model
snow_parameters =
    SnowParameters{FT}(dt; earth_param_set = earth_param_set, α_snow = FT(0.6));
snow_args = (; parameters = snow_parameters, domain = canopy_domain);
snow_model_type = Snow.SnowModel
# Integrated plant hydraulics and soil model
land_input = (
    atmos = atmos,
    radiation = radiation,
    soil_organic_carbon = Csom,
    runoff = ClimaLand.Soil.Runoff.SurfaceRunoff(),
)
land = LandModel{FT}(;
    soilco2_type = soilco2_type,
    soilco2_args = soilco2_args,
    land_args = land_input,
    soil_model_type = soil_model_type,
    soil_args = soil_args,
    canopy_component_types = canopy_component_types,
    canopy_component_args = canopy_component_args,
    canopy_model_args = canopy_model_args,
    snow_args = snow_args,
    snow_model_type = snow_model_type,
)

Y, p, cds = initialize(land)

#Initial conditions
Y.soil.ϑ_l =
    drivers.SWC.status != absent ?
    drivers.SWC.values[1 + Int(round(t0 / DATA_DT))] : soil_ν / 2 # Get soil water content at t0
# Both data and simulation are reference to 2005-01-01-00 (LOCAL)
# or 2005-01-01-06 (UTC)
Y.soil.θ_i = FT(0.0)
T_0 =
    drivers.TS.status != absent ?
    drivers.TS.values[1 + Int(round(t0 / DATA_DT))] :
    drivers.TA.values[1 + Int(round(t0 / DATA_DT))] + 40# Get soil temperature at t0
ρc_s =
    volumetric_heat_capacity.(
        Y.soil.ϑ_l,
        Y.soil.θ_i,
        land.soil.parameters.ρc_ds,
        earth_param_set,
    )
Y.soil.ρe_int =
    volumetric_internal_energy.(Y.soil.θ_i, ρc_s, T_0, earth_param_set)
Y.soilco2.C .= FT(0.000412) # set to atmospheric co2, mol co2 per mol air
ψ_stem_0 = FT(-1e5 / 9800) # pressure in the leaf divided by rho_liquid*gravitational acceleration [m]
ψ_leaf_0 = FT(-2e5 / 9800)
ψ_comps = n_stem > 0 ? [ψ_stem_0, ψ_leaf_0] : ψ_leaf_0

S_l_ini =
    inverse_water_retention_curve.(retention_model, ψ_comps, plant_ν, plant_S_s)

for i in 1:(n_stem + n_leaf)
    Y.canopy.hydraulics.ϑ_l.:($i) .=
        augmented_liquid_fraction.(plant_ν, S_l_ini[i])
end

Y.canopy.energy.T = drivers.TA.values[1 + Int(round(t0 / DATA_DT))] # Get atmos temperature at t0

Y.snow.S .= 0.0
Y.snow.U .= 0.0
Y.snow.S_l .= 0.0
set_initial_cache! = make_set_initial_cache(land)
set_initial_cache!(p, Y, t0);

exp_tendency! = make_exp_tendency(land)
imp_tendency! = make_imp_tendency(land)
jacobian! = make_jacobian(land);
    jac_kwargs = (;
        jac_prototype = ClimaLand.FieldMatrixWithSolver(Y),
        Wfact = jacobian!,
    )


# Callbacks
outdir =
    joinpath(pkgdir(ClimaLand), "experiments/integrated/fluxnet/($site_ID)")
output_dir = ClimaUtilities.OutputPathGenerator.generate_output_path(outdir)

## How often we want to update the drivers. Note that this uses the defined `t0` and `tf`
## defined in the simulatons file
updateat = Array(t0:DATA_DT:tf)
model_drivers = ClimaLand.get_drivers(land)
updatefunc = ClimaLand.make_update_drivers(model_drivers)
driver_cb = ClimaLand.DriverUpdateCallback(updateat, updatefunc)
saveat = Array(t_spinup:DATA_DT:tf)
sv = (;
    t = Array{FT}(undef, length(saveat)),
    saveval = Array{NamedTuple}(undef, length(saveat)),
);
saving_cb = ClimaLand.NonInterpSavingCallback(sv, saveat);
cb = SciMLBase.CallbackSet(driver_cb, saving_cb)


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

sol = SciMLBase.solve(prob, ode_algo; dt = dt, callback = cb, saveat = saveat);

# LWP
# Leaf water potential data from Pallardy et al (2018)
# Predawn Leaf Water Potential of Oak-Hickory Forest at Missouri Ozark (MOFLUX) Site: 2004-2020
# https://doi.org/10.3334/CDIAC/ORNLSFA.004
lwp_path = "/Users/katherinedeck/Downloads/MOFLUX_PredawnLeafWaterPotential_2020_20210125.csv"
lwp_data = readdlm(lwp_path, ',', skipstart = 1)
# We are using 2005 data in this test, so restrict to this year
YEAR = lwp_data[:, 1]
DOY = lwp_data[YEAR .== 2010, 2]
# Our t0 = Dec 31, midnight, 2004. Predawn = guess of 0600 hours
seconds_since_t0 = FT.(DOY) * 24 .* 3600 .+ (6 * 3600)
lwp_measured = lwp_data[YEAR .== 2010, 7] .* 1e6 ./ 1000 ./ 9.8# MPa to Pa and then to m


# Extract model output from the saved diagnostics
lwp = [parent(sv.saveval[k].canopy.hydraulics.ψ.:2)[1] for k in 1:length(sol.t)]
swe = [parent(sol.u[k].snow.S)[1] for k in 1:length(sol.t)];
scf =
    [parent(sv.saveval[k].snow.snow_cover_fraction)[1] for k in 1:length(sol.t)];
z_snow = [parent(sv.saveval[k].snow.z_snow)[1] for k in 1:length(sol.t)];
ϑl = [parent(sol.u[k].soil.ϑ_l)[end] for k in 1:length(sol.t)];
θ_i = [parent(sol.u[k].soil.θ_i)[end] for k in 1:length(sol.t)];
T_soil = [parent(sv.saveval[k].soil.T)[end] for k in 1:length(sol.t)];

dt_save = 3600.0 # hourly diagnostics
# Number of days to plot post spinup
num_days = N_days - N_spinup_days
model_times = Array(0:DATA_DT:(num_days * S_PER_DAY)) .+ t_spinup # post spin-up


# For the data, we also restrict to post-spinup period
data_id_post_spinup = Array(Int64(t_spinup ÷ DATA_DT):1:Int64(tf ÷ DATA_DT))
data_times = Array(0:DATA_DT:(num_days * S_PER_DAY)) .+ t_spinup
# Plotting
savedir = output_dir#generate_output_path("experiments/integrated/fluxnet/$site_ID/out/")

# Monthly averages
GPP = [
    parent(sv.saveval[k].canopy.photosynthesis.GPP)[1] for k in 1:length(sol.t)
]
GPP = GPP .* 1e6 # mol to μmol
GPP_data = drivers.GPP.values[data_id_post_spinup] .* 1e6
ref_time = DateTime(2010)
GPP_model_monthly, _ = compute_monthly_avg(GPP, model_times, ref_time)
GPP_data_monthly, _ = compute_monthly_avg(GPP_data, data_times, ref_time)

# 
R_n = [
    parent(
        sv.saveval[k].drivers.SW_d .- sv.saveval[k].SW_u .+
        sv.saveval[k].drivers.LW_d .- sv.saveval[k].LW_u,
    )[1] for k in 1:length(sol.t)
];
LHF = [
    parent(
        sv.saveval[k].snow.turbulent_fluxes.lhf .+
        sv.saveval[k].soil.turbulent_fluxes.lhf .+
        sv.saveval[k].canopy.energy.turbulent_fluxes.lhf,
    )[1] for k in 1:length(sol.t)
];
SHF = [
    parent(
        sv.saveval[k].canopy.energy.turbulent_fluxes.shf .+
        sv.saveval[k].snow.turbulent_fluxes.shf .+
        sv.saveval[k].soil.turbulent_fluxes.shf,
    )[1] for k in 1:length(sol.t)
]

R_n_model_monthly, _ = compute_monthly_avg(R_n, model_times, ref_time)
SW_u_data = drivers.SW_OUT.values[data_id_post_spinup]
LW_u_data = drivers.LW_OUT.values[data_id_post_spinup]
SW_d_data = drivers.SW_IN.values[data_id_post_spinup]
LW_d_data = drivers.LW_IN.values[data_id_post_spinup]
R_n_data = SW_d_data .- SW_u_data .+ LW_d_data .- LW_u_data
R_n_data_monthly, _ = compute_monthly_avg(R_n_data, model_times, ref_time)

LHF_data = drivers.LE.values[data_id_post_spinup]
LHF_model_monthly, _ = compute_monthly_avg(LHF, model_times, ref_time)
LHF_data_monthly, _ = compute_monthly_avg(LHF_data, data_times, ref_time)

SHF_data = drivers.H.values[data_id_post_spinup]
SHF_model_monthly, _ = compute_monthly_avg(SHF, model_times, ref_time)
SHF_data_monthly, _ = compute_monthly_avg(SHF_data, data_times, ref_time)
fig = Figure(size = (800, 800), fontsize = 19)
ax4 = Axis(
    fig[1, 1],
    xlabel = "",
    ylabel = "Rₙ (W/m²)",
    xticks = 1:12,
    xticklabelsvisible = false,
    xgridvisible = false,
    ygridvisible = false,
)
lines!(ax4, Array(1:1:12), R_n_model_monthly, label = "Model", color = "blue", linewidth = 3)
lines!(ax4, Array(1:1:12), R_n_data_monthly, label = "Data", color = "orange", linewidth = 3)
axislegend(ax4, position = :rt)
ax3 = Axis(
    fig[3, 1],
    xlabel = "",
    ylabel = "H (W/m²)",
    xticks = 1:12,
    xticklabelsvisible = false,
    xgridvisible = false,
    ygridvisible = false,
)
lines!(ax3, Array(1:1:12), SHF_model_monthly, label = "", color = "blue", linewidth = 3)
lines!(ax3, Array(1:1:12), SHF_data_monthly, label = "", color = "orange", linewidth = 3)
ax2 = Axis(
    fig[2, 1],
    xlabel = "",
    ylabel = "L (W/m²)",
    xticks = 1:12,
    xticklabelsvisible = false,
    xgridvisible = false,
    ygridvisible = false,
)
lines!(ax2, Array(1:1:12), LHF_model_monthly, label = "", color = "blue", linewidth = 3)
lines!(ax2, Array(1:1:12), LHF_data_monthly, label = "", color = "orange", linewidth = 3)

ax1 = Axis(
    fig[4, 1],
    xlabel = "Month of year",
    ylabel = "GPP (mol/m²/s)",
    xticks = (
        1:1:12,
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dev",
        ],
    ),
    xgridvisible = false,
    ygridvisible = false,
)
lines!(ax1, Array(1:1:12), GPP_model_monthly, label = "", color = "blue", linewidth = 3)
lines!(ax1, Array(1:1:12), GPP_data_monthly, label = "", color = "orange", linewidth = 3)

CairoMakie.save(joinpath(savedir, "ozark_monthly.png"), fig)

fig = Figure(size = (800, 800), fontsize = 19)
ax0 = Axis(
    fig[4, 1],
    xlabel = "Day of year",
    ylabel = "Temperature (K)",
    xgridvisible = false,
    ygridvisible = false,
    yticks = [270, 290, 310]
)
limits!(
    ax0,
    minimum(model_times ./ 3600 ./ 24),
    maximum(model_times ./ 3600 ./ 24),
    265,
    315,
)

lines!(ax0, model_times ./ 3600 ./ 24, T_soil, label = "Model", color = "blue", linewidth = 3)

lines!(
    ax0,
    data_times ./ 3600 ./ 24,
    drivers.TS.values[data_id_post_spinup],
    label = "Data",
    color = "orange",
    linewidth = 3
)

axislegend(ax0, position = :rt)

ax1 = Axis(
    fig[3, 1],
    xlabel = "",
    ylabel = "Volumetric water(m/m)",
    xticklabelsvisible = false,
    xgridvisible = false,
    ygridvisible = false,
    yticks = [0.2,0.4,0.6]
)
limits!(
    ax1,
    minimum(model_times ./ 3600 ./ 24),
    maximum(model_times ./ 3600 ./ 24),
    0.1,
    0.65,
)
#lines!(ax1, model_times ./ 3600 ./ 24, ϑl, label = "Liquid (model)", color = "cyan")
lines!(ax1, model_times ./ 3600 ./ 24, ϑl .+ θ_i, color = "blue", label = "", linewidth = 3)
lines!(
    ax1,
    data_times ./ 3600 ./ 24,
    drivers.SWC.values[data_id_post_spinup],
    label = "",
    color = "orange",
    linewidth = 3
)
#axislegend(ax1, position = :rt)
ax2 = Axis(
    fig[2, 1],
    xlabel = "",
    ylabel = "Snow water (m)",
    xticklabelsvisible = false,
    xgridvisible = false,
    ygridvisible = false,
    yticks = [0,0.02,0.04]
)
limits!(
    ax2,
    minimum(model_times ./ 3600 ./ 24),
    maximum(model_times ./ 3600 ./ 24),
    0.0,
    0.05,
)

lines!(
    ax2,
    model_times ./ 3600 ./ 24,
    swe,
    color = "blue",
    linewidth = 3
)
#lines!(
#    ax2,
#    model_times ./ 3600 ./ 24,
#    z_snow,
#    label = "Depth (model)",
#    color = "cyan",
#)
axislegend(ax2, position = :rt)
ax3 = Axis(
    fig[1, 1],
    xlabel = "",
    ylabel = "Precip (mm/day)",
    xticklabelsvisible = false,
    xgridvisible = false,
    ygridvisible = false,
)
limits!(
    ax3,
    minimum(model_times ./ 3600 ./ 24),
    maximum(model_times ./ 3600 ./ 24),
    0,
    1000,
)

lines!(
    ax3,
    data_times ./ 3600 ./ 24,
    (drivers.P.values .* (1e3 * 24 * 3600) .* (1 .- snow_frac))[data_id_post_spinup],
    label = "Rain",
    color = "red",
    linewidth = 3
)
lines!(
    ax3,
    data_times ./ 3600 ./ 24,
    (drivers.P.values .* (1e3 * 24 * 3600) .* snow_frac)[data_id_post_spinup],
    label = "Snow",
    color = "green",
    linewidth = 3
)
axislegend(ax3, position = :lt)
CairoMakie.save(joinpath(savedir, "ozark_hydrology.png"), fig)


lwp_data_times = unique(seconds_since_t0 ./ 3600 ./ 24)
isin(pt, list) = pt ∈ list
predawn_mask = isin.((model_times ./ 3600 ./ 24), Ref(lwp_data_times))
fig = Figure(size = (800, 800), fontsize = 19)
ax1 = Axis(
    fig[1, 1],
    xlabel = "Day of year",
    ylabel = "Predawn Leaf Water Potential (MPa)",
    xgridvisible = false,
    ygridvisible = false,
)
limits!(
    ax1,
    minimum(model_times ./ 3600 ./ 24),
    maximum(model_times ./ 3600 ./ 24),
    -200 * 1000 * 9.8 / 1e6,
    0,
)
scatter!(
    ax1,
    model_times[predawn_mask] ./ 3600 ./ 24,
    lwp[predawn_mask] .* (9.8 * 1000 / 1e6),
    label = "Model",
    color = "blue",
)

scatter!(
    ax1,
    seconds_since_t0 ./ 3600 ./ 24,
    lwp_measured .* (9.8 * 1000 / 1e6),
    label = "Data",
    color = "orange",
)
axislegend(ax1, position = :lb)
CairoMakie.save(joinpath(savedir, "lwp.png"), fig)

LAI = [
    parent(sv.saveval[k].canopy.hydraulics.area_index.leaf)[1] for
    k in 1:length(sol.t)
]
LAI_monthly, _ = compute_monthly_avg(LAI, model_times, ref_time)
β = [parent(sv.saveval[k].canopy.hydraulics.β)[1] for k in 1:length(sol.t)]
β_monthly, _ = compute_monthly_avg(β, model_times, ref_time)
fig = Figure(size = (800, 400), fontsize = 18)
ax1 = Axis(
    fig[1, 1],
    xlabel = "Months",
    ylabel = "Leaf Area Index",
    xgridvisible = false,
    ygridvisible = false,
    xticks = 1:12,
)

lines!(ax1, Array(1:1:12), LAI_monthly, label = "Modis", color = "orange")

axislegend(ax1, position = :lb)
ax2 = Axis(
    fig[2, 1],
    xlabel = "Months",
    ylabel = "β-factor",
    xgridvisible = false,
    ygridvisible = false,
    xticks = 1:12,
)

lines!(ax2, Array(1:1:12), β_monthly, label = "Model", color = "blue")

axislegend(ax2, position = :lb)
CairoMakie.save(joinpath(savedir, "lai_β.png"), fig)
