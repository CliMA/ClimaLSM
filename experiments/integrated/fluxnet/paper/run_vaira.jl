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

site_ID = "US-Var"

# Read all site-specific domain parameters from the simulation file for the site
include(
    joinpath(
        climaland_dir,
        "experiments/integrated/fluxnet/$site_ID/$(site_ID)_simulation.jl",
    ),
)


# Read all site-specific parameters from the parameter file for the site
include(
    joinpath(
        climaland_dir,
        "experiments/integrated/fluxnet/$site_ID/$(site_ID)_parameters.jl",
    ),
)
include(
    joinpath(
        climaland_dir,
        "experiments/integrated/fluxnet/fluxnet_domain.jl",
    ),
)
include(
    joinpath(
        climaland_dir,
        "experiments/integrated/fluxnet/fluxnet_simulation.jl",
    ),
)
# This reads in the data from the flux tower site and creates
# the atmospheric and radiative driver structs for the model

include(
    joinpath(
        climaland_dir,
        "experiments/integrated/fluxnet/met_drivers_FLUXNET_Var.jl",
    ),
)

# Now we set up the model. For the soil model, we pick
# a model type and model args:
soil_domain = land_domain
hydrology_cm = vanGenuchten{FT}(; α = soil_vg_α, n = soil_vg_n)
soil_ps = Soil.EnergyHydrologyParameters(
    FT;
    ν = soil_ν,
    ν_ss_om,
    ν_ss_quartz,
    ν_ss_gravel,
    hydrology_cm = hydrology_cm,
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

Y.soil.ϑ_l = FT(0.3)

# Both data and simulation are reference to 2005-01-01-00 (LOCAL)
# or 2005-01-01-06 (UTC)
Y.soil.θ_i = FT(0.0)
T_0 =
    drivers.TS5.status != absent ?
    drivers.TS5.values[1 + Int(round(t0 / DATA_DT))] :
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
outdir = joinpath(
    pkgdir(ClimaLand),
    "experiments/integrated/fluxnet/out_($pc)_($sc)_$(Weibull_param)_($a)_($ψ63)_($plant_ν)_($K_sat_plant)",
)
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

# Extract model output from the saved diagnostics
lwp = [parent(sv.saveval[k].canopy.hydraulics.ψ.:1)[1] for k in 1:length(sol.t)]
θ1 = [
    parent(sol.u[k].soil.ϑ_l .+ sol.u[k].soil.θ_i)[end] for k in 1:length(sol.t)
];
θ2 = [
    parent(sol.u[k].soil.ϑ_l .+ sol.u[k].soil.θ_i)[end - 2] for
    k in 1:length(sol.t)
];
θ3 = [
    parent(sol.u[k].soil.ϑ_l .+ sol.u[k].soil.θ_i)[end - 5] for
    k in 1:length(sol.t)
];
T_soil1 = [parent(sv.saveval[k].soil.T)[end] for k in 1:length(sol.t)]; # 2cm
T_soil3 = [parent(sv.saveval[k].soil.T)[end - 2] for k in 1:length(sol.t)];
T_soil4 = [parent(sv.saveval[k].soil.T)[end - 4] for k in 1:length(sol.t)];
T_soil5 = [parent(sv.saveval[k].soil.T)[end - 9] for k in 1:length(sol.t)];

dt_save = 3600.0 # hourly diagnostics
# Number of days to plot post spinup
num_days = N_days - N_spinup_days
model_times = Array(0:DATA_DT:(num_days * S_PER_DAY)) .+ t_spinup # post spin-up


# For the data, we also restrict to post-spinup period
data_id_post_spinup =
    Array(Int64(t_spinup ÷ DATA_DT + 1):1:Int64(tf ÷ DATA_DT + 1))
data_times = Array(0:DATA_DT:(num_days * S_PER_DAY)) .+ t_spinup
# Plotting
savedir = output_dir#generate_output_path("experiments/integrated/fluxnet/$site_ID/out/")

# Monthly averages
GPP = [
    parent(sv.saveval[k].canopy.photosynthesis.GPP)[1] for k in 1:length(sol.t)
]
GPP = GPP .* 1e6 # mol to μmol
GPP_data = drivers.GPP.values[data_id_post_spinup] .* 1e6
GPP_model_monthly, _ = compute_monthly_avg(GPP, model_times, ref_time)
GPP_data_monthly, _ = compute_monthly_avg(GPP_data, data_times, ref_time)

# 
SW_u = [
    parent(sv.saveval[k].SW_u)[1] for
    k in 1:length(sol.t)
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

SW_u_model_monthly, _ = compute_monthly_avg(SW_u, model_times, ref_time)
SW_u_data = drivers.SW_OUT.values[data_id_post_spinup]
SW_d_data = drivers.SW_IN.values[data_id_post_spinup]
SW_n_data = SW_d_data .- SW_u_data
SW_u_data_monthly, _ = compute_monthly_avg(SW_u_data, model_times, ref_time)

LHF_data = drivers.LE.values[data_id_post_spinup]
LHF_model_monthly, _ = compute_monthly_avg(LHF, model_times, ref_time)
LHF_data_monthly, _ = compute_monthly_avg(LHF_data, data_times, ref_time)

SHF_data = drivers.H.values[data_id_post_spinup]
SHF_model_monthly, _ = compute_monthly_avg(SHF, model_times, ref_time)
SHF_data_monthly, _ = compute_monthly_avg(SHF_data, data_times, ref_time)
fig = Figure(size = (800, 800), fontsize = 24)
ax4 = Axis(
    fig[1, 1],
    xlabel = "",
    ylabel = "SWᵤ (W/m²)",
    xticks = 1:12,
    xticklabelsvisible = false,
    xgridvisible = false,
    ygridvisible = false,
)
lines!(ax4, Array(1:1:12), SW_u_model_monthly, label = "Model", color = "blue", linewidth = 3)
lines!(ax4, Array(1:1:12), SW_u_data_monthly, label = "Data", color = "orange", linewidth = 3)
@show mean(abs.(SW_u_model_monthly .- SW_u_data_monthly))
axislegend(ax4, position = :rt, framevisible=false)
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
@show mean(abs.(SHF_model_monthly .- SHF_data_monthly))
ax2 = Axis(
    fig[2, 1],
    xlabel = "",
    ylabel = "LE (W/m²)",
    xticks = 1:12,
    xticklabelsvisible = false,
    xgridvisible = false,
    ygridvisible = false,
)
lines!(ax2, Array(1:1:12), LHF_model_monthly, label = "", color = "blue", linewidth = 3)
lines!(ax2, Array(1:1:12), LHF_data_monthly, label = "", color = "orange", linewidth = 3)
@show mean(abs.(LHF_model_monthly .- LHF_data_monthly))
ax1 = Axis(
    fig[4, 1],
    xlabel = "Month of year",
    ylabel = "GPP (μmol/m²/s)",
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
@show mean(abs.(GPP_model_monthly .- GPP_data_monthly))
CairoMakie.save(joinpath(savedir, "vaira_monthly.png"), fig)




fig = Figure(size = (800, 800), fontsize = 24)
ax0 = Axis(
    fig[4, 1],
    xlabel = "Day of year",
    #ylabel = "Temp (K; 2cm)",
    xgridvisible = false,
    ygridvisible = false,
    yticks = [270,290,310]
)
limits!(
    ax0,
    minimum(model_times ./ 3600 ./ 24),
    maximum(model_times ./ 3600 ./ 24),
    270,
    320,
)
text!(ax0, 310, 310, text = "2 cm")
lines!(ax0, model_times ./ 3600 ./ 24, T_soil1, label = "", color = "blue", linewidth = 3)

lines!(
    ax0,
    data_times ./ 3600 ./ 24,
    drivers.TS1.values[data_id_post_spinup],
    label = "",
    color = "orange",
    linewidth = 3,
)
ax1 = Axis(
    fig[3, 1],
    xlabel = "",
    xticklabelsvisible = false,
    #ylabel = "Temp (K; 32cm)",
    xgridvisible = false,
    ygridvisible = false,
    yticks = [270,290,310]
)
limits!(
    ax1,
    minimum(model_times ./ 3600 ./ 24),
    maximum(model_times ./ 3600 ./ 24),
    270,
    320,
)
text!(ax1, 310, 310, text = "32 cm")
lines!(ax1, model_times ./ 3600 ./ 24, T_soil5, label = "", color = "blue", linewidth = 3)

lines!(
    ax1,
    data_times ./ 3600 ./ 24,
    drivers.TS5.values[data_id_post_spinup],
    label = "",
    color = "orange",
    linewidth = 3)


ax2 = Axis(
    fig[2, 1],
    xlabel = "",
    #ylabel = "SWC [m/m; near surface]",
    xticklabelsvisible = false,
    xgridvisible = false,
    ygridvisible = false,
    yticks = [0,0.2,0.4]
)
limits!(
    ax2,
    minimum(model_times ./ 3600 ./ 24),
    maximum(model_times ./ 3600 ./ 24),
    0.0,
    0.5,
)
text!(ax2, 310, 0, text = "0-2 cm")
lines!(ax2, model_times ./ 3600 ./ 24, θ1, color = "blue", label = "", linewidth = 3)
lines!(
    ax2,
    data_times ./ 3600 ./ 24,
    drivers.SWC1.values[data_id_post_spinup],
    label = "",
    color = "orange",
    linewidth = 3)

ax3 = Axis(
    fig[1, 1],
    xlabel = "",
    #ylabel = "SWC [m/m; 20cm]",
    xticklabelsvisible = false,
    xgridvisible = false,
    ygridvisible = false,
    yticks = [0,0.2,0.4]
)
limits!(
    ax3,
    minimum(model_times ./ 3600 ./ 24),
    maximum(model_times ./ 3600 ./ 24),
    0.0,
    0.5,
)
text!(ax3, 310, 0, text = "20 cm")
lines!(ax3, model_times ./ 3600 ./ 24, θ3, color = "blue", label = "Model", linewidth = 3)
lines!(
    ax3,
    data_times ./ 3600 ./ 24,
    drivers.SWC3.values[data_id_post_spinup],
    label = "Data",
    color = "orange", linewidth = 3)

ylabel01 = Label(fig[1:2, 0], "Volumetric Water (m/m)", rotation = pi / 2)
ylabel23 = Label(fig[3:4, 0], "Temperature (K)", rotation = pi / 2)

CairoMakie.save(joinpath(savedir, "vaira_soil.png"), fig)

LAI = [
    parent(sv.saveval[k].canopy.hydraulics.area_index.leaf)[1] for
    k in 1:length(sol.t)
]
LAI_monthly, _ = compute_monthly_avg(LAI, model_times, ref_time)
β = [parent(sv.saveval[k].canopy.hydraulics.β)[1] for k in 1:length(sol.t)]
β_monthly, _ = compute_monthly_avg(β, model_times, ref_time)
fig = Figure(size = (800, 800), fontsize = 18)
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
    xlabel = "",
    ylabel = "β-factor",
    xgridvisible = false,
    ygridvisible = false,
    xticks = 1:12,
)

lines!(ax2, Array(1:1:12), β_monthly, label = "Model", color = "blue")

axislegend(ax2, position = :lb)

ax3 = Axis(
    fig[3, 1],
    xlabel = "",
    ylabel = "Leaf Water Potential",
    xgridvisible = false,
    ygridvisible = false,
    xticks = 1:12,
)
lwp_monthly, _ = compute_monthly_avg(lwp, model_times, ref_time)
lines!(ax3, Array(1:1:12), lwp_monthly, label = "Model", color = "blue")

axislegend(ax3, position = :lb)
ψ = [parent(sv.saveval[k].soil.ψ)[end - 7] for k in 1:length(sol.t)] # 1m depth
ψ_monthly, _ = compute_monthly_avg(ψ, model_times, ref_time)
ax4 = Axis(
    fig[4, 1],
    xlabel = "",
    ylabel = "ψ (1m) soil",
    xgridvisible = false,
    ygridvisible = false,
    xticks = 1:12,
)

lines!(ax4, Array(1:1:12), ψ_monthly, label = "Model", color = "blue")

axislegend(ax4, position = :lb)
CairoMakie.save(joinpath(savedir, "lai_β_lwp.png"), fig)
