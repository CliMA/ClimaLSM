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

# Read in the site to be run from the command line
if length(ARGS) < 1
    error("Must provide site ID on command line")
end

site_ID = ARGS[1]

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
photosynthesis_args =
    (; parameters = FarquharParameters(FT, is_c3; Vcmax25 = Vcmax25))
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
snow_parameters = SnowParameters{FT}(dt; earth_param_set = earth_param_set);
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
Y.snow.S_l .= 0.0
Y.snow.U .= 0.0

set_initial_cache! = make_set_initial_cache(land)
set_initial_cache!(p, Y, t0);

exp_tendency! = make_exp_tendency(land)
imp_tendency! = make_imp_tendency(land)
jacobian! = make_jacobian(land);
jac_kwargs =
    (; jac_prototype = ClimaLand.ImplicitEquationJacobian(Y), Wfact = jacobian!);


# Callbacks
outdir = joinpath(pkgdir(ClimaLand), "experiments/integrated/fluxnet/out")
output_dir = ClimaUtilities.OutputPathGenerator.generate_output_path(outdir)

d_writer = ClimaDiagnostics.Writers.DictWriter()

ref_time = DateTime(2010)

diags = ClimaLand.default_diagnostics(
    land,
    ref_time;
    output_writer = d_writer,
    output_vars = :long,
    average_period = :hourly,
)

diagnostic_handler =
    ClimaDiagnostics.DiagnosticsHandler(diags, Y, p, t0, dt = dt);

diag_cb = ClimaDiagnostics.DiagnosticsCallback(diagnostic_handler);

## How often we want to update the drivers. Note that this uses the defined `t0` and `tf`
## defined in the simulatons file
updateat = Array(t0:DATA_DT:tf)
model_drivers = ClimaLand.get_drivers(land)
updatefunc = ClimaLand.make_update_drivers(model_drivers)
driver_cb = ClimaLand.DriverUpdateCallback(updateat, updatefunc)
cb = SciMLBase.CallbackSet(driver_cb, diag_cb)


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

sol = SciMLBase.solve(prob, ode_algo; dt = dt, callback = cb);

# Extract model output from the saved diagnostics
short_names_1D = [
    "sif", # SIF
    "ra", # AR
    "gs", # g_stomata
    "gpp", # GPP
    "ct", # canopy_T
    "swu", # SW_u
    "lwu", # LW_u
    "er", # ER
    "et", # ET
    "msf", # β
    "shf", # SHF
    "lhf", # LHF
    "ghf", # G
    "rn", # Rn
    "swe",
]
short_names_2D = [
    "swc", # swc_sfc or swc_5 or swc_10
    "tsoil", # soil_T_sfc or soil_T_5 or soil_T_10
    "si", # si_sfc or si_5 or si_10
]

hourly_diag_name = short_names_1D .* "_1h_average"
hourly_diag_name_2D = short_names_2D .* "_1h_average"


# diagnostic_as_vectors()[2] is a vector of a variable,
# whereas diagnostic_as_vectors()[1] is a vector or time associated with that variable.
# We index to only extract the period post-spinup.
SIF, AR, g_stomata, GPP, canopy_T, SW_u, LW_u, ER, ET, β, SHF, LHF, G, Rn, SWE =
    [
        ClimaLand.Diagnostics.diagnostic_as_vectors(d_writer, diag_name)[2][(N_spinup_days * 24):end]
        for diag_name in hourly_diag_name
    ]

swc, soil_T, si = [
    ClimaLand.Diagnostics.diagnostic_as_vectors(
        d_writer,
        diag_name;
        layer = 20, #surface layer
    )[2][(N_spinup_days * 24):end] for diag_name in hourly_diag_name_2D
]
dt_save = 3600.0 # hourly diagnostics
# Number of days to plot post spinup
num_days = N_days - N_spinup_days
model_times = Array(0:dt_save:(num_days * S_PER_DAY)) .+ t_spinup # post spin-up

# convert units for GPP and ET
GPP = GPP .* 1e6 # mol to μmol
AR = AR .* 1e6
ET = ET .* 24 .* 3600

# For the data, we also restrict to post-spinup period
data_id_post_spinup = Array(Int64(t_spinup ÷ DATA_DT):1:Int64(tf ÷ DATA_DT))
data_times = Array(0:DATA_DT:(num_days * S_PER_DAY)) .+ t_spinup
# Plotting
savedir = generate_output_path("experiments/integrated/fluxnet/$site_ID/out/")

if !isdir(savedir)
    mkdir(savedir)
end

# Plot model diurnal cycles without data comparisons
# Autotrophic Respiration
plot_daily_avg(
    "AutoResp",
    AR,
    dt_save,
    num_days,
    "μmol/m^2/s",
    savedir,
    "Model",
)

# Plot all comparisons of model diurnal cycles to data diurnal cycles
# GPP
if drivers.GPP.status == absent
    plot_daily_avg(
        "GPP",
        GPP,
        dt_save,
        num_days,
        "μmol/m^2/s",
        savedir,
        "Model",
    )
else
    GPP_data = drivers.GPP.values[data_id_post_spinup] .* 1e6
    plot_avg_comp(
        "GPP",
        GPP,
        dt_save,
        GPP_data,
        FT(DATA_DT),
        num_days,
        drivers.GPP.units,
        savedir,
    )
end

# Upwelling shortwave radiation is referred to as outgoing in the data
if drivers.SW_OUT.status == absent
    plot_daily_avg("SW up", SW_u, dt_save, num_days, "w/m^2", savedir, "model")
else
    SW_u_data = drivers.SW_OUT.values[data_id_post_spinup]
    plot_avg_comp(
        "SW up",
        SW_u,
        dt_save,
        SW_u_data,
        FT(DATA_DT),
        num_days,
        drivers.SW_OUT.units,
        savedir,
    )
end

# Upwelling longwave radiation is referred to outgoing in the data
if drivers.LW_OUT.status == absent
    plot_daily_avg("LW up", LW_u, dt_save, num_days, "w/m^2", savedir, "model")
else
    LW_u_data = drivers.LW_OUT.values[data_id_post_spinup]
    plot_avg_comp(
        "LW up",
        LW_u,
        dt_save,
        LW_u_data,
        FT(DATA_DT),
        num_days,
        drivers.LW_OUT.units,
        savedir,
    )
end

# ET
if drivers.LE.status == absent
    plot_daily_avg("ET", ET, dt_save, num_days, "mm/day", savedir, "Model")
else
    measured_T =
        drivers.LE.values ./ (LP.LH_v0(earth_param_set) * 1000) .*
        (1e3 * 24 * 3600)
    ET_data = measured_T[data_id_post_spinup]
    plot_avg_comp(
        "ET",
        ET,
        dt_save,
        ET_data,
        FT(DATA_DT),
        num_days,
        "mm/day",
        savedir,
    )
end

# Sensible Heat Flux
if drivers.H.status == absent
    plot_daily_avg("SHF", SHF, dt_save, num_days, "w/m^2", savedir, "Model")
else
    SHF_data = drivers.H.values[data_id_post_spinup]
    plot_avg_comp(
        "SHF",
        SHF,
        dt_save,
        SHF_data,
        FT(DATA_DT),
        num_days,
        drivers.H.units,
        savedir,
    )
end

# Latent Heat Flux
if drivers.LE.status == absent
    plot_daily_avg("LHF", LHF, dt_save, num_days, "w/m^2", savedir)
else
    LHF_data = drivers.LE.values[data_id_post_spinup]
    plot_avg_comp(
        "LHF",
        LHF,
        dt_save,
        LHF_data,
        FT(DATA_DT),
        num_days,
        drivers.LE.units,
        savedir,
    )
end

# Water stress factor
plot_daily_avg("moisture_stress", β, dt_save, num_days, "", savedir, "Model")

# Stomatal conductance
plot_daily_avg(
    "stomatal_conductance",
    g_stomata,
    dt_save,
    num_days,
    "m s^-1",
    savedir,
    "Model",
)

if isfile(
    joinpath(
        climaland_dir,
        "experiments/integrated/fluxnet/$site_ID/Artifacts.toml",
    ),
)
    rm(
        joinpath(
            climaland_dir,
            "experiments/integrated/fluxnet/$site_ID/Artifacts.toml",
        ),
    )
end

# Water content in soil and snow
# Soil water content
# Current resolution has the first layer at 0.1 cm, the second at 5cm.
fig = Figure(size = (1500, 800), fontsize = 20)
ax1 = Axis(fig[3, 1], xlabel = "Days", ylabel = "SWC [m/m]")
limits!(
    ax1,
    minimum(model_times ./ 3600 ./ 24),
    maximum(model_times ./ 3600 ./ 24),
    0.05,
    0.6,
)
lines!(ax1, model_times ./ 3600 ./ 24, swc, label = "1.25cm", color = "blue")
lines!(
    ax1,
    model_times ./ 3600 ./ 24,
    si,
    color = "cyan",
    label = "Ice, 1.25cm",
)

if drivers.SWC.status != absent
    lines!(
        ax1,
        data_times ./ 3600 ./ 24,
        drivers.SWC.values[data_id_post_spinup],
        label = "Data",
    )
end
ax2 =
    Axis(fig[2, 1], xlabel = "", ylabel = "SWE [m]", xticklabelsvisible = false)
limits!(
    ax2,
    minimum(model_times ./ 3600 ./ 24),
    maximum(model_times ./ 3600 ./ 24),
    0.0,
    maximum(SWE),
)

lines!(ax2, model_times ./ 3600 ./ 24, SWE, label = "Model", color = "blue")
ax3 = Axis(
    fig[1, 1],
    xlabel = "",
    ylabel = "Precipitation [mm/day]",
    xticklabelsvisible = false,
)
limits!(
    ax3,
    minimum(model_times ./ 3600 ./ 24),
    maximum(model_times ./ 3600 ./ 24),
    -500,
    0.0,
)

lines!(
    ax3,
    data_times ./ 3600 ./ 24,
    (drivers.P.values .* (-1e3 * 24 * 3600) .* (1 .- snow_frac))[data_id_post_spinup],
    label = "Rain (data)",
)
lines!(
    ax3,
    data_times ./ 3600 ./ 24,
    (drivers.P.values .* (-1e3 * 24 * 3600) .* snow_frac)[data_id_post_spinup],
    label = "Snow (data)",
)
CairoMakie.save(joinpath(savedir, "ground_water_content.png"), fig)



fig2 = Figure(size = (1500, 800))
ax12 = Axis(fig2[1, 1], xlabel = "Days", ylabel = "Temperature (K)")
limits!(
    ax12,
    minimum(model_times ./ 3600 ./ 24),
    maximum(model_times ./ 3600 ./ 24),
    265,
    315,
)

lines!(
    ax12,
    model_times ./ 3600 ./ 24,
    soil_T,
    label = "1.25cm",
    color = "blue",
)

if drivers.TS.status != absent
    lines!(
        ax12,
        data_times ./ 3600 ./ 24,
        drivers.TS.values[data_id_post_spinup],
        label = "Data",
    )
end

CairoMakie.save(joinpath(savedir, "soil_temperature.png"), fig2)
