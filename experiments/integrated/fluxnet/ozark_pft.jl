"""
This experiment tests running the Ozark site (US-MOz) using plant parameters 
defined by plant functional types instead of fully site-specific parameters.
"""

import ClimaLand

import SciMLBase
import ClimaTimeSteppers as CTS
using ClimaCore
import ClimaParams as CP
using Plots
using Statistics
using Dates
using Insolation
using StatsBase

using ClimaLand
using ClimaLand.Domains: Column
using ClimaLand.Soil
using ClimaLand.Soil.Biogeochemistry
using ClimaLand.Canopy
using ClimaLand.Canopy.PlantHydraulics
import ClimaLand
import ClimaLand.Parameters as LP
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

# Read in the site-specific parameters for all parameters not defined by the PFTs
include(
    joinpath(
        climaland_dir,
        "experiments/integrated/fluxnet/$site_ID/$(site_ID)_parameters.jl",
    ),
)

# Define the PFT land cover percentages for the Ozark site. Currently we only
# use the dominant PFT, which for Ozark is deciduous broadleaf temperate trees.
pft_pcts = [
    0.0, # NET_Temp
    0.0, # NET_Bor
    0.0, # NDT_Bor
    0.0, # BET_Trop
    0.0, # BET_Temp
    0.0, # BDT_Trop
    1.0, # BDT_Temp
    0.0, # BDT_Bor
    0.0, # BES_Temp
    0.0, # BDS_Temp
    0.0, # BDT_Bor
    0.0, # C3G_A
    0.0, # C3G_NA
    0.0, # C4G
]

# Load the PFT parameters into the namespace
(
    Ω,
    α_PAR_leaf,
    α_NIR_leaf,
    τ_PAR_leaf,
    τ_NIR_leaf,
    ϵ_canopy,
    χl,
    ac_canopy,
    g1,
    Vcmax25,
    f_root_to_shoot,
    K_sat_plant,
    ψ63,
    plant_ν,
    rooting_depth,
) = FT.(params_from_pfts(pft_pcts))

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
    ν_ss_om = ν_ss_om,
    ν_ss_quartz = ν_ss_quartz,
    ν_ss_gravel = ν_ss_gravel,
    hydrology_cm = vanGenuchten{FT}(; α = soil_vg_α, n = soil_vg_n),
    K_sat = soil_K_sat,
    S_s = soil_S_s,
    θ_r = θ_r,
    earth_param_set = earth_param_set,
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

soilco2_ps = SoilCO2ModelParameters(
    FT;
    ν = soil_ν, # same as soil
)

# soil microbes args
Csom = (z, t) -> eltype(z)(5.0)

# Set the soil CO2 BC to being atmospheric CO2
soilco2_top_bc = Soil.Biogeochemistry.AtmosCO2StateBC()
soilco2_bot_bc = Soil.Biogeochemistry.SoilCO2FluxBC((p, t) -> 0.0) # no flux
soilco2_sources = (MicrobeProduction{FT}(),)

soilco2_boundary_conditions = (; top = soilco2_top_bc, bottom = soilco2_bot_bc)

soilco2_drivers = Soil.Biogeochemistry.SoilDrivers(
    Soil.Biogeochemistry.PrognosticMet{FT}(),
    Soil.Biogeochemistry.PrescribedSOC{FT}(Csom),
    atmos,
)

soilco2_args = (;
    boundary_conditions = soilco2_boundary_conditions,
    sources = soilco2_sources,
    domain = soil_domain,
    parameters = soilco2_ps,
    drivers = soilco2_drivers,
)

# Now we set up the canopy model, which we set up by component:
# Component Types
canopy_component_types = (;
    autotrophic_respiration = Canopy.AutotrophicRespirationModel{FT},
    radiative_transfer = Canopy.TwoStreamModel{FT},
    photosynthesis = Canopy.FarquharModel{FT},
    conductance = Canopy.MedlynConductanceModel{FT},
    hydraulics = Canopy.BigLeafHydraulicsModel{FT},
    energy = Canopy.BigLeafEnergyModel{FT},
)
# Individual Component arguments
# Set up autotrophic respiration
autotrophic_respiration_args =
    (; parameters = AutotrophicRespirationParameters(FT))
# Set up radiative transfer
G_Function = CLMGFunction(χl)
radiative_transfer_args = (;
    parameters = TwoStreamParameters(
        FT;
        Ω,
        G_Function,
        α_PAR_leaf,
        τ_PAR_leaf,
        α_NIR_leaf,
        τ_NIR_leaf,
    )
)
# Set up conductance
conductance_args = (; parameters = MedlynConductanceParameters(FT; g1))
# Set up photosynthesis
# Set up photosynthesis
photosynthesis_args =
    (; parameters = FarquharParameters(FT, Canopy.C3(); Vcmax25 = Vcmax25))
# Set up plant hydraulics
ai_parameterization = PrescribedSiteAreaIndex{FT}(LAIfunction, SAI, RAI)

function root_distribution(z::T; rooting_depth = rooting_depth) where {T}
    return T(1.0 / rooting_depth) * exp(z / T(rooting_depth)) # 1/m
end

plant_hydraulics_ps = PlantHydraulics.PlantHydraulicsParameters(;
    ai_parameterization = ai_parameterization,
    ν = plant_ν,
    S_s = plant_S_s,
    root_distribution = root_distribution,
    conductivity_model = conductivity_model,
    retention_model = retention_model,
)
plant_hydraulics_args =
    (parameters = plant_hydraulics_ps, h_stem = h_stem, h_leaf = h_leaf)

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

# Integrated plant hydraulics and soil model
land_input = (atmos = atmos, radiation = radiation)
land = SoilCanopyModel{FT}(;
    soilco2_type = soilco2_type,
    soilco2_args = soilco2_args,
    land_args = land_input,
    soil_model_type = soil_model_type,
    soil_args = soil_args,
    canopy_component_types = canopy_component_types,
    canopy_component_args = canopy_component_args,
    canopy_model_args = canopy_model_args,
)
Y, p, cds = initialize(land)
exp_tendency! = make_exp_tendency(land)

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
    volumetric_heat_capacity.(Y.soil.ϑ_l, Y.soil.θ_i, Ref(land.soil.parameters))
Y.soil.ρe_int =
    volumetric_internal_energy.(
        Y.soil.θ_i,
        ρc_s,
        T_0,
        Ref(land.soil.parameters),
    )
Y.soilco2.C .= FT(0.000412) # set to atmospheric co2, mol co2 per mol air
ψ_stem_0 = FT(-1e5 / 9800) # pressure in the leaf divided by rho_liquid*gravitational acceleration [m] 
ψ_leaf_0 = FT(-2e5 / 9800)
ψ_comps = h_stem > 0 ? [ψ_stem_0, ψ_leaf_0] : ψ_leaf_0

S_l_ini =
    inverse_water_retention_curve.(retention_model, ψ_comps, plant_ν, plant_S_s)

for i in 1:(h_stem < 0 + 1)
    Y.canopy.hydraulics.ϑ_l.:($i) .=
        augmented_liquid_fraction.(plant_ν, S_l_ini[i])
end

Y.canopy.energy.T = drivers.TA.values[1 + Int(round(t0 / DATA_DT))] # Get atmos temperature at t0

set_initial_cache! = make_set_initial_cache(land)
set_initial_cache!(p, Y, t0);

# Simulation
sv = (;
    t = Array{Float64}(undef, length(saveat)),
    saveval = Array{NamedTuple}(undef, length(saveat)),
)
saving_cb = ClimaLand.NonInterpSavingCallback(sv, saveat)
## How often we want to update the drivers. Note that this uses the defined `t0` and `tf`
## defined in the simulatons file
updateat = Array(t0:DATA_DT:tf)
updatefunc = ClimaLand.make_update_drivers(atmos, radiation)
driver_cb = ClimaLand.DriverUpdateCallback(updateat, updatefunc)
cb = SciMLBase.CallbackSet(driver_cb, saving_cb)

prob = SciMLBase.ODEProblem(
    CTS.ClimaODEFunction((T_exp!) = exp_tendency!),
    Y,
    (t0, tf),
    p,
);
sol = SciMLBase.solve(
    prob,
    ode_algo;
    dt = dt,
    callback = cb,
    adaptive = false,
    saveat = saveat,
)

# Plotting
daily = sol.t ./ 3600 ./ 24
if !isdir(
    joinpath(climaland_dir, "experiments/integrated/fluxnet/$site_ID/out"),
)
    mkdir(
        joinpath(climaland_dir, "experiments/integrated/fluxnet/$site_ID/out"),
    )
end
savedir =
    joinpath(climaland_dir, "experiments/integrated/fluxnet/$site_ID/out/pft/")
if !isdir(savedir)
    mkdir(savedir)
end

# Number of days to plot
num_days = N_days - N_spinup_days

# Time series of model and data outputs
data_times = [0:DATA_DT:(num_days * S_PER_DAY);]
model_times = [0:(n * dt):(num_days * S_PER_DAY);]

# Plot model diurnal cycles without data comparisons
# Autotrophic Respiration
AR =
    [
        parent(sv.saveval[k].canopy.autotrophic_respiration.Ra)[1] for
        k in 1:length(sv.saveval)
    ] .* 1e6
plot_daily_avg("AutoResp", AR, dt * n, num_days, "μmol/m^2/s", savedir, "Model")

# Plot all comparisons of model diurnal cycles to data diurnal cycles
# GPP
model_GPP =
    [
        parent(sv.saveval[k].canopy.photosynthesis.GPP)[1] for
        k in 1:length(sv.saveval)
    ] .* 1e6

if drivers.GPP.status == absent
    plot_daily_avg(
        "GPP",
        model_GPP,
        dt * n,
        num_days,
        "μmol/m^2/s",
        savedir,
        "Model",
    )
else
    GPP_data =
        drivers.GPP.values[Int64(t_spinup ÷ DATA_DT):Int64(tf ÷ DATA_DT)] .* 1e6
    plot_avg_comp(
        "GPP",
        model_GPP,
        dt * n,
        GPP_data,
        FT(DATA_DT),
        num_days,
        drivers.GPP.units,
        savedir,
    )
end

# SW_OUT
SW_out_model = [parent(sv.saveval[k].SW_out)[1] for k in 1:length(sv.saveval)]
if drivers.SW_OUT.status == absent
    plot_daily_avg(
        "SW OUT",
        SW_out_model,
        dt * n,
        num_days,
        "w/m^2",
        savedir,
        "model",
    )
else
    SW_out_data = FT.(drivers.SW_OUT.values)[Int64(t_spinup ÷ DATA_DT):Int64(
        tf ÷ DATA_DT,
    )]
    plot_avg_comp(
        "SW OUT",
        SW_out_model,
        dt * n,
        SW_out_data,
        FT(DATA_DT),
        num_days,
        drivers.SW_OUT.units,
        savedir,
    )
end

# LW_OUT
LW_out_model = [parent(sv.saveval[k].LW_out)[1] for k in 1:length(sv.saveval)]
if drivers.LW_OUT.status == absent
    plot_daily_avg(
        "LW OUT",
        LW_out_model,
        dt * n,
        num_days,
        "w/m^2",
        savedir,
        "model",
    )
else
    LW_out_data = FT.(drivers.LW_OUT.values)[Int64(t_spinup ÷ DATA_DT):Int64(
        tf ÷ DATA_DT,
    )]
    plot_avg_comp(
        "LW OUT",
        LW_out_model,
        dt * n,
        LW_out_data,
        FT(DATA_DT),
        num_days,
        drivers.LW_OUT.units,
        savedir,
    )
end

# ET
T =
    [
        parent(sv.saveval[k].canopy.conductance.transpiration)[1] for
        k in 1:length(sol.t)
    ] .* (1e3 * 24 * 3600)
E =
    [
        parent(sv.saveval[k].soil.turbulent_fluxes.vapor_flux)[1] for
        k in 1:length(sol.t)
    ] .* (1e3 * 24 * 3600)
ET_model = T .+ E
if drivers.LE.status == absent
    plot_daily_avg("ET", ET_model, dt * n, num_days, "mm/day", savedir, "Model")
else
    measured_T =
        drivers.LE.values ./ (LP.LH_v0(earth_param_set) * 1000) .*
        (1e3 * 24 * 3600)
    ET_data = measured_T[Int64(t_spinup ÷ DATA_DT):Int64(tf ÷ DATA_DT)]
    plot_avg_comp(
        "ET",
        ET_model,
        dt * n,
        ET_data,
        FT(DATA_DT),
        num_days,
        "mm/day",
        savedir,
    )
end

# Sensible Heat Flux
SHF_soil = [
    parent(sv.saveval[k].soil.turbulent_fluxes.shf)[1] for k in 1:length(sol.t)
]
SHF_canopy =
    [parent(sv.saveval[k].canopy.energy.shf)[1] for k in 1:length(sol.t)]
SHF_model = SHF_soil + SHF_canopy
if drivers.H.status == absent
    plot_daily_avg(
        "SHF",
        SHF_model,
        dt * n,
        num_days,
        "w/m^2",
        savedir,
        "Model",
    )
else
    SHF_data = drivers.H.values[Int64(t_spinup ÷ DATA_DT):Int64(tf ÷ DATA_DT)]
    plot_avg_comp(
        "SHF",
        SHF_model,
        dt * n,
        SHF_data,
        FT(DATA_DT),
        N_days - N_spinup_days,
        drivers.H.units,
        savedir,
    )
end

# Latent Heat Flux
LHF_soil = [
    parent(sv.saveval[k].soil.turbulent_fluxes.lhf)[1] for k in 1:length(sol.t)
]
LHF_canopy =
    [parent(sv.saveval[k].canopy.energy.lhf)[1] for k in 1:length(sol.t)]
LHF_model = LHF_soil + LHF_canopy
if drivers.LE.status == absent
    plot_daily_avg("LHF", LHF_model, dt * n, num_days, "w/m^2", savedir)
else
    LHF_data = drivers.LE.values[Int64(t_spinup ÷ DATA_DT):Int64(tf ÷ DATA_DT)]
    plot_avg_comp(
        "LHF",
        LHF_model,
        dt * n,
        LHF_data,
        FT(DATA_DT),
        N_days - N_spinup_days,
        drivers.LE.units,
        savedir,
    )
end

# Ground Heat Flux
# In the land model, positive fluxes are always in the upwards direction
# In the data, a positive G indicates a flux into the soil, so we adjust the sign
# on the data G
if drivers.G.status != absent
    G_data_avg = compute_diurnal_avg(
        FT.(drivers.G.values)[Int64(t_spinup ÷ DATA_DT):Int64(tf ÷ DATA_DT)],
        data_times,
        num_days,
    )
    plt1 = Plots.plot(0.5:0.5:24, -1 .* G_data_avg, label = "Data: -G")
    Plots.plot!(
        plt1,
        ylabel = "Flux (W/m^2)",
        title = "Energy balance at the site",
    )
    if drivers.LE.status != absent &&
       drivers.H.status != absent &&
       drivers.SW_OUT.status != absent &&
       drivers.LW_OUT.status != absent
        Rn =
            (drivers.SW_IN.values .- drivers.SW_OUT.values) .+
            (drivers.LW_IN.values .- drivers.LW_OUT.values)
        G_alternate_data_avg = compute_diurnal_avg(
            FT.(drivers.H.values .+ drivers.LE.values .- Rn)[Int64(
                t_spinup ÷ DATA_DT,
            ):Int64(tf ÷ DATA_DT)],
            data_times,
            num_days,
        )
        HplusL_avg = compute_diurnal_avg(
            FT.(drivers.H.values .+ drivers.LE.values)[Int64(
                t_spinup ÷ DATA_DT,
            ):Int64(tf ÷ DATA_DT)],
            data_times,
            num_days,
        )
        RminusG_avg = compute_diurnal_avg(
            FT.(Rn .- drivers.G.values)[Int64(t_spinup ÷ DATA_DT):Int64(
                tf ÷ DATA_DT,
            )],
            data_times,
            num_days,
        )
        Plots.plot!(
            plt1,
            0.5:0.5:24,
            G_alternate_data_avg,
            label = "Data: (H+L-Rn)_site",
            xlabel = "Hour of day",
        )
        plt2 = Plots.scatter(
            HplusL_avg,
            RminusG_avg,
            label = "Diurnally averaged data",
            xlabel = "H+L",
            ylabel = "R-G",
        )
        Plots.plot(plt1, plt2, layout = (2, 1))
    end
end

Plots.savefig(joinpath(savedir, "energy_balance_data.png"))

Δz = parent(cds.subsurface.z)[end] - parent(cds.subsurface.z)[end - 2]
first_layer_flux = [
    -parent(sv.saveval[k].soil.κ)[1] * (
        parent(sv.saveval[k].soil.T)[end] -
        parent(sv.saveval[k].soil.T)[end - 2]
    ) / Δz for k in 1:length(sol.t)
]
G_model = [
    (
        parent(sv.saveval[k].soil.turbulent_fluxes.shf)[1] +
        parent(sv.saveval[k].soil.turbulent_fluxes.lhf)[1] -
        parent(sv.saveval[k].soil.R_n)[1]
    ) for k in 1:length(sol.t)
]
canopy_G = [
    (
        parent(sv.saveval[k].canopy.energy.shf)[1] +
        parent(sv.saveval[k].canopy.energy.lhf)[1] -
        parent(sv.saveval[k].canopy.radiative_transfer.LW_n)[1] -
        parent(sv.saveval[k].canopy.radiative_transfer.SW_n)[1]
    ) for k in 1:length(sol.t)
]

G_model_avg = compute_diurnal_avg(G_model, model_times, num_days)
canopy_G_avg = compute_diurnal_avg(canopy_G, model_times, num_days)

plt1 = Plots.plot(size = (1500, 400))
Plots.plot!(
    plt1,
    0.5:0.5:24,
    canopy_G_avg,
    label = "Model: (H+L-Rn)_canopy",
    margins = 10Plots.mm,
    xlabel = "Hour of day",
)
Plots.plot!(
    plt1,
    0.5:0.5:24,
    G_model_avg,
    label = "Model: (H+L-Rn)_soil",
    title = "Ground Heat Flux [W/m^2]",
)
if drivers.LE.status != absent &&
   drivers.H.status != absent &&
   drivers.SW_OUT.status != absent &&
   drivers.LW_OUT.status != absent
    Plots.plot!(
        plt1,
        0.5:0.5:24,
        G_alternate_data_avg,
        label = "Data: (H+L-Rn)_site",
    )
end
Plots.plot!(
    plt1,
    0.5:0.5:24,
    compute_diurnal_avg(first_layer_flux, model_times, num_days),
    label = "Model: -κ∂T∂z|_5cm",
)
Plots.savefig(joinpath(savedir, "ground_heat_flux.png"))

# Water stress factor
β = [parent(sv.saveval[k].canopy.hydraulics.β)[1] for k in 1:length(sol.t)]
plt1 =
    Plots.plot(size = (1500, 400), xlabel = "Day of year", margin = 10Plots.mm)
Plots.plot!(
    plt1,
    daily,
    β,
    label = "Model",
    xlim = [minimum(daily), maximum(daily)],
    title = "Moisture stress factor",
)
Plots.savefig(joinpath(savedir, "moisture_stress.png"))

# Stomatal conductance
g_stomata =
    [parent(sv.saveval[k].canopy.conductance.gs)[1] for k in 1:length(sol.t)]
plt1 =
    Plots.plot(size = (1500, 400), xlabel = "Day of year", margin = 10Plots.mm)
Plots.plot!(
    plt1,
    daily,
    g_stomata,
    label = "Model",
    xlim = [minimum(daily), maximum(daily)],
    title = "Stomatal conductance (mol/m^2/s)",
)
Plots.savefig(joinpath(savedir, "stomatal_conductance.png"))

# Soil water content
# Current resolution has the first layer at 0.1 cm, the second at 5cm.
plt1 = Plots.plot(size = (1500, 800))
Plots.plot!(
    plt1,
    daily,
    [parent(sol.u[k].soil.ϑ_l)[end - 1] for k in 1:1:length(sol.t)],
    label = "5cm",
    xlim = [minimum(daily), maximum(daily)],
    ylim = [0.05, 0.55],
    xlabel = "Days",
    ylabel = "SWC [m/m]",
    color = "blue",
    margin = 10Plots.mm,
)

plot!(
    plt1,
    daily,
    [parent(sol.u[k].soil.θ_i)[end - 1] for k in 1:1:length(sol.t)],
    color = "cyan",
    label = "Ice, 5cm",
)

if drivers.SWC.status != absent
    Plots.plot!(plt1, seconds ./ 3600 ./ 24, drivers.SWC.values, label = "Data")
end

plt2 = Plots.plot(
    seconds ./ 3600 ./ 24,
    drivers.P.values .* (-1e3 * 24 * 3600),
    label = "Data",
    ylabel = "Precipitation [mm/day]",
    xlim = [minimum(daily), maximum(daily)],
    margin = 10Plots.mm,
    ylim = [-200, 0],
    size = (1500, 400),
)
Plots.plot(plt2, plt1, layout = grid(2, 1, heights = [0.2, 0.8]))
Plots.savefig(joinpath(savedir, "soil_water_content.png"))

# Cumulative ET

dt_model = sol.t[2] - sol.t[1]
dt_data = seconds[2] - seconds[1]
# Find which index in the data our simulation starts at:
idx = argmin(abs.(seconds .- sol.t[1]))
if drivers.LE.status != absent
    Plots.plot(
        seconds ./ 24 ./ 3600,
        cumsum(measured_T[:]) * dt_data,
        label = "Data ET",
    )

    Plots.plot!(
        seconds ./ 24 ./ 3600,
        cumsum(drivers.P.values[:]) * dt_data * (1e3 * 24 * 3600),
        label = "Data P",
    )
    Plots.plot!(
        daily,
        cumsum(T .+ E) * dt_model .+ cumsum(measured_T[:])[idx] * dt_data,
        label = "Model ET",
    )

    Plots.plot!(
        ylabel = "∫ Water fluxes dt",
        xlabel = "Days",
        margins = 10Plots.mm,
    )
    Plots.savefig(joinpath(savedir, "cumul_p_et.png"))
end

# Soil Temperature

# The second layer is ~ 5cm, third is at 11cm
soil_T_5 = [parent(sv.saveval[k].soil.T)[end - 1] for k in 1:length(sol.t)]
soil_T_5_avg = compute_diurnal_avg(soil_T_5, model_times, num_days)
soil_T_10 = [parent(sv.saveval[k].soil.T)[end - 2] for k in 1:length(sol.t)]
soil_T_10_avg = compute_diurnal_avg(soil_T_10, model_times, num_days)

TA_avg = compute_diurnal_avg(
    FT.(drivers.TA.values)[Int64(t_spinup ÷ DATA_DT):Int64(tf ÷ DATA_DT)],
    data_times,
    num_days,
)
if drivers.TS.status != absent
    TS_avg = compute_diurnal_avg(
        FT.(drivers.TS.values)[Int64(t_spinup ÷ DATA_DT):Int64(tf ÷ DATA_DT)],
        data_times,
        num_days,
    )
end

plt1 = Plots.plot(size = (1500, 400))
if drivers.TS.status != absent
    Plots.plot!(
        plt1,
        0.5:0.5:24,
        TS_avg,
        label = "Tsoil (data)",
        title = "Temperature",
    )
end
Plots.plot!(plt1, 0.5:0.5:24, TA_avg, label = "Tair (data)")
Plots.plot!(plt1, 0.5:0.5:24, soil_T_5_avg, label = "Tsoil (model; 5cm)")
Plots.plot!(plt1, 0.5:0.5:24, soil_T_10_avg, label = "Tsoil (model; 11cm)")
Plots.plot!(plt1, xlabel = "Hour of day", ylabel = "Average over Simulation")
Plots.plot!(plt1, margins = 10Plots.mm)
Plots.savefig(joinpath(savedir, "soil_temperature.png"))

# Temperatures
soil_T_sfc = [parent(sv.saveval[k].soil.T)[end] for k in 1:length(sol.t)]
soil_T_sfc_avg = compute_diurnal_avg(soil_T_sfc, model_times, num_days)

canopy_T = [
    parent(
        ClimaLand.Canopy.canopy_temperature(
            land.canopy.energy,
            land.canopy,
            sol.u[k],
            sv.saveval[k],
            sol.t[k],
        ),
    )[1] for k in 1:length(sol.t)
]
canopy_T_avg = compute_diurnal_avg(canopy_T, model_times, num_days)

plt1 = Plots.plot(size = (1500, 400))
if drivers.TS.status != absent
    Plots.plot!(
        plt1,
        0.5:0.5:24,
        TS_avg,
        label = "Soil-D",
        title = "Temperature",
    )
end
Plots.plot!(plt1, 0.5:0.5:24, TA_avg, label = "Atmos-D")

Plots.plot!(plt1, 0.5:0.5:24, soil_T_sfc_avg, label = "Soil-M-2.5cm")

Plots.plot!(plt1, 0.5:0.5:24, canopy_T_avg, label = "Canopy-M")
Plots.plot!(plt1, xlabel = "Hour of day", ylabel = "Average over Simulation")
Plots.plot!(plt1, margins = 10Plots.mm)
Plots.savefig(joinpath(savedir, "temperature.png"))

# Run script with comand line argument "save" to save model output to CSV
if length(ARGS) ≥ 1 && ARGS[1] == "save"
    # Formats fields as semicolon seperated strings
    field_to_array = (field) -> join(parent(field), ';')
    # Recursively unpacks a nested NamedTuple of fields into an array of strings
    function unpack(tup, data)
        for entry in tup
            if entry isa NamedTuple
                unpack(entry, data)
            else
                push!(data, field_to_array(entry))
            end
        end
    end
    # Recursively extracts the names of all fields in a nested namedTuple
    function extract_names(nt, names)
        for entry in pairs(nt)
            if entry[2] isa NamedTuple
                extract_names(entry[2], names)
            else
                push!(names, entry[1])
            end
        end
    end
    # Collect unpacked data from each timestep into an array
    timestamps = [[]]
    push!(timestamps[1], "Timestep")
    extract_names(sv.saveval[1], timestamps[1])
    local cnt = 0
    for timestamp in sv.saveval
        cnt = cnt + 1
        save_data = Any[cnt]
        unpack(timestamp, save_data)
        push!(timestamps, save_data)
    end
    # Write all data to a csv file
    writedlm(joinpath(savedir, "model_output.csv"), timestamps, ',')
    @info "Saved model output to $(savedir)model_output.csv"
end

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
