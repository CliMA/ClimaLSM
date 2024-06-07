# # Coupling the CliMA Canopy and Soil Hydraulics Models

# In the
# [`previous tutorial`](@ref https://clima.github.io/ClimaLand.jl/dev/generated/soil_plant_hydrology_tutorial/),
# we demonstrated how to run the canopy model in
# standalone mode using prescribed values for the inputs of soil hydraulics
# into the canopy hydraulics model. However, ClimaLand has the built-in capacity
# to couple the canopy model with a soil physics model and timestep the two
# simulations together to model a canopy-soil system. This tutorial
# demonstrates how to setup and run a coupled simulation, again using
# initial conditions, atmospheric and radiative flux conditions, and canopy
# properties observed at the US-MOz flux tower, a flux tower located within an
# oak-hickory forest in Ozark, Missouri, USA. See [Wang et al. 2021]
# (https://doi.org/10.5194/gmd-14-6741-2021) for details on the site and
# parameters.


# In ClimaLand, the coupling of the canopy and soil models is done by
# pairing the inputs and outputs which between the two models so that they match.
# For example, the root extraction of the canopy hydraulics model, which acts as a
# boundary flux for the plant system, is paired with a source term for root extraction in
# the soil model, so that the flux of water from the soil into the roots is
# equal and factored into both models. This pairing is done automatically in the
# constructor for a
# [`SoilCanopyModel`](https://clima.github.io/ClimaLand.jl/dev/APIs/ClimaLand/#LSM-Model-Types-and-methods)
# so that a
# user needs only specify the necessary arguments for each of the component
# models, and the two models will automatically be paired into a coupled
# simulation.

# # Preliminary Setup

# Load External Packages:

import SciMLBase
using Plots
using Statistics
using Dates
using Insolation


# Load CliMA Packages and ClimaLand Modules:

using ClimaCore
import ClimaParams as CP
import ClimaTimeSteppers as CTS
using ClimaLand
using ClimaLand.Domains: Column, obtain_surface_domain
using ClimaLand.Soil
using ClimaLand.Soil.Biogeochemistry
using ClimaLand.Canopy
using ClimaLand.Canopy.PlantHydraulics
import ClimaLand
import ClimaLand.Parameters as LP

# Define the floating point precision desired (64 or 32 bit), and get the
# parameter set holding constants used across CliMA Models:

const FT = Float32;
earth_param_set = LP.LandParameters(FT);

# Setup the domain for the model:

nelements = 10
zmin = FT(-2)
zmax = FT(0)
f_root_to_shoot = FT(3.5)
SAI = FT(0.00242)
maxLAI = FT(4.2)
plant_ν = FT(2.46e-4)
h_stem = FT(9)
h_leaf = FT(9.5)
land_domain = Column(; zlim = (zmin, zmax), nelements = nelements);

# - We will be using prescribed atmospheric and radiative drivers from the
#   US-MOz tower, which we read in here. We are using prescribed
#   atmospheric and radiative flux conditions, but it is also possible to couple
#   the simulation with atmospheric and radiative flux models. We also
# read in the observed LAI and let that vary in time in a prescribed manner.

# Use the data tools for reading FLUXNET data sets 
include(
    joinpath(pkgdir(ClimaLand), "experiments/integrated/fluxnet/data_tools.jl"),
);

# First provide some information about the site
# Timezone (offset from UTC in hrs)
time_offset = 7

# Site latitude and longitude
lat = FT(38.7441) # degree
long = FT(-92.2000) # degree

# Height of the sensor at the site
atmos_h = FT(32) # m

# Provide the site site ID and the path to the data file:
site_ID = "US-MOz"
data_link = "https://caltech.box.com/shared/static/7r0ci9pacsnwyo0o9c25mhhcjhsu6d72.csv"

include(
    joinpath(
        pkgdir(ClimaLand),
        "experiments/integrated/fluxnet/met_drivers_FLUXNET.jl",
    ),
);

# # Setup the Coupled Canopy and Soil Physics Model

# We want to simulate the canopy-soil system together, so the model type
# [`SoilCanopyModel`](https://clima.github.io/ClimaLand.jl/dev/APIs/ClimaLand/#LSM-Model-Types-and-methods)
# is chosen.
# From the linked documentation, we see that we need to provide the soil model
# type and arguments as well as the canopy model component types, component
# arguments, and the canopy model arguments, so we first need to initialize
# all of these.

# For our soil model, we will choose the
# [`EnergyHydrology`](https://clima.github.io/ClimaLand.jl/dev/APIs/Soil/#Soil-Models-2)
# and set up all the necessary arguments. See the
# [tutorial](https://clima.github.io/ClimaLand.jl/dev/generated/Soil/soil_energy_hydrology/)
# on the model for a more detailed explanation of the soil model.

# Define the parameters for the soil model and provide them to the model
# parameters struct:

# Soil parameters
soil_ν = FT(0.5) # m3/m3
soil_K_sat = FT(4e-7) # m/s
soil_S_s = FT(1e-3) # 1/m
soil_vg_n = FT(2.05) # unitless
soil_vg_α = FT(0.04) # inverse meters
θ_r = FT(0.067); # m3/m3

# Soil heat transfer parameters
ν_ss_quartz = FT(0.1)
ν_ss_om = FT(0.1)
ν_ss_gravel = FT(0.0);
z_0m_soil = FT(0.1)
z_0b_soil = FT(0.1)
soil_ϵ = FT(0.98)
soil_α_PAR = FT(0.2)
soil_α_NIR = FT(0.4)

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

# For the heterotrophic respiration model, see the
# [documentation](https://clima.github.io/ClimaLand.jl/previews/PR214/dynamicdocs/pages/soil_biogeochemistry/microbial_respiration/)
# to understand the parameterisation.
# The domain is defined similarly to the soil domain described above.
soilco2_type = Soil.Biogeochemistry.SoilCO2Model{FT}

soilco2_ps = SoilCO2ModelParameters(FT; ν = soil_ν);

# soil microbes args
Csom = (z, t) -> eltype(z)(5); # kg C m⁻³, this is a guess, not measured at the site

soilco2_top_bc = Soil.Biogeochemistry.AtmosCO2StateBC()
soilco2_bot_bc = Soil.Biogeochemistry.SoilCO2StateBC((p, t) -> 0.0);
soilco2_sources = (MicrobeProduction{FT}(),);

soilco2_boundary_conditions = (; top = soilco2_top_bc, bottom = soilco2_bot_bc);

soilco2_drivers = Soil.Biogeochemistry.SoilDrivers(
    Soil.Biogeochemistry.PrognosticMet{FT}(),
    Soil.Biogeochemistry.PrescribedSOC{FT}(Csom),
    atmos,
);

soilco2_args = (;
    boundary_conditions = soilco2_boundary_conditions,
    sources = soilco2_sources,
    domain = soil_domain,
    parameters = soilco2_ps,
    drivers = soilco2_drivers,
);

# Next we need to set up the [`CanopyModel`](https://clima.github.io/ClimaLand.jl/dev/APIs/canopy/Canopy/#Canopy-Model-Structs).
# For more details on the specifics of this model see the previous tutorial.

# Begin by declaring the component types of the canopy model. Unlike in the
# previous tutorial, collect the arguments to each component into tuples and do
# not instantiate the component models yet. The constructor for the
# SoilPlantHydrologyModel will use these arguments and internally instatiate the
# component CanopyModel and RichardsModel instances. This is done so that the
# constructor may enforce consistency constraints between the two models, and
# this must be done internally from the constructor.

canopy_component_types = (;
    autotrophic_respiration = Canopy.AutotrophicRespirationModel{FT},
    radiative_transfer = Canopy.TwoStreamModel{FT},
    photosynthesis = Canopy.FarquharModel{FT},
    conductance = Canopy.MedlynConductanceModel{FT},
    hydraulics = Canopy.BigLeafHydraulicsModel{FT},
);

# Then provide arguments to the canopy radiative transfer, stomatal conductance,
# and photosynthesis models as was done in the previous tutorial.

autotrophic_respiration_args =
    (; parameters = AutotrophicRespirationParameters(FT))

radiative_transfer_args = (;
    parameters = TwoStreamParameters(
        FT;
        G_Function = ConstantGFunction(FT(0.5)),
        α_PAR_leaf = 0.1,
        α_NIR_leaf = 0.45,
        τ_PAR_leaf = 0.05,
        τ_NIR_leaf = 0.25,
        Ω = 0.69,
    )
)

conductance_args = (; parameters = MedlynConductanceParameters(FT; g1 = 141))

photosynthesis_args =
    (; parameters = FarquharParameters(FT, Canopy.C3(); Vcmax25 = FT(5e-5)));

K_sat_plant = FT(1.8e-8)
RAI = (SAI + maxLAI) * f_root_to_shoot;
# Note: LAIfunction was determined from data in the script we included above.
ai_parameterization = PrescribedSiteAreaIndex{FT}(LAIfunction, SAI, RAI)
function root_distribution(z::T; rooting_depth = FT(1.0)) where {T}
    return T(1.0 / rooting_depth) * exp(z / T(rooting_depth)) # 1/m
end

ψ63 = FT(-4 / 0.0098)
Weibull_param = FT(4)
a = FT(0.05 * 0.0098)

conductivity_model =
    PlantHydraulics.Weibull{FT}(K_sat_plant, ψ63, Weibull_param)

retention_model = PlantHydraulics.LinearRetentionCurve{FT}(a)

plant_ν = FT(0.7)
plant_S_s = FT(1e-2 * 0.0098)

plant_hydraulics_ps = PlantHydraulics.PlantHydraulicsParameters(;
    ai_parameterization = ai_parameterization,
    ν = plant_ν,
    S_s = plant_S_s,
    root_distribution = root_distribution,
    conductivity_model = conductivity_model,
    retention_model = retention_model,
)

plant_hydraulics_args =
    (parameters = plant_hydraulics_ps, h_stem = h_stem, h_leaf = h_leaf);

# We may now collect all of the canopy component argument tuples into one
# arguments tuple for the canopy component models.

canopy_component_args = (;
    autotrophic_respiration = autotrophic_respiration_args,
    radiative_transfer = radiative_transfer_args,
    photosynthesis = photosynthesis_args,
    conductance = conductance_args,
    hydraulics = plant_hydraulics_args,
);

# We also need to provide the shared parameter struct to the canopy.
z0_m = FT(2)
z0_b = FT(0.2)

shared_params = SharedCanopyParameters{FT, typeof(earth_param_set)}(
    z0_m,
    z0_b,
    earth_param_set,
)
canopy_domain = obtain_surface_domain(land_domain)
canopy_model_args = (; parameters = shared_params, domain = canopy_domain);

# We may now instantiate the integrated plant and soil model. In this example,
# we will compute transpiration diagnostically, and work with prescribed
# atmospheric and radiative flux conditions from the observations at the Ozark
# site as was done in the previous tutorial.

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
);

# Now we can initialize the state vectors and model coordinates, and initialize
# the explicit/implicit tendencies as usual. The Richard's equation time
# stepping is done implicitly, while the canopy model may be explicitly stepped,
# so we use an IMEX (implicit-explicit) scheme for the combined model.

Y, p, coords = initialize(land);
exp_tendency! = make_exp_tendency(land);

# We need to provide initial conditions for the soil and canopy hydraulics
# models:
Y.soil.ϑ_l = FT(0.4)
Y.soil.θ_i = FT(0.0)
T_0 = FT(288.7)
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

ψ_stem_0 = FT(-1e5 / 9800)
ψ_leaf_0 = FT(-2e5 / 9800)

S_l_ini =
    inverse_water_retention_curve.(
        retention_model,
        [ψ_stem_0, ψ_leaf_0],
        plant_ν,
        plant_S_s,
    )

for i in 1:2
    Y.canopy.hydraulics.ϑ_l.:($i) .=
        augmented_liquid_fraction.(plant_ν, S_l_ini[i])
end;

# Select the timestepper and solvers needed for the specific problem. Specify the time range and dt
# value over which to perform the simulation.

t0 = Float64(150 * 3600 * 24)# start mid year
N_days = 100
tf = t0 + Float64(3600 * 24 * N_days)
dt = Float64(30)
n = 120
saveat = Array(t0:(n * dt):tf)

timestepper = CTS.RK4()
ode_algo = CTS.ExplicitAlgorithm(timestepper);

# Now set the initial values for the cache variables for the combined soil and plant model.

set_initial_cache! = make_set_initial_cache(land)
set_initial_cache!(p, Y, t0);

# Set the callbacks, which govern
# how often we save output, and how often we update
# the forcing data ("drivers")

sv = (;
    t = Array{Float64}(undef, length(saveat)),
    saveval = Array{NamedTuple}(undef, length(saveat)),
)
saving_cb = ClimaLand.NonInterpSavingCallback(sv, saveat)
updatefunc = ClimaLand.make_update_drivers(atmos, radiation)
updateat = Array(t0:1800:tf)
driver_cb = ClimaLand.DriverUpdateCallback(updateat, updatefunc)
cb = SciMLBase.CallbackSet(driver_cb, saving_cb);

# Carry out the simulation
prob = SciMLBase.ODEProblem(
    CTS.ClimaODEFunction(T_exp! = exp_tendency!),
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
);

# # Plotting

# Now that we have both a soil and canopy model incorporated together, we will
# show how to plot some model data demonstrating the time series produced from
# each of these models. As before, we may plot the GPP of the system as well
# as transpiration showing fluxes in the canopy.

daily = sol.t ./ 3600 ./ 24
model_GPP = [
    parent(sv.saveval[k].canopy.photosynthesis.GPP)[1] for
    k in 1:length(sv.saveval)
]

plt1 = Plots.plot(size = (600, 700));
Plots.plot!(
    plt1,
    daily,
    model_GPP .* 1e6,
    label = "Model",
    xlim = [minimum(daily), maximum(daily)],
    xlabel = "days",
    ylabel = "GPP [μmol/mol]",
);

# Transpiration plot:

T = [
    parent(sv.saveval[k].canopy.conductance.transpiration)[1] for
    k in 1:length(sv.saveval)
]
T = T .* (1e3 * 24 * 3600)

plt2 = Plots.plot(size = (500, 700));
Plots.plot!(
    plt2,
    daily,
    T,
    label = "Model",
    xlim = [minimum(daily), maximum(daily)],
    xlabel = "days",
    ylabel = "Vapor Flux [mm/day]",
);

# Show the two plots together:

Plots.plot(plt1, plt2, layout = (2, 1));

# Save the output:
savefig("ozark_canopy_flux_test.png");
# ![](ozark_canopy_flux_test.png)

# Now, we will plot the augmented volumetric liquid water fraction at different
# depths in the soil over the course of the simulation.

plt1 = Plots.plot(size = (500, 700));
ϑ_l_10 = [parent(sol.u[k].soil.ϑ_l)[end] for k in 1:1:length(sol.t)]
plt1 = Plots.plot(
    daily,
    ϑ_l_10,
    label = "10 cm",
    xlabel = "Days",
    ylabel = "SWC [m/m]",
    xlim = [minimum(daily), maximum(daily)],
    size = (500, 700),
    margins = 10Plots.mm,
    color = "blue",
);

plot!(
    plt1,
    daily,
    [parent(sol.u[k].soil.ϑ_l)[end - 1] for k in 1:1:length(sol.t)],
    label = "20cm",
    color = "red",
);

plot!(
    plt1,
    daily,
    [parent(sol.u[k].soil.ϑ_l)[end - 2] for k in 1:1:length(sol.t)],
    label = "30cm",
    color = "purple",
);

# Save the output:
savefig("ozark_soil_test.png");
# ![](ozark_soil_test.png)

# And now to demonstrate the coupling of the soil and canopy models we will plot
# the water fluxes from the soil up into the plant hydraulic system:

root_stem_flux = [
    sum(sv.saveval[k].root_extraction) .* (1e3 * 3600 * 24) for
    k in 1:length(sol.t)
]
plt1 = Plots.plot(
    daily,
    root_stem_flux,
    label = "soil-root-stem water flux",
    ylabel = "Water flux[mm/day]",
    xlim = [minimum(daily), maximum(daily)],
    size = (500, 700),
    margins = 10Plots.mm,
);

# And save the output
savefig("ozark_soil_plant_flux.png");
# ![](ozark_soil_plant_flux.png)
