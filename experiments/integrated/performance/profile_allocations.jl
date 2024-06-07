import SciMLBase
import ClimaTimeSteppers as CTS
using ClimaCore
using Dates
using Insolation
import ClimaComms
import ClimaUtilities.TimeVaryingInputs: TimeVaryingInput

using ClimaLand
using ClimaLand.Domains
using ClimaLand.Soil
using ClimaLand.Soil.Biogeochemistry
using ClimaLand.Canopy
using ClimaLand.Canopy.PlantHydraulics
import ClimaLand
import ClimaLand.Parameters as LP
import ClimaParams


PROFILING = false
try
    import Profile, ProfileCanvas
    global PROFILING = true
    @info "ProfileCanvas found, running with profiler"
catch
end

function set_initial_conditions(land, t0)
    Y, p, cds = initialize(land)
    FT = eltype(Y.soil.ϑ_l)
    set_initial_cache! = make_set_initial_cache(land)

    Y.soil.ϑ_l = FT(0.3)
    Y.soil.θ_i = FT(0.0)
    T_0 = FT(297.5)
    ρc_s =
        volumetric_heat_capacity.(
            Y.soil.ϑ_l,
            Y.soil.θ_i,
            Ref(land.soil.parameters),
        )
    Y.soil.ρe_int =
        volumetric_internal_energy.(
            Y.soil.θ_i,
            ρc_s,
            T_0,
            Ref(land.soil.parameters),
        )

    Y.soilco2.C = FT(0.000412) # set to atmospheric co2, mol co2 per mol air

    ψ_stem_0 = FT(-1e5 / 9800)
    ψ_leaf_0 = FT(-2e5 / 9800)
    canopy_params = land.canopy.hydraulics.parameters
    S_l_ini =
        inverse_water_retention_curve.(
            canopy_params.retention_model,
            [ψ_stem_0, ψ_leaf_0],
            canopy_params.ν,
            canopy_params.S_s,
        )

    for i in 1:2
        Y.canopy.hydraulics.ϑ_l.:($i) .=
            augmented_liquid_fraction.(canopy_params.ν, S_l_ini[i])
    end

    Y.canopy.energy.T = FT(297.5)
    set_initial_cache!(p, Y, t0)
    return Y, p
end



context = ClimaComms.context()
device_suffix =
    typeof(ClimaComms.context().device) <: ClimaComms.CPUSingleThreaded ?
    "cpu" : "gpu"
const FT = Float64
climaland_dir = pkgdir(ClimaLand)
savedir = joinpath(climaland_dir, "experiments/integrated/performance")
earth_param_set = LP.LandParameters(FT)

# Set the model domain
dz_bottom = FT(2.0)
dz_top = FT(0.2)
soil_depth = FT(5)
z_sfc = FT(0)
h_stem = FT(9) # m
h_leaf = FT(9.5) # m

h_canopy = h_stem + h_leaf

land_domain = ClimaLand.Domains.SphericalShell(;
    radius = FT(6.3781e6),
    depth = soil_depth,
    nelements = (10, 5),
    npolynomial = 1,
    dz_tuple = FT.((dz_bottom, dz_top)),
);
canopy_domain = ClimaLand.Domains.obtain_surface_domain(land_domain)
sfc_cds = ClimaCore.Fields.coordinate_field(land_domain.space.surface)
# First pick the reference time and start time of the simulation, since time varying input depends on that
t0 = Float64(0)# start at reference time
ref_time = DateTime("202001010000", "yyyymmddHHMM")
# Time varying input
LAIfunction = TimeVaryingInput((t) -> 2.0)
# Atmospheric and radiative forcing
precip = TimeVaryingInput((t) -> -1.0e-7)
atmos_q = TimeVaryingInput((t) -> 0.002)
atmos_T = TimeVaryingInput((t) -> 298.0)
atmos_p = TimeVaryingInput((t) -> 101320)
atmos_u = TimeVaryingInput((t) -> 3.0)
LW_IN = TimeVaryingInput((t) -> 5.67e-8 * 298.0^4)
SW_IN = TimeVaryingInput((t) -> 500.0)
snow_precip = TimeVaryingInput((t) -> 0.0)
atmos_h = FT(32)
# Construct the drivers. For the reference time we will use the UTC time at the
# start of the simulation
atmos = ClimaLand.PrescribedAtmosphere(
    precip,
    snow_precip,
    atmos_T,
    atmos_u,
    atmos_q,
    atmos_p,
    ref_time,
    atmos_h,
    earth_param_set;
)
function zenith_angle(
    t,
    ref_time;
    cd_field = sfc_cds,
    insol_params::Insolation.Parameters.InsolationParameters{FT} = earth_param_set.insol_params,
) where {FT}
    # This should be time in UTC
    current_datetime = ref_time + Dates.Second(round(t))

    # Orbital Data uses Float64, so we need to convert to our sim FT
    d, δ, η_UTC =
        FT.(
            Insolation.helper_instantaneous_zenith_angle(
                current_datetime,
                ref_time,
                insol_params,
            )
        )

    Insolation.instantaneous_zenith_angle.(
        d,
        δ,
        η_UTC,
        sfc_cds.long,
        sfc_cds.lat,
    ).:1
end

radiation = ClimaLand.PrescribedRadiativeFluxes(
    FT,
    SW_IN,
    LW_IN,
    ref_time,
    θs = zenith_angle,
)

# Model parameters
# Soil parameters
soil_ν = FT(0.5) # m3/m3
soil_K_sat = FT(4e-7) # m/
soil_S_s = FT(1e-3) # 1/m
soil_vg_n = FT(2.05) # unitless
soil_vg_α = FT(0.04) # inverse meters
θ_r = FT(0.067)
ν_ss_quartz = FT(0.1)
ν_ss_om = FT(0.1)
ν_ss_gravel = FT(0.0);
# Energy Balance model
ac_canopy = FT(2.5e3)
# Conductance Model
g1 = FT(141)
#Photosynthesis model
Vcmax25 = FT(9e-5)
# Plant Hydraulics and general plant parameters
K_sat_plant = 5e-9
ψ63 = FT(-4 / 0.0098)
Weibull_param = FT(4)
a = FT(0.05 * 0.0098)
conductivity_model =
    PlantHydraulics.Weibull{FT}(K_sat_plant, ψ63, Weibull_param)
retention_model = PlantHydraulics.LinearRetentionCurve{FT}(a)
plant_ν = FT(1.44e-4)
SAI = FT(1.0)
maxLAI = FT(2.0)
f_root_to_shoot = FT(3.5)
RAI = maxLAI * f_root_to_shoot
capacity = plant_ν * maxLAI * h_leaf * FT(1000)
plant_S_s = FT(1e-2 * 0.0098) # m3/m3/MPa to m3/m3/m
z0_m = FT(0.13) * h_canopy
z0_b = FT(0.1) * z0_m

# Set up model
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
)

soil_args = (domain = soil_domain, parameters = soil_ps)
soil_model_type = Soil.EnergyHydrology{FT}

# Soil microbes model
soilco2_type = Soil.Biogeochemistry.SoilCO2Model{FT}

soilco2_ps = SoilCO2ModelParameters(FT; ν = soil_ν)

# soil microbes args
Csom = (z, t) -> eltype(z)(5.0)

# Set the soil CO2 BC to being atmospheric CO2
soilco2_top_bc = Soil.Biogeochemistry.AtmosCO2StateBC()
soilco2_bot_bc = Soil.Biogeochemistry.SoilCO2StateBC((p, t) -> 0.0)
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
energy_args = (parameters = Canopy.BigLeafEnergyParameters{FT}(ac_canopy),)

# Set up autotrophic respiration
autotrophic_respiration_args =
    (; parameters = AutotrophicRespirationParameters(FT))
# Set up radiative transfer
radiative_transfer_args = (; parameters = TwoStreamParameters(FT))
# Set up conductance
conductance_args = (; parameters = MedlynConductanceParameters(FT; g1))
# Set up photosynthesis
photosynthesis_args =
    (; parameters = FarquharParameters(FT, Canopy.C3(); Vcmax25 = Vcmax25))
# Set up plant hydraulics
ai_parameterization = PrescribedSiteAreaIndex{FT}(LAIfunction, SAI, RAI)
function root_distribution(z::T) where {T}
    return T(1.0 / 0.5 * exp(z / 0.5)) # 1/m
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
exp_tendency! = make_exp_tendency(land)
# Set up timestepping and simulation callbacks
dt = Float64(150)
tf = Float64(t0 + 100dt)
saveat = Array(t0:dt:tf)
updateat = Array(t0:dt:tf)
timestepper = CTS.RK4()
ode_algo = CTS.ExplicitAlgorithm(timestepper)

updatefunc = ClimaLand.make_update_drivers(atmos, radiation)
driver_cb = ClimaLand.DriverUpdateCallback(updateat, updatefunc)
# Set initial conditions
Y, p = set_initial_conditions(land, t0)

# Solve simulation
prob = SciMLBase.ODEProblem(
    CTS.ClimaODEFunction(
        T_exp! = exp_tendency!,
        dss! = ClimaLand.dss!,
        T_imp! = nothing,
    ),
    Y,
    (t0, tf),
    p,
)
sol = SciMLBase.solve(
    prob,
    ode_algo;
    dt = dt,
    callback = driver_cb,
    adaptive = false,
)
if PROFILING
    # Now that we compiled, solve again but collect profiling information
    Y, p = set_initial_conditions(land, t0)
    updateat = Array(t0:dt:tf)
    updatefunc = ClimaLand.make_update_drivers(atmos, radiation)
    driver_cb = ClimaLand.DriverUpdateCallback(updateat, updatefunc)
    prob = SciMLBase.ODEProblem(
        CTS.ClimaODEFunction(
            T_exp! = exp_tendency!,
            dss! = ClimaLand.dss!,
            T_imp! = nothing,
        ),
        Y,
        (t0, tf),
        p,
    )
    Profile.@profile SciMLBase.solve(
        prob,
        ode_algo;
        dt = dt,
        callback = driver_cb,
    )
    results = Profile.fetch()
    flame_file = joinpath(savedir, "flame_$(device_suffix).html")
    ProfileCanvas.html_file(flame_file, results)
    @info "Save compute flame to flame_file"
    Y, p = set_initial_conditions(land, t0)
    updateat = Array(t0:dt:tf)
    updatefunc = ClimaLand.make_update_drivers(atmos, radiation)
    driver_cb = ClimaLand.DriverUpdateCallback(updateat, updatefunc)
    prob = SciMLBase.ODEProblem(
        CTS.ClimaODEFunction(
            T_exp! = exp_tendency!,
            dss! = ClimaLand.dss!,
            T_imp! = nothing,
        ),
        Y,
        (t0, tf),
        p,
    )
    Profile.Allocs.@profile sample_rate = 0.01 SciMLBase.solve(
        prob,
        ode_algo;
        dt = dt,
        callback = driver_cb,
    )
    results = Profile.Allocs.fetch()
    profile = ProfileCanvas.view_allocs(results)
    alloc_flame_file = joinpath(savedir, "alloc_flame_$(device_suffix).html")
    ProfileCanvas.html_file(alloc_flame_file, profile)
    @info "Save allocation flame to alloc_flame_file"
end
