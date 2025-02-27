import SciMLBase
import ClimaComms
ClimaComms.@import_required_backends
import ClimaTimeSteppers as CTS
using ClimaCore
using ClimaUtilities.ClimaArtifacts
import ClimaUtilities.TimeVaryingInputs:
    TimeVaryingInput, LinearInterpolation, PeriodicCalendar
import ClimaUtilities.OutputPathGenerator: generate_output_path
import ClimaUtilities.ClimaArtifacts: @clima_artifact
import ClimaParams as CP

using ClimaLand
using ClimaLand.Soil
using ClimaLand.Canopy
import ClimaLand
import ClimaLand.Parameters as LP

import CairoMakie
import GeoMakie
using Statistics
using Dates
import NCDatasets

import ClimaDiagnostics
import ClimaAnalysis
import ClimaAnalysis.Visualize as viz
import ClimaUtilities
time_interpolation_method = LinearInterpolation(PeriodicCalendar())
context = ClimaComms.context()
ClimaComms.init(context)
outdir = generate_output_path("experiments/integrated/global")

device_suffix =
    typeof(context.device) <: ClimaComms.CPUSingleThreaded ? "cpu" : "gpu"

FT = Float64
earth_param_set = LP.LandParameters(FT)
nelements = (50, 10)
domain = ClimaLand.global_domain(FT; nelements = nelements)
surface_space = domain.space.surface
subsurface_space = domain.space.subsurface

start_date = DateTime(2008);

# Forcing data
era5_artifact_path =
    ClimaLand.Artifacts.era5_land_forcing_data2008_folder_path(; context)
era5_ncdata_path = joinpath(era5_artifact_path, "era5_2008_1.0x1.0.nc")
atmos, radiation = ClimaLand.prescribed_forcing_era5(
    era5_ncdata_path,
    surface_space,
    start_date,
    earth_param_set,
    FT;
    time_interpolation_method = time_interpolation_method,
)

include(
    joinpath(
        pkgdir(ClimaLand),
        "experiments/integrated/global/global_parameters.jl",
    ),
)
soil_args = (domain = domain, parameters = soil_params)
soil_model_type = Soil.EnergyHydrology{FT}

# Soil microbes model
soilco2_type = Soil.Biogeochemistry.SoilCO2Model{FT}

# soil microbes args
Csom = ClimaLand.PrescribedSoilOrganicCarbon{FT}(TimeVaryingInput((t) -> 5))

# Set the soil CO2 BC to being atmospheric CO2
soilco2_top_bc = Soil.Biogeochemistry.AtmosCO2StateBC()
soilco2_bot_bc = Soil.Biogeochemistry.SoilCO2FluxBC((p, t) -> 0.0) # no flux
soilco2_sources = (Soil.Biogeochemistry.MicrobeProduction{FT}(),)

soilco2_boundary_conditions = (; top = soilco2_top_bc, bottom = soilco2_bot_bc)

soilco2_args = (;
    boundary_conditions = soilco2_boundary_conditions,
    sources = soilco2_sources,
    domain = domain,
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
    (; parameters = Canopy.AutotrophicRespirationParameters(FT))
# Set up radiative transfer
radiative_transfer_args = (;
    parameters = Canopy.TwoStreamParameters(
        FT;
        Ω,
        α_PAR_leaf,
        τ_PAR_leaf,
        α_NIR_leaf,
        τ_NIR_leaf,
    )
)
# Set up conductance
conductance_args = (; parameters = Canopy.MedlynConductanceParameters(FT; g1))
# Set up photosynthesis
photosynthesis_args =
    (; parameters = Canopy.FarquharParameters(FT, is_c3; Vcmax25 = Vcmax25))
# Set up plant hydraulics
modis_lai_artifact_path =
    ClimaLand.Artifacts.modis_lai_forcing_data2008_path(; context)
modis_lai_ncdata_path =
    joinpath(modis_lai_artifact_path, "Yuan_et_al_2008_1x1.nc")
LAIfunction = ClimaLand.prescribed_lai_modis(
    modis_lai_ncdata_path,
    surface_space,
    start_date;
    time_interpolation_method = time_interpolation_method,
)
ai_parameterization = Canopy.PrescribedSiteAreaIndex{FT}(LAIfunction, SAI, RAI)

plant_hydraulics_ps = Canopy.PlantHydraulics.PlantHydraulicsParameters(;
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
shared_params = Canopy.SharedCanopyParameters{FT, typeof(earth_param_set)}(
    z0_m,
    z0_b,
    earth_param_set,
)

canopy_model_args = (;
    parameters = shared_params,
    domain = ClimaLand.obtain_surface_domain(domain),
)

# Integrated plant hydraulics and soil model
land_input = (
    atmos = atmos,
    radiation = radiation,
    runoff = runoff_model,
    soil_organic_carbon = Csom,
)
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

t0 = 0.0
dt = 450.0
tf = 3600

init_soil(ν, θ_r) = θ_r + (ν - θ_r) / 2
Y.soil.ϑ_l .= init_soil.(ν, θ_r)
Y.soil.θ_i .= FT(0.0)
T = FT(276.85)
ρc_s =
    Soil.volumetric_heat_capacity.(
        Y.soil.ϑ_l,
        Y.soil.θ_i,
        soil_params.ρc_ds,
        soil_params.earth_param_set,
    )
Y.soil.ρe_int .=
    Soil.volumetric_internal_energy.(
        Y.soil.θ_i,
        ρc_s,
        T,
        soil_params.earth_param_set,
    )
Y.soilco2.C .= FT(0.000412) # set to atmospheric co2, mol co2 per mol air
Y.canopy.hydraulics.ϑ_l.:1 .= plant_ν
evaluate!(Y.canopy.energy.T, atmos.T, t0)

set_initial_cache! = make_set_initial_cache(land)
exp_tendency! = make_exp_tendency(land);
imp_tendency! = ClimaLand.make_imp_tendency(land);
jacobian! = ClimaLand.make_jacobian(land);
set_initial_cache!(p, Y, t0)
stepper = CTS.ARS343()
ode_algo = CTS.IMEXAlgorithm(
    stepper,
    CTS.NewtonsMethod(
        max_iters = 1,
        update_j = CTS.UpdateEvery(CTS.NewTimeStep),
    ),
)

# set up jacobian info
jac_kwargs =
    (; jac_prototype = ClimaLand.FieldMatrixWithSolver(Y), Wfact = jacobian!)

prob = SciMLBase.ODEProblem(
    CTS.ClimaODEFunction(
        T_exp! = exp_tendency!,
        T_imp! = SciMLBase.ODEFunction(imp_tendency!; jac_kwargs...),
        dss! = ClimaLand.dss!,
    ),
    Y,
    (t0, tf),
    p,
)

# ClimaDiagnostics
nc_writer =
    ClimaDiagnostics.Writers.NetCDFWriter(subsurface_space, outdir; start_date)

diags = ClimaLand.default_diagnostics(
    land,
    start_date;
    output_writer = nc_writer,
    average_period = :hourly,
)

diagnostic_handler =
    ClimaDiagnostics.DiagnosticsHandler(diags, Y, p, t0; dt = dt)

diag_cb = ClimaDiagnostics.DiagnosticsCallback(diagnostic_handler)

updateat = Array(t0:(3600 * 3):tf)
drivers = ClimaLand.get_drivers(land)
updatefunc = ClimaLand.make_update_drivers(drivers)
driver_cb = ClimaLand.DriverUpdateCallback(updateat, updatefunc)

sol = ClimaComms.@time ClimaComms.device() SciMLBase.solve(
    prob,
    ode_algo;
    dt = dt,
    callback = SciMLBase.CallbackSet(driver_cb, diag_cb),
)

# ClimaAnalysis
simdir = ClimaAnalysis.SimDir(outdir)

for short_name in ClimaAnalysis.available_vars(simdir)
    var = get(simdir; short_name)
    times = var.dims["time"]
    for t in times
        fig = CairoMakie.Figure(size = (800, 600))
        kwargs = ClimaAnalysis.has_altitude(var) ? Dict(:z => 1) : Dict()
        viz.heatmap2D_on_globe!(
            fig,
            ClimaAnalysis.slice(var, time = t; kwargs...),
        )
        CairoMakie.save(joinpath(outdir, "$short_name.png"), fig)
    end
end
