using CairoMakie
import ClimaComms
ClimaComms.@import_required_backends
using Statistics
using Dates
import SciMLBase
import ClimaTimeSteppers as CTS
using ClimaCore
using ClimaUtilities.ClimaArtifacts
import Interpolations

import ClimaUtilities.TimeVaryingInputs: TimeVaryingInput
import ClimaUtilities.Regridders: InterpolationsRegridder
import ClimaUtilities.OutputPathGenerator: generate_output_path
import NCDatasets
import ClimaParams as CP
using ClimaComms

using ClimaLand
using ClimaLand.Soil
import ClimaLand
import ClimaLand.Parameters as LP

regridder_type = :InterpolationsRegridder
context = ClimaComms.context()
ClimaComms.init(context)
device_suffix =
    typeof(context.device) <: ClimaComms.CPUSingleThreaded ? "cpu" : "gpu"
outdir = generate_output_path(
    joinpath("experiments", "standalone", "Soil", "artifacts", device_suffix),
)
!ispath(outdir) && mkpath(outdir)
FT = Float64
dz_tuple = (2.0, 0.1)
nelements = (101, 15)
domain = ClimaLand.global_domain(
    FT;
    nelements = (101, 15),
    dz_tuple = dz_tuple,
    npolynomial = 1,
)

surface_space = domain.space.surface
subsurface_space = domain.space.subsurface
spatially_varying_soil_params =
    ClimaLand.default_spatially_varying_soil_parameters(
        subsurface_space,
        surface_space,
        FT,
    )
(; ν, hydrology_cm, K_sat, S_s, θ_r, f_max, mask) =
    spatially_varying_soil_params

f_over = FT(3.28) # 1/m
R_sb = FT(1.484e-4 / 1000) # m/s
runoff_model = ClimaLand.Soil.Runoff.TOPMODELRunoff{FT}(;
    f_over = f_over,
    f_max = f_max,
    R_sb = R_sb,
)
soil_params = ClimaLand.Soil.RichardsParameters(;
    hydrology_cm = hydrology_cm,
    ν = ν,
    K_sat = K_sat,
    S_s = S_s,
    θ_r = θ_r,
)

era5_artifact_path =
    ClimaLand.Artifacts.era5_land_forcing_data2008_folder_path(; context)

# Below, the preprocess_func argument is used to
# 1. Convert precipitation to be negative (as it is downwards)
# 2. Convert mass flux to equivalent liquid water flux
start_date = DateTime(2008);
# Precipitation:
precip = TimeVaryingInput(
    joinpath(era5_artifact_path, "era5_2008_1.0x1.0.nc"),
    "mtpr",
    surface_space;
    start_date,
    regridder_type,
    file_reader_kwargs = (; preprocess_func = (data) -> -data / 1000,),
)
atmos = ClimaLand.PrescribedPrecipitation{FT, typeof(precip)}(precip)
bottom_bc = ClimaLand.Soil.WaterFluxBC((p, t) -> 0.0)
bc = (;
    top = ClimaLand.Soil.RichardsAtmosDrivenFluxBC(atmos, runoff_model),
    bottom = bottom_bc,
)
model = ClimaLand.Soil.RichardsModel{FT}(;
    parameters = soil_params,
    domain = domain,
    boundary_conditions = bc,
    sources = (),
    lateral_flow = false,
)
Y, p, t = initialize(model)
z = ClimaCore.Fields.coordinate_field(domain.space.subsurface).z
lat = ClimaCore.Fields.coordinate_field(domain.space.subsurface).lat
function hydrostatic_profile(
    lat::FT,
    z::FT,
    ν::FT,
    θ_r::FT,
    α::FT,
    n::FT,
    S_s::FT,
    fmax,
) where {FT}
    m = 1 - 1 / n
    zmin = FT(-50.0)
    zmax = FT(0.0)

    z_∇ = FT(zmin / 5.0 + (zmax - zmin) / 2.5 * (fmax - 0.35) / 0.7)
    if z > z_∇
        S = FT((FT(1) + (α * (z - z_∇))^n)^(-m))
        ϑ_l = S * (ν - θ_r) + θ_r
    else
        ϑ_l = -S_s * (z - z_∇) + ν
    end
    return FT(ϑ_l)
end
t0 = 0.0
tf = 3600.0 * 24 * 2
dt = 1800.0
vg_α = hydrology_cm.α
vg_n = hydrology_cm.n
Y.soil.ϑ_l .= hydrostatic_profile.(lat, z, ν, θ_r, vg_α, vg_n, S_s, f_max)
set_initial_cache! = make_set_initial_cache(model)
exp_tendency! = make_exp_tendency(model);
imp_tendency! = ClimaLand.make_imp_tendency(model);
jacobian! = ClimaLand.make_jacobian(model);

set_initial_cache!(p, Y, t0)
stepper = CTS.ARS111()
ode_algo = CTS.IMEXAlgorithm(
    stepper,
    CTS.NewtonsMethod(
        max_iters = 2,
        update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
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
save_every = 100
saveat = [t0, tf]
sv = (;
    t = Array{Float64}(undef, length(saveat)),
    saveval = Array{NamedTuple}(undef, length(saveat)),
)
saving_cb = ClimaLand.NonInterpSavingCallback(sv, saveat)
updateat = Array(t0:dt:tf)
drivers = ClimaLand.get_drivers(model)
updatefunc = ClimaLand.make_update_drivers(drivers)
driver_cb = ClimaLand.DriverUpdateCallback(updateat, updatefunc)
cb = SciMLBase.CallbackSet(driver_cb, saving_cb)
sol = @time SciMLBase.solve(
    prob,
    ode_algo;
    dt = dt,
    saveat = saveat,
    callback = cb,
)

# Make plots on CPU
oceans_to_zero(x, mask) = mask == 1 ? x : eltype(x)(0)
if context.device isa ClimaComms.CPUSingleThreaded
    longpts = range(-180.0, 180.0, 101)
    latpts = range(-90.0, 90.0, 101)
    hcoords = [
        ClimaCore.Geometry.LatLongPoint(lat, long) for long in longpts,
        lat in latpts
    ]
    remapper = ClimaCore.Remapping.Remapper(surface_space, hcoords)

    h∇_end = ClimaCore.Remapping.interpolate(
        remapper,
        oceans_to_zero.(sv.saveval[end].soil.h∇, mask),
    )
    fig = Figure(size = (600, 400))
    ax = Axis(
        fig[1, 1],
        xlabel = "Longitude",
        ylabel = "Latitude",
        title = "Water table thickness",
    )
    clims = extrema(h∇_end)
    CairoMakie.heatmap!(ax, longpts, latpts, h∇_end, colorrange = clims)
    Colorbar(fig[:, end + 1], colorrange = clims)
    outfile = joinpath(outdir, string("heatmap_h∇.png"))
    CairoMakie.save(outfile, fig)

    R_s_end = ClimaCore.Remapping.interpolate(
        remapper,
        oceans_to_zero.(sv.saveval[end].soil.R_s, mask),
    )
    fig = Figure(size = (600, 400))
    ax = Axis(
        fig[1, 1],
        xlabel = "Longitude",
        ylabel = "Latitude",
        title = "Surface Runoff",
    )
    clims = extrema(R_s_end)
    CairoMakie.heatmap!(ax, longpts, latpts, R_s_end, colorrange = clims)
    Colorbar(fig[:, end + 1], colorrange = clims)
    outfile = joinpath(outdir, string("heatmap_R_s.png"))
    CairoMakie.save(outfile, fig)

    R_ss_end = ClimaCore.Remapping.interpolate(
        remapper,
        oceans_to_zero.(sv.saveval[end].soil.R_ss, mask),
    )

    fig = Figure(size = (600, 400))
    ax = Axis(
        fig[1, 1],
        xlabel = "Longitude",
        ylabel = "Latitude",
        title = "Subsurface Runoff",
    )
    clims = extrema(R_ss_end)
    CairoMakie.heatmap!(ax, longpts, latpts, R_ss_end, colorrange = clims)
    Colorbar(fig[:, end + 1], colorrange = clims)
    outfile = joinpath(outdir, string("heatmap_R_ss.png"))
    CairoMakie.save(outfile, fig)
    field_to_error(field) =
        field < 1 & ~isnan(field) ? field : eltype(field)(-0.1)

    θ_sfc_end = ClimaCore.Remapping.interpolate(
        remapper,
        ClimaLand.Domains.top_center_to_surface(
            oceans_to_zero.(field_to_error.(sol.u[end].soil.ϑ_l), mask),
        ),
    )

    fig = Figure(size = (1000, 400))
    ax = Axis(
        fig[1, 1],
        xlabel = "Longitude",
        ylabel = "Latitude",
        title = "θ_sfc",
    )
    clims1 = extrema(θ_sfc_end)
    CairoMakie.heatmap!(ax, longpts, latpts, θ_sfc_end, colorrange = clims1)
    Colorbar(fig[1, 2], colorrange = clims1)

    Δθ_sfc = ClimaCore.Remapping.interpolate(
        remapper,
        ClimaLand.Domains.top_center_to_surface(
            oceans_to_zero.(
                field_to_error.(sol.u[end].soil.ϑ_l .- sol.u[1].soil.ϑ_l),
                mask,
            ),
        ),
    )
    ax2 = Axis(fig[1, 3], xlabel = "Longitude", title = "θ_sfc Δ")
    clims2 = extrema(Δθ_sfc)
    CairoMakie.heatmap!(ax2, longpts, latpts, Δθ_sfc, colorrange = clims2)
    Colorbar(fig[1, 4], colorrange = clims2)
    outfile = joinpath(outdir, string("heatmap_θ_sfc.png"))
    CairoMakie.save(outfile, fig)


    int_ϑ_end = similar(sv.saveval[1].soil.h∇)
    ClimaCore.Operators.column_integral_definite!(
        int_ϑ_end,
        oceans_to_zero.(field_to_error.(sol.u[end].soil.ϑ_l), mask),
    )
    normalized_int_ϑ_end =
        ClimaCore.Remapping.interpolate(remapper, int_ϑ_end ./ 50.0)
    fig = Figure(size = (1000, 400))
    ax = Axis(
        fig[1, 1],
        xlabel = "Longitude",
        ylabel = "Latitude",
        title = "∫θ dz/Δz",
    )
    clims1 = extrema(normalized_int_ϑ_end)
    CairoMakie.heatmap!(
        ax,
        longpts,
        latpts,
        normalized_int_ϑ_end,
        colorrange = clims1,
    )
    Colorbar(fig[1, 2], colorrange = clims1)

    int_ϑ_1 = similar(sv.saveval[1].soil.h∇)
    ClimaCore.Operators.column_integral_definite!(
        int_ϑ_1,
        oceans_to_zero.(field_to_error.(sol.u[1].soil.ϑ_l), mask),
    )
    Δ_normalized_int_ϑ = ClimaCore.Remapping.interpolate(
        remapper,
        (int_ϑ_end .- int_ϑ_1) ./ 50.0,
    )
    ax2 = Axis(fig[1, 3], xlabel = "Longitude", title = "Δ∫θ dz /Δz")
    clims2 = extrema(Δ_normalized_int_ϑ)
    CairoMakie.heatmap!(
        ax2,
        longpts,
        latpts,
        Δ_normalized_int_ϑ,
        colorrange = clims2,
    )
    Colorbar(fig[1, 4], colorrange = clims2)
    outfile = joinpath(outdir, string("heatmap_∫ϑdz.png"))
    CairoMakie.save(outfile, fig)
end
