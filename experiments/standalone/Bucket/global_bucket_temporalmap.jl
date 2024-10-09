# # Global bucket run using temporal map albedo

# The code sets up and runs the bucket for 50 days using albedo read in from
# a file containing temporally-varying data over the globe, and analytic atmospheric
# and radiative forcings.
# This driver is used to verify that the file reading infrastructure used for the
# temporally-varying albedo data is functioning correctly, and that it can run
# on both CPU and GPU, with reasonable computational differences between the results.

# Outputs:
# The final state of the simulation is saved to a CSV file so we can compare
# between CPU and GPU runs.
# Plots of the temporal evolution of water content, snow cover fraction,
# surface temperature, evaporation, and surface energy flux.

import SciMLBase
import ClimaComms
ClimaComms.@import_required_backends
using CairoMakie
using Dates
using DelimitedFiles
using Statistics

import ClimaUtilities.TimeVaryingInputs: TimeVaryingInput

import ClimaTimeSteppers as CTS
import NCDatasets
using ClimaCore
using ClimaCore: Remapping, Geometry
import ClimaParams as CP
import ClimaComms
import ClimaLand
import ClimaLand.Parameters as LP
using ClimaLand.Bucket:
    BucketModel, BucketModelParameters, PrescribedSurfaceAlbedo
using ClimaLand.Domains: coordinates, Column
using ClimaLand:
    initialize,
    make_update_aux,
    make_exp_tendency,
    make_set_initial_cache,
    PrescribedAtmosphere,
    PrescribedRadiativeFluxes

PROFILING = false
try
    import Profile, ProfileCanvas
    global PROFILING = true
    @info "ProfileCanvas found, running with profiler"
catch
end

"""
   compute_extrema(v)

Computes and returns the minimum value in `v` and
the maximum value in `v`, as a tuple, assuming that
`v` is a vector of arrays.
"""
function compute_extrema(v)
    maxes = [maximum(u) for u in v]
    mins = [minimum(u) for u in v]
    return (minimum(mins), maximum(maxes))
end

anim_plots = false
FT = Float64;
context = ClimaComms.context()
earth_param_set = LP.LandParameters(FT);
outdir = joinpath(
    pkgdir(ClimaLand),
    "experiments/standalone/Bucket/artifacts_temporalmap",
)
device_suffix =
    typeof(ClimaComms.context().device) <: ClimaComms.CPUSingleThreaded ?
    "cpu" : "gpu"
!ispath(outdir) && mkpath(outdir)
# Use separate output directory for CPU and GPU runs to avoid race condition
device_suffix =
    typeof(ClimaComms.context().device) <: ClimaComms.CPUSingleThreaded ?
    "cpu" : "gpu"
t0 = 0.0;
# run for 50 days to test monthly file update
tf = 50 * 86400;
Δt = 3600.0;


function setup_prob(t0, tf, Δt)
    # We set up the problem in a function so that we can make multiple copies (for profiling)

    # Set up simulation domain
    soil_depth = FT(3.5)
    bucket_domain = ClimaLand.Domains.SphericalShell(;
        radius = FT(6.3781e6),
        depth = soil_depth,
        nelements = (10, 10), # this failed with (50,10)
        npolynomial = 1,
        dz_tuple = FT.((1.0, 0.05)),
    )
    start_date = DateTime(2005)

    # Initialize parameters
    σS_c = FT(0.2)
    W_f = FT(0.15)
    z_0m = FT(1e-2)
    z_0b = FT(1e-3)
    κ_soil = FT(0.7)
    ρc_soil = FT(2e6)
    τc = FT(3600)

    surface_space = bucket_domain.space.surface
    # Construct albedo parameter object using temporal map
    albedo = PrescribedSurfaceAlbedo{FT}(start_date, surface_space)

    bucket_parameters = BucketModelParameters(FT; albedo, z_0m, z_0b, τc)

    # Precipitation:
    precip = (t) -> 0
    snow_precip = (t) -> -5e-7 * (t < 1 * 86400)
    # Diurnal temperature variations:
    T_atmos = (t) -> 275.0 + 5.0 * sin(2.0 * π * t / 86400 - π / 2)
    # Constant otherwise:
    u_atmos = (t) -> 3.0
    q_atmos = (t) -> 0.001
    h_atmos = FT(2)
    P_atmos = (t) -> 101325

    bucket_atmos = PrescribedAtmosphere(
        TimeVaryingInput(precip),
        TimeVaryingInput(snow_precip),
        TimeVaryingInput(T_atmos),
        TimeVaryingInput(u_atmos),
        TimeVaryingInput(q_atmos),
        TimeVaryingInput(P_atmos),
        start_date,
        h_atmos,
        earth_param_set,
    )

    # Prescribed radiation -- a prescribed downwelling SW diurnal cycle, with a
    # peak at local noon, and a prescribed downwelling LW radiative
    # flux, assuming the air temperature is on average 275 degrees
    # K with a diurnal amplitude of 5 degrees K:
    SW_d = (t) -> max(1361 * sin(2π * t / 86400 - π / 2), 0.0)
    LW_d = (t) -> 5.67e-8 * (275.0 + 5.0 * sin(2.0 * π * t / 86400 - π / 2))^4
    bucket_rad = PrescribedRadiativeFluxes(
        FT,
        TimeVaryingInput(SW_d),
        TimeVaryingInput(LW_d),
        start_date,
    )


    model = BucketModel(
        parameters = bucket_parameters,
        domain = bucket_domain,
        atmosphere = bucket_atmos,
        radiation = bucket_rad,
    )

    Y, p, _coords = initialize(model)

    Y.bucket.T .= FT(270)
    Y.bucket.W .= FT(0.05)
    Y.bucket.Ws .= FT(0.0)
    Y.bucket.σS .= FT(0.08)

    set_initial_cache! = make_set_initial_cache(model)
    set_initial_cache!(p, Y, t0)
    exp_tendency! = make_exp_tendency(model)
    prob = SciMLBase.ODEProblem(
        CTS.ClimaODEFunction((T_exp!) = exp_tendency!, (dss!) = ClimaLand.dss!),
        Y,
        (t0, tf),
        p,
    )
    saveat = collect(t0:(Δt * 3):tf)
    saved_values = (;
        t = Array{Float64}(undef, length(saveat)),
        saveval = Array{NamedTuple}(undef, length(saveat)),
    )
    saving_cb = ClimaLand.NonInterpSavingCallback(saved_values, saveat)
    updateat = copy(saveat)
    drivers = ClimaLand.get_drivers(model)
    updatefunc = ClimaLand.make_update_drivers(drivers)
    driver_cb = ClimaLand.DriverUpdateCallback(updateat, updatefunc)
    cb = SciMLBase.CallbackSet(driver_cb, saving_cb)

    return prob, cb, saveat, saved_values
end

prob, cb, saveat, saved_values = setup_prob(t0, tf, Δt);
timestepper = CTS.RK4()
ode_algo = CTS.ExplicitAlgorithm(timestepper)

sol = ClimaComms.@time ClimaComms.device() SciMLBase.solve(
    prob,
    ode_algo;
    dt = Δt,
    saveat = saveat,
    callback = cb,
)

if PROFILING
    # Now that we compiled, solve again but collect profiling information

    # We run only for one day for profiling
    tf = 86400.0
    prob, cb, saveat, _ = setup_prob(t0, tf, Δt)

    Profile.@profile SciMLBase.solve(
        prob,
        ode_algo;
        dt = Δt,
        saveat = saveat,
        callback = cb,
    )
    results = Profile.fetch()
    flame_file = joinpath(outdir, "flame_$device_suffix.html")
    ProfileCanvas.html_file(flame_file, results)
    @info "Save compute flame to $flame_file"

    prob, cb, saveat, _ = setup_prob(t0, tf, Δt)
    Profile.Allocs.@profile sample_rate = 1.0 SciMLBase.solve(
        prob,
        ode_algo;
        dt = Δt,
        saveat = saveat,
        callback = cb,
    )
    results = Profile.Allocs.fetch()
    profile = ProfileCanvas.view_allocs(results)
    alloc_flame_file = joinpath(outdir, "alloc_flame_$device_suffix.html")
    ProfileCanvas.html_file(alloc_flame_file, profile)
    @info "Save allocation flame to $alloc_flame_file"
end

# Interpolate to grid
space = axes(sol.prob.p.drivers.T)
longpts = range(-180.0, 180.0, 21)
latpts = range(-90.0, 90.0, 21)
hcoords = [Geometry.LatLongPoint(lat, long) for long in longpts, lat in latpts]
remapper = Remapping.Remapper(space, hcoords)

W = [
    Array(Remapping.interpolate(remapper, sol.u[k].bucket.W)) for
    k in 1:length(sol.t)
];
Ws = [
    Array(Remapping.interpolate(remapper, sol.u[k].bucket.Ws)) for
    k in 1:length(sol.t)
];
σS = [
    Array(Remapping.interpolate(remapper, sol.u[k].bucket.σS)) for
    k in 1:length(sol.t)
];
T_sfc = [
    Array(
        Remapping.interpolate(remapper, saved_values.saveval[k].bucket.T_sfc),
    ) for k in 1:length(sol.t)
];
evaporation = [
    Array(
        Remapping.interpolate(
            remapper,
            saved_values.saveval[k].bucket.turbulent_fluxes.vapor_flux,
        ),
    ) for k in 1:length(sol.t)
];
F_sfc = [
    Array(
        Remapping.interpolate(
            remapper,
            -1 .* saved_values.saveval[k].bucket.R_n .+
            saved_values.saveval[k].bucket.turbulent_fluxes.lhf .+
            saved_values.saveval[k].bucket.turbulent_fluxes.shf,
        ),
    ) for k in 1:length(sol.t)
];

# save prognostic state to CSV - for comparison between GPU and CPU output
open(joinpath(outdir, "tf_state_$(device_suffix)_temporalmap.txt"), "w") do io
    writedlm(io, hcat(T_sfc[end][:], W[end][:], Ws[end][:], σS[end][:]), ',')
end;
# animation settings
nframes = length(W)
framerate = 2
fig_ts = Figure(size = (1000, 1000))
for (i, (field_ts, field_name)) in enumerate(
    zip(
        [W, σS, T_sfc, evaporation, F_sfc],
        ["W", "σS", "T_sfc", "evaporation", "F_sfc"],
    ),
)
    if anim_plots
        fig = Figure(size = (1000, 1000))
        ax = Axis(
            fig[1, 1],
            xlabel = "Longitude",
            ylabel = "Latitude",
            title = field_name,
        )
        clims = compute_extrema(field_ts)
        CairoMakie.Colorbar(fig[:, end + 1], colorrange = clims)
        outfile = joinpath(
            outdir,
            string("anim_$(device_suffix)_", field_name, ".mp4"),
        )
        record(fig, outfile, 1:nframes; framerate = framerate) do frame
            CairoMakie.heatmap!(
                longpts,
                latpts,
                field_ts[frame],
                colorrange = clims,
            )
        end
    end
    # Plot the timeseries of the mean value as well.
    xlabel = i == 5 ? "Time (days)" : ""
    ax2 = Axis(
        fig_ts[i, 1],
        xlabel = xlabel,
        ylabel = field_name,
        title = "Global bucket with temporal map albedo",
    )
    CairoMakie.lines!(ax2, sol.t ./ 3600 ./ 24, [mean(x) for x in field_ts])
end
outfile = joinpath(outdir, string("ts_$device_suffix.png"))
CairoMakie.save(outfile, fig_ts)
