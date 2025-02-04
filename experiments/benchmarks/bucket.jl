# # Global bucket run using temporal map albedo

# The code sets up and runs the bucket for 7 days using albedo read in from a file
# containing temporally-varying data over the globe, and analytic atmospheric and radiative
# forcings.

# This code runs the bucket multiple times and collects statistics for execution time and
# allocations
#
# When run with buildkite on clima, this code also compares with the previous best time
# saved at the bottom of this file
delete!(ENV, "JULIA_CUDA_MEMORY_POOL")

import SciMLBase
using Dates
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaUtilities.TimeVaryingInputs: TimeVaryingInput

import ClimaTimeSteppers as CTS
import NCDatasets
using ClimaCore
import ClimaLand
using ClimaParams
using ClimaLand.Bucket:
    BucketModel, BucketModelParameters, PrescribedSurfaceAlbedo
using ClimaLand:
    initialize,
    make_update_aux,
    make_exp_tendency,
    make_set_initial_cache,
    PrescribedAtmosphere,
    PrescribedRadiativeFluxes

import Profile, ProfileCanvas

const FT = Float64;

context = ClimaComms.context()
ClimaComms.init(context)
device = ClimaComms.device()
device_suffix = device isa ClimaComms.CPUSingleThreaded ? "cpu" : "gpu"

earth_param_set = ClimaLand.Parameters.LandParameters(FT);
outdir = "bucket_benchmark_$(device_suffix)"
!ispath(outdir) && mkpath(outdir)

function setup_prob(t0, tf, Δt; nelements = (101, 7))
    # We set up the problem in a function so that we can make multiple copies (for profiling)

    # Set up simulation domain
    soil_depth = FT(3.5)
    bucket_domain = ClimaLand.Domains.SphericalShell(;
        radius = FT(6.3781e6),
        depth = soil_depth,
        nelements = nelements,
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
    updateat = collect(t0:(3Δt):tf)
    drivers = ClimaLand.get_drivers(model)
    updatefunc = ClimaLand.make_update_drivers(drivers)
    driver_cb = ClimaLand.DriverUpdateCallback(updateat, updatefunc)
    cb = SciMLBase.CallbackSet(driver_cb)

    return prob, cb
end

function setup_simulation(; greet = false)
    t0 = 0.0
    tf = 7 * 86400
    Δt = 3600.0
    nelements = (101, 7)
    if greet
        @info "Run: Bucket with temporal map"
        @info "Resolution: $nelements"
        @info "Timestep: $Δt s"
        @info "Duration: $(tf - t0) s"
    end

    prob, cb = setup_prob(t0, tf, Δt; nelements)
    timestepper = CTS.RK4()
    ode_algo = CTS.ExplicitAlgorithm(timestepper)
    return prob, ode_algo, Δt, cb
end

# Warm up and greet
prob, ode_algo, Δt, cb = setup_simulation(; greet = true)
SciMLBase.solve(prob, ode_algo; dt = Δt, callback = cb)

@info "Starting profiling"
# Stop when we profile for MAX_PROFILING_TIME_SECONDS or MAX_PROFILING_SAMPLES
MAX_PROFILING_TIME_SECONDS = 500
MAX_PROFILING_SAMPLES = 100
time_now = time()
timings_s = Float64[]
ClimaComms.device() isa ClimaComms.CUDADevice && import CUDA
while (time() - time_now) < MAX_PROFILING_TIME_SECONDS &&
    length(timings_s) < MAX_PROFILING_SAMPLES
    lprob, lode_algo, lΔt, lcb = setup_simulation()
    ClimaComms.device() isa ClimaComms.CUDADevice && CUDA.device_synchronize()
    push!(
        timings_s,
        ClimaComms.@elapsed device SciMLBase.solve(
            lprob,
            lode_algo;
            dt = lΔt,
            callback = lcb,
        )
    )
end
num_samples = length(timings_s)
average_timing_s = round(sum(timings_s) / num_samples, sigdigits = 3)
max_timing_s = round(maximum(timings_s), sigdigits = 3)
min_timing_s = round(minimum(timings_s), sigdigits = 3)
std_timing_s = round(
    sqrt(sum(((timings_s .- average_timing_s) .^ 2) / num_samples)),
    sigdigits = 3,
)
# Runs: 4
@info "Num samples: $num_samples"
@info "Average time: $(average_timing_s) s"
@info "Max time: $(max_timing_s) s"
@info "Min time: $(min_timing_s) s"
@info "Standard deviation time: $(std_timing_s) s"
@info "Timings: $timings_s"
@info "Done profiling"

prob, ode_algo, Δt, cb = setup_simulation()
Profile.@profile SciMLBase.solve(prob, ode_algo; dt = Δt, callback = cb)
results = Profile.fetch()
flame_file = joinpath(outdir, "flame_$device_suffix.html")
ProfileCanvas.html_file(flame_file, results)
@info "Saved compute flame to $flame_file"

prob, ode_algo, Δt, cb = setup_simulation()
Profile.Allocs.@profile sample_rate = 0.1 SciMLBase.solve(
    prob,
    ode_algo;
    dt = Δt,
    callback = cb,
)
results = Profile.Allocs.fetch()
profile = ProfileCanvas.view_allocs(results)
alloc_flame_file = joinpath(outdir, "alloc_flame_$device_suffix.html")
ProfileCanvas.html_file(alloc_flame_file, profile)
@info "Saved allocation flame to $alloc_flame_file"

if ClimaComms.device() isa ClimaComms.CUDADevice
    import CUDA
    lprob, lode_algo, lΔt, lcb = setup_simulation()
    p = CUDA.@profile SciMLBase.solve(
        lprob,
        lode_algo;
        dt = lΔt,
        callback = lcb,
    )
    # use "COLUMNS" to set how many horizontal characters to crop:
    # See https://github.com/ronisbr/PrettyTables.jl/issues/11#issuecomment-2145550354
    envs = ("COLUMNS" => 120,)
    withenv(envs...) do
        io = IOContext(
            stdout,
            :crop => :horizontal,
            :limit => true,
            :displaysize => displaysize(),
        )
        show(io, p)
    end
    println()
end

if get(ENV, "BUILDKITE_PIPELINE_SLUG", nothing) == "climaland-benchmark"
    PREVIOUS_BEST_TIME = 0.486
    if average_timing_s > PREVIOUS_BEST_TIME + std_timing_s
        @info "Possible performance regression, previous average time was $(PREVIOUS_BEST_TIME)"
    elseif average_timing_s < PREVIOUS_BEST_TIME - std_timing_s
        @info "Possible significant performance improvement, please update PREVIOUS_BEST_TIME in $(@__DIR__)"
    end
    @testset "Performance" begin
        @test PREVIOUS_BEST_TIME - 2std_timing_s <=
              average_timing_s <=
              PREVIOUS_BEST_TIME + 2std_timing_s
    end
end
