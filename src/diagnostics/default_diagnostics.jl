export default_diagnostics

# This file is included by Diagnostics.jl and defines all the defaults for
# various models (e.g., Bucket, SoilCanopyModel). A model here is either a
# standalone (e.g., Bucket) or integrated (e.g., SoilCanopy) model.
#
# If you are developing new models, add your defaults here. If you want to add
# more high level interfaces, add them here. Feel free to include extra files.

# Bucket model

"""
    function common_diagnostics(
                                period,
                                reduction,
                                output_writer,
                                start_date,
                                short_names...;
                                pre_output_hook! = nothing,
                               )

Helper function to define functions like `daily_max`.
"""
function common_diagnostics(
    period,
    reduction,
    output_writer,
    start_date,
    short_names...;
    pre_output_hook! = nothing,
)
    return vcat(
        map(short_names) do short_name
            output_schedule_func =
                period isa Period ?
                EveryCalendarDtSchedule(period; start_date) :
                EveryDtSchedule(period)
            return ScheduledDiagnostic(
                variable = get_diagnostic_variable(short_name),
                compute_schedule_func = EveryStepSchedule(),
                output_schedule_func = output_schedule_func,
                reduction_time_func = reduction,
                output_writer = output_writer,
                pre_output_hook! = pre_output_hook!,
            )
        end...,
    )
end

include("standard_diagnostic_frequencies.jl")

# Bucket
function default_diagnostics(
    land_model::BucketModel{FT},
    start_date;
    output_writer,
    average_period = :daily,
) where {FT}

    define_diagnostics!(land_model)

    bucket_diagnostics = [
        "swa",
        "rn",
        "tsfc",
        "qsfc",
        "lhf",
        "rae",
        "shf",
        "vflux",
        "rhosfc",
        "tsoil",
        "wsoil",
        "wsfc",
        "ssfc",
    ]

    if average_period == :hourly
        default_outputs = hourly_averages(
            FT,
            bucket_diagnostics...;
            output_writer,
            start_date,
        )
    elseif average_period == :daily
        default_outputs =
            daily_averages(FT, bucket_diagnostics...; output_writer, start_date)
    elseif average_period == :monthly
        default_outputs = monthly_averages(
            FT,
            bucket_diagnostics...;
            output_writer,
            start_date,
        )
    end

    return [default_outputs...]
end

# SoilCanopyModel
function default_diagnostics(
    land_model::SoilCanopyModel{FT},
    start_date;
    output_writer,
    output_vars = :long,
    average_period = :daily,
) where {FT}

    define_diagnostics!(land_model)

    if output_vars == :long
        soilcanopy_diagnostics = [
            "swa",
            "sif",
            "ra",
            "gs",
            "trans",
            "clhf",
            "cshf",
            "lwp",
            # "fa", # return a Tuple
            "far",
            "lai",
            "msf",
            "rai",
            "sai",
            "gpp",
            "an",
            "rd",
            "vcmax25",
            "nir",
            "anir",
            "rnir",
            "tnir",
            "par",
            "apar",
            "rpar",
            "tpar",
            "lwn",
            "swn",
            "soc",
            "airp",
            "rain",
            "lwd",
            "swd",
            "snow",
            "qsfc",
            "ws",
            "infil",
            "shc",
            "stc",
            "swp",
            "soilrn",
            "tsoil",
            "soillhf",
            "soilshf",
            "hr",
            "scd",
            "scms",
            "ct",
            "sco2",
            "swc",
            # "pwc", # return a Tuple
            "si",
            "sie",
            "swu",
            "lwu",
            "er",
            "et",
            "sr",
            "rn",
            "lhf",
            "shf",
            "ghf",
            "salb",
        ]
    elseif output_vars == :short
        soilcanopy_diagnostics = [
            "gpp",
            "ct",
            "lai",
            "sco2",
            "swc",
            "si",
            "swa",
            "lwu",
            "et",
            "er",
            "sr",
            "sif",
        ]
    end

    if average_period == :hourly
        default_outputs = hourly_averages(
            FT,
            soilcanopy_diagnostics...;
            output_writer,
            start_date,
        )
    elseif average_period == :daily
        default_outputs = daily_averages(
            FT,
            soilcanopy_diagnostics...;
            output_writer,
            start_date,
        )
    elseif average_period == :monthly
        default_outputs = monthly_averages(
            FT,
            soilcanopy_diagnostics...;
            output_writer,
            start_date,
        )
    end

    return [default_outputs...]
end


# SoilModel
function default_diagnostics(
    land_model::EnergyHydrology{FT},
    start_date;
    output_writer,
    average_period = :daily,
) where {FT}

    define_diagnostics!(land_model)

    soil_diagnostics = ["swc", "si", "sie", "et"]

    if average_period == :hourly
        default_outputs =
            hourly_averages(FT, soil_diagnostics...; output_writer, start_date)
    elseif average_period == :daily
        default_outputs =
            daily_averages(FT, soil_diagnostics...; output_writer, start_date)
    elseif average_period == :monthly
        default_outputs =
            monthly_averages(FT, soil_diagnostics...; output_writer, start_date)
    end

    return [default_outputs...]
end

# Land Model
function default_diagnostics(
    land_model::LandModel{FT},
    start_date;
    output_writer,
    output_vars = :long,
    average_period = :daily,
) where {FT}

    define_diagnostics!(land_model)

    if output_vars == :long
        snowyland_diagnostics = [
            "swa",
            "sif",
            "ra",
            "gs",
            "trans",
            "clhf",
            "cshf",
            "lwp",
            # "fa", # return a Tuple
            "far",
            "lai",
            "msf",
            "rai",
            "sai",
            "gpp",
            "an",
            "rd",
            "vcmax25",
            "nir",
            "anir",
            "rnir",
            "tnir",
            "par",
            "apar",
            "rpar",
            "tpar",
            "lwn",
            "swn",
            "soc",
            "airp",
            "rain",
            "lwd",
            "swd",
            "snow",
            "qsfc",
            "ws",
            "infil",
            "shc",
            "stc",
            "swp",
            "soilrn",
            "tsoil",
            "soillhf",
            "soilshf",
            "hr",
            "scd",
            "scms",
            "ct",
            "sco2",
            "swc",
            # "pwc", # return a Tuple
            "si",
            "sie",
            "swu",
            "lwu",
            "er",
            "et",
            "sr",
            "swe",
            "snd",
            "rn",
            "lhf",
            "shf",
            "ghf",
            "iwc",
        ]
    elseif output_vars == :short
        snowyland_diagnostics = [
            "gpp",
            "ct",
            "swc",
            "si",
            "sie",
            "swu",
            "lwu",
            "et",
            "sr",
            "ssr",
            "swe",
            "shf",
            "lhf",
            "trans",
            "msf",
            "lwp",
            "iwc",
            "swd",
            "lwd",
        ]
    end

    if average_period == :hourly
        default_outputs = hourly_averages(
            FT,
            snowyland_diagnostics...;
            output_writer,
            start_date,
        )
    elseif average_period == :daily
        default_outputs = daily_averages(
            FT,
            snowyland_diagnostics...;
            output_writer,
            start_date,
        )
    elseif average_period == :monthly
        default_outputs = monthly_averages(
            FT,
            snowyland_diagnostics...;
            output_writer,
            start_date,
        )
    end
end
