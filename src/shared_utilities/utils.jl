import ClimaCore
import SciMLBase
import ClimaDiagnostics.Schedules: EveryCalendarDtSchedule
import Dates

export FTfromY, count_nans_state

"""
     heaviside(x::FT)::FT where {FT}

Computes the heaviside function H(x) = 1 (x ≥0), 0 (x < 0)
"""
function heaviside(x::FT)::FT where {FT}
    heaviside(x, FT(0))
end

"""
     heaviside(x::FT, a::FT)::FT where {FT}

Computes the heaviside function H(y) = 1 (y ≥0), 0 (y < 0),
where y = x-a.
"""
function heaviside(x::FT, a::FT)::FT where {FT}
    if x - a > eps(FT)
        return FT(1.0)
    else
        return FT(0.0)
    end
end

"""
    add_dss_buffer_to_aux(p::NamedTuple, domain::Domains.AbstractDomain)

Fallback method for `add_dss_buffer_to_aux` which does not add a dss buffer.
"""
add_dss_buffer_to_aux(p::NamedTuple, domain::Domains.AbstractDomain) = p

"""
    add_dss_buffer_to_aux(
        p::NamedTuple,
        domain::Union{Domains.Plane, Domains.SphericalSurface},
    )

Adds a dss buffer corresponding to `domain.space` to `p` with the name `dss_buffer_2d`,
appropriate for a 2D domain.

This buffer is added so that we preallocate memory for the dss step and do not allocate it
at every timestep. We use a name which specifically denotes that
the buffer is on a 2d space. This is because some models
require both a buffer on the 3d space as well as on the surface
2d space, e.g. in the case when they have prognostic variables that are only
defined on the surface space.
"""
function add_dss_buffer_to_aux(
    p::NamedTuple,
    domain::Union{Domains.Plane, Domains.SphericalSurface},
)
    buffer = ClimaCore.Spaces.create_dss_buffer(
        ClimaCore.Fields.zeros(domain.space.surface),
    )
    return merge(p, (; dss_buffer_2d = buffer))
end

"""
    add_dss_buffer_to_aux(
        p::NamedTuple,
        domain::Union{Domains.HybridBox, Domains.SphericalShell},
    )

Adds a 2d and 3d dss buffer corresponding to `domain.space` to `p` with the names
`dss_buffer_3d`, and `dss_buffer_2d`.

This buffer is added so that we preallocate memory for the dss step and do not allocate it
at every timestep. We use a name which specifically denotes that
the buffer is on a 3d space. This is because some models
require both a buffer on the 3d space as well as on the surface
2d space, e.g. in the case when they have prognostic variables that are only
defined on the surface space.
"""
function add_dss_buffer_to_aux(
    p::NamedTuple,
    domain::Union{Domains.HybridBox, Domains.SphericalShell},
)
    buffer_2d = ClimaCore.Spaces.create_dss_buffer(
        ClimaCore.Fields.zeros(domain.space.surface),
    )
    buffer_3d = ClimaCore.Spaces.create_dss_buffer(
        ClimaCore.Fields.zeros(domain.space.subsurface),
    )
    return merge(p, (; dss_buffer_3d = buffer_3d, dss_buffer_2d = buffer_2d))
end


"""
      dss!(Y::ClimaCore.Fields.FieldVector, p::NamedTuple, t)

Computes the weighted direct stiffness summation and updates `Y` in place.
In the case of a column domain, no dss operations are performed.
"""
function dss!(Y::ClimaCore.Fields.FieldVector, p::NamedTuple, t)
    #for key in propertynames(Y)
    #    property = getproperty(Y, key)
    #    dss_helper!(property, axes(property), p)
    #end
end

"""
    dss_helper!(field_vec::ClimaCore.Fields.FieldVector, space, p::NamedTuple)

Method of `dss_helper!` which unpacks properties of Y when on a
domain that is 2-dimensional in the horizontal.

The assumption is that Y contains FieldVectors which themselves contain either
FieldVectors or Fields, and that the final unpacked variable is a Field.
This method is invoked when the current property itself contains additional
property(ies).
"""
function dss_helper!(
    field_vec::ClimaCore.Fields.FieldVector,
    space,
    p::NamedTuple,
)
    for key in propertynames(field_vec)
        property = getproperty(field_vec, key)
        dss_helper!(property, axes(property), p)
    end
end

"""
    dss_helper!(
        field::ClimaCore.Fields.Field,
        space::ClimaCore.Spaces.ExtrudedFiniteDifferenceSpace,
        p::NamedTuple)

Method of `dss_helper!` which performs dss on a Field which is defined
on a 3-dimensional domain.

The assumption is that Y contains FieldVectors which themselves contain either
FieldVectors or Fields, and that the final unpacked variable is a Field.
This method is invoked when the element cannot be unpacked further.
We further assume that all fields in `Y` are defined on cell centers.
"""
function dss_helper!(
    field::ClimaCore.Fields.Field,
    space::ClimaCore.Spaces.ExtrudedFiniteDifferenceSpace,
    p::NamedTuple,
)
    buffer = p.dss_buffer_3d
    ClimaCore.Spaces.weighted_dss!(field, buffer)
end

"""
    dss_helper!(
        field::ClimaCore.Fields.Field,
        space::ClimaCore.Spaces.AbstractSpectralElementSpace,
        p::NamedTuple)

Method of `dss_helper!` which performs dss on a Field which is defined
on a 2-dimensional domain.

The assumption is that Y contains FieldVectors which themselves contain either
FieldVectors or Fields, and that the final unpacked variable is a Field.
This method is invoked when the element cannot be unpacked further.
We further assume that all fields in `Y` are defined on cell centers.
"""
function dss_helper!(
    field::ClimaCore.Fields.Field,
    space::ClimaCore.Spaces.AbstractSpectralElementSpace,
    p::NamedTuple,
)
    buffer = p.dss_buffer_2d
    # The buffer is set up expecting a single scalar at each point in
    # the space. if the field element type is a scalar, we are ready to
    # compute the dss. However, if the field is Tuple-valued,
    # we need to carry out the dss for each element of the tuple (each
    # of which is equivalent to a scalar valued field)
    field_element_type = eltype(field)
    if field_element_type <: AbstractFloat
        ClimaCore.Spaces.weighted_dss!(field, buffer)
    elseif field_element_type <: Tuple
        n = length(field_element_type.types)
        for i in 1:n
            ClimaCore.Spaces.weighted_dss!(field.:($i), buffer)
        end
    else
        @error("No DSS method defined for your field type.")
    end

end

"""
    dss_helper!(
        field::Union{ClimaCore.Fields.Field, Vector},
        space::Union{
            ClimaCore.Spaces.FiniteDifferenceSpace,
            ClimaCore.Spaces.PointSpace,
            Tuple,
        }, _)

Method of `dss_helper!` which does not perform dss.

This is intended for spaces that don't use spectral
elements (FiniteDifferenceSpace, PointSpace, etc).
Model components with no prognostic variables appear in Y as empty
Vectors, and also do not need dss.
"""
function dss_helper!(
    field::Union{ClimaCore.Fields.Field, Vector},
    space::Union{
        ClimaCore.Spaces.FiniteDifferenceSpace,
        ClimaCore.Spaces.PointSpace,
        Tuple,
    },
    _,
) end


"""
    DriverAffect{updateType, updateFType}

This struct is used by `DriverUpdateCallback` to update the values of
`p.drivers` at different timesteps specified by `updateat`, using the
function `updatefunc` which takes as arguments (p, t).
"""
mutable struct DriverAffect{updateType, updateFType}
    updateat::updateType
    updatefunc::updateFType
end

"""
    (affect!::DriverAffect)(integrator)

This function is used by `DriverUpdateCallback` to perform the updating.
"""
function (affect!::DriverAffect)(integrator)
    # If there are still update times in the queue and
    # they are less than the current simulation time,
    # cycle through until you reach the `updateat` value
    # closest to, but less than, the current simulation time.
    # This is important if the user happens to set update times
    # such that there are multiple per timestep
    while !isempty(affect!.updateat) && first(affect!.updateat) <= integrator.t
        curr_t = popfirst!(affect!.updateat)
        cond = curr_t <= integrator.t && curr_t > (integrator.t - integrator.dt)
        if cond
            # update all drivers to curr_t
            affect!.updatefunc(integrator.p, curr_t)
        end
    end
end

"""
    DriverUpdateCallback(updateat::Vector{FT}, updatefunc)

Constructs a DiscreteCallback which updates the cache `p.drivers` at each time
specified by `updateat`, using the function `updatefunc` which takes as arguments (p,t).
"""
function DriverUpdateCallback(updateat::Vector{FT}, updatefunc) where {FT}
    cond = update_condition(updateat)
    affect! = DriverAffect(updateat, updatefunc)

    SciMLBase.DiscreteCallback(
        cond,
        affect!;
        initialize = driver_initialize,
        save_positions = (false, false),
    )
end

"""
    CheckpointCallback(checkpoint_frequency::Union{AbstractFloat, Dates.Period},
                        output_dir, start_date, t_start; model, dt)

Constructs a DiscreteCallback which saves the state to disk with the
`save_checkpoint` function.

# Arguments
- `checkpoint_frequency`: The frequency at which checkpoints are saved. Can be
  specified as a float (in seconds) or a `Dates.Period`.
- `output_dir`: The directory where the checkpoint files will be saved.
- `start_date`: The start date of the simulation.
- `t_start`: The starting time of the simulation (in seconds).
- `model`: The ClimaLand model object.
- `dt`: The timestep of the model (optional), used to check for consistency.

The callback uses `ClimaDiagnostics.EveryCalendarDtSchedule` to determine when
to save checkpoints based on the `checkpoint_frequency`. The schedule is
initialized with the `start_date` and `t_start` to ensure that the first
checkpoint is saved at the correct time.

The `save_checkpoint` function is called with the current state vector `u`, the
current time `t`, and the `output_dir` to save the checkpoint to disk.
"""
function CheckpointCallback(
    checkpoint_frequency::Union{AbstractFloat, Dates.Period},
    output_dir,
    start_date,
    t_start;
    model,
    dt = nothing,
)
    # TODO: Move to a more general callback system. For the time being, we use
    # the ClimaDiagnostics one because it is flexible and it supports calendar
    # dates.

    if checkpoint_frequency isa AbstractFloat
        # Assume it is in seconds, but go through Millisecond to support
        # fractional seconds
        checkpoint_frequency_period =
            Dates.Millisecond(1000checkpoint_frequency)
    else
        checkpoint_frequency_period = checkpoint_frequency
    end

    schedule = EveryCalendarDtSchedule(
        checkpoint_frequency_period;
        start_date,
        date_last = start_date + Dates.Millisecond(1000t_start),
    )

    if !isnothing(dt)
        dt_period = Dates.Millisecond(1000dt)
        if !isdivisible(checkpoint_frequency_period, dt_period)
            @warn "Checkpoint frequency ($(checkpoint_frequency_period)) is not an integer multiple of dt $(dt_period)"
        end
    end

    cond = let schedule = schedule
        (u, t, integrator) -> schedule(integrator)
    end
    affect! = let output_dir = output_dir, model = model
        (integrator) ->
            save_checkpoint(integrator.u, integrator.t, output_dir; model)
    end

    SciMLBase.DiscreteCallback(cond, affect!)
end

"""
    driver_initialize(cb, u, t, integrator)

This function updates `p.drivers` at the start of the simulation.
"""
function driver_initialize(cb, u, t, integrator)
    cb.affect!.updatefunc(integrator.p, t)
end

"""
    update_condition(updateat)

This function returns a function with the type signature expected by
`SciMLBase.DiscreteCallback`, and determines whether `affect!` gets called in
the callback. This implementation simply checks if the current time of the
simulation is within the (inclusive) bounds of `updateat`.
"""
update_condition(updateat) =
    (_, t, _) -> t >= minimum(updateat) && t <= maximum(updateat)
"""
    SavingAffect{saveatType}

This struct is used by `NonInterpSavingCallback` to fill `saved_values` with
values of `p` at various timesteps. The `saveiter` field allows us to
allocate `saved_values` before the simulation and fill it during the run,
rather than pushing to an initially empty structure.
"""
mutable struct SavingAffect{saveatType}
    saved_values::NamedTuple
    saveat::saveatType
    saveiter::Int
end

"""
    (affect!::SavingAffect)(integrator)

This function is used by `NonInterpSavingCallback` to perform the saving.
"""
function (affect!::SavingAffect)(integrator)
    while !isempty(affect!.saveat) && first(affect!.saveat) <= integrator.t
        affect!.saveiter += 1
        curr_t = popfirst!(affect!.saveat)
        # @assert curr_t == integrator.t
        if curr_t == integrator.t
            affect!.saved_values.t[affect!.saveiter] = curr_t
            affect!.saved_values.saveval[affect!.saveiter] =
                deepcopy(integrator.p)
        end
    end
end

"""
    saving_initialize(cb, u, t, integrator)

This function saves t and p at the start of the simulation, as long as the
initial time is in `saveat`. To run the simulation without saving these
initial values, don't pass the `initialize` argument to the `DiscreteCallback`
constructor.
"""
function saving_initialize(cb, u, t, integrator)
    (t in cb.affect!.saveat) && cb.affect!(integrator)
end

"""
    NonInterpSavingCallback(saved_values, saveat::Vector{FT})

Constructs a DiscreteCallback which saves the time and cache `p` at each time
specified by `saveat`. The first argument must be a named
tuple containing `t` and `saveval`, each having the same length as `saveat`.

Important: The times in `saveat` must be times the simulation is
evaluated at for this function to work.

Note that unlike SciMLBase's SavingCallback, this version does not
interpolate if a time in saveat is not a multiple of our timestep. This
function also doesn't work with adaptive timestepping.
"""
function NonInterpSavingCallback(saved_values, saveat::Vector{FT}) where {FT}
    # This assumes that saveat contains multiples of the timestep
    cond = condition(saveat)
    saveiter = 0
    affect! = SavingAffect(saved_values, saveat, saveiter)

    SciMLBase.DiscreteCallback(
        cond,
        affect!;
        initialize = saving_initialize,
        save_positions = (false, false),
    )
end

"""
    condition(saveat)

This function returns a function with the type signature expected by
`SciMLBase.DiscreteCallback`, and determines whether `affect!` gets
called in the callback. This implementation simply checks if the current time
is contained in the list of affect times used for the callback.
"""
condition(saveat) = (_, t, _) -> t in saveat

function FTfromY(Y::ClimaCore.Fields.FieldVector)
    return eltype(Y)
end


"""
    isdivisible(dt_large::Dates.Period, dt_small::Dates.Period)

Check if two periods are evenly divisible, i.e., if the larger period can be
expressed as an integer multiple of the smaller period.

In this, take into account the case when periods do not have fixed size, e.g.,
one month is a variable number of days.

# Examples
```
julia> isdivisible(Dates.Year(1), Dates.Month(1))
true

julia> isdivisible(Dates.Month(1), Dates.Day(1))
true

julia> isdivisible(Dates.Month(1), Dates.Week(1))
false
```

## Notes

Not all the combinations are fully implemented. If something is missing, please
consider adding it.
"""
function isdivisible(dt_large::Dates.Period, dt_small::Dates.Period)
    @warn "The combination $(typeof(dt_large)) and $(dt_small) was not covered. Please add a method to handle this case."
    return false
end

# For FixedPeriod and OtherPeriod, it is easy, we can directly divide the two
# (as long as they are both the same)
function isdivisible(dt_large::Dates.FixedPeriod, dt_small::Dates.FixedPeriod)
    return isinteger(dt_large / dt_small)
end

function isdivisible(dt_large::Dates.OtherPeriod, dt_small::Dates.OtherPeriod)
    return isinteger(dt_large / dt_small)
end

function isdivisible(
    dt_large::Union{Dates.Month, Dates.Year},
    dt_small::Dates.FixedPeriod,
)
    # The only case where periods are commensurate for Month/Year is when we
    # have a Day or an integer divisor of a day. (Note that 365 and 366 don't
    # have any common divisor)
    return isinteger(Dates.Day(1) / dt_small)
end

"""
    count_nans_state(state, mask::ClimaCore.Fields.Field = nothing, verbose = false)

Count the number of NaNs in the state, e.g. the FieldVector given by
`sol.u[end]` after calling `solve`. This function is useful for
debugging simulations to determine quantitatively if a simulation is stable; this is only
for use in non-column simulations.

If this function is called on a FieldVector, it will recursively call itself
on each Field in the FieldVector. If it is called on a Field, it will count
the number of NaNs in the Field and produce a warning if any are found; this behavior
is different for 2d and 3d fields, which is why we dispatch on the `space`.

If a ClimaCore Field is provided as `mask`, the function will only count NaNs
in the state variables where the mask is 1. This is intended to be used with
the land/sea mask, to avoid counting NaNs over the ocean. Note this assumes
the mask is 1 over land and 0 over ocean, and that the mask is a 2D field.

The `verbose` argument toggles whether the function produces output when no
NaNs are found.
"""
function count_nans_state(
    state::ClimaCore.Fields.FieldVector;
    mask = nothing,
    verbose = false,
)
    for pn in propertynames(state)
        state_new = getproperty(state, pn)
        @info "Checking NaNs in $pn"
        count_nans_state(state_new; mask, verbose)
    end
    return nothing
end

function count_nans_state(
    state::ClimaCore.Fields.Field;
    mask = nothing,
    verbose = false,
)
    return count_nans_state(state, axes(state); mask = mask, verbose = verbose)
end

"""
    count_nans_state(
        state::ClimaCore.Fields.Field,
        space::ClimaCore.Spaces.AbstractSpectralElementSpace;
        mask = nothing,
        verbose = false,
    )

Counts the NaNs in the field `state` where the `state is defined on a
spectral element space.
"""
function count_nans_state(
    state::ClimaCore.Fields.Field,
    space::ClimaCore.Spaces.AbstractSpectralElementSpace;
    mask = nothing,
    verbose = false,
)
    # Note: this code uses `parent`; this pattern should not be replicated
    num_nans = 0
    ClimaComms.allowscalar(ClimaComms.device()) do
        num_nans =
            isnothing(mask) ? round(sum(isnan, parent(state))) :
            round(sum(isnan, parent(state)[Bool.(parent(mask))]; init = 0))
    end
    if isapprox(num_nans, 0)
        verbose && @info "No NaNs found"
    else
        @warn "$num_nans NaNs found"
    end
    return nothing
end

"""
    count_nans_state(
        state::ClimaCore.Fields.Field,
        space::ClimaCore.Spaces.ExtrudedFiniteDifferenceSpace;
        mask = nothing,
        verbose = false,
    )

Counts the NaNs in the surface level of the field `state`,
 where the `state is defined on an extruded finite difference space.
"""
function count_nans_state(
    state::ClimaCore.Fields.Field,
    space::ClimaCore.Spaces.ExtrudedFiniteDifferenceSpace;
    mask = nothing,
    verbose = false,
)
    # Note: this code uses `parent`; this pattern should not be replicated
    surface_state = ClimaLand.Domains.top_center_to_surface(state)
    num_nans = 0
    ClimaComms.allowscalar(ClimaComms.device()) do
        num_nans =
            isnothing(mask) ? round(sum(isnan, parent(surface_state))) :
            round(
                sum(
                    isnan,
                    parent(surface_state)[Bool.(parent(mask))];
                    init = 0,
                ),
            )
    end
    if isapprox(num_nans, 0)
        verbose && @info "No NaNs found"
    else
        @warn "$num_nans NaNs found"
    end
    return nothing
end

"""
    NaNCheckCallback(nancheck_frequency::Union{AbstractFloat, Dates.Period},
                        start_date, t_start, dt)

Constructs a DiscreteCallback which counts the number of NaNs in the state
and produces a warning if any are found.

# Arguments
- `nancheck_frequency`: The frequency at which the state is checked for NaNs.
  Can be specified as a float (in seconds) or a `Dates.Period`.
- `start_date`: The start date of the simulation.
- `t_start`: The starting time of the simulation (in seconds).
- `dt`: The timestep of the model (optional), used to check for consistency.

The callback uses `ClimaDiagnostics.EveryCalendarDtSchedule` to determine when
to check for NaNs based on the `nancheck_frequency`. The schedule is
initialized with the `start_date` and `t_start` to ensure that it is first
called at the correct time.
"""
function NaNCheckCallback(
    nancheck_frequency::Union{AbstractFloat, Dates.Period},
    start_date,
    t_start,
    dt;
    mask = nothing,
)
    # TODO: Move to a more general callback system. For the time being, we use
    # the ClimaDiagnostics one because it is flexible and it supports calendar
    # dates.

    if nancheck_frequency isa AbstractFloat
        # Assume it is in seconds, but go through Millisecond to support
        # fractional seconds
        nancheck_frequency_period = Dates.Millisecond(1000nancheck_frequency)
    else
        nancheck_frequency_period = nancheck_frequency
    end

    schedule = EveryCalendarDtSchedule(
        nancheck_frequency_period;
        start_date,
        date_last = start_date + Dates.Millisecond(1000t_start),
    )

    if !isnothing(dt)
        dt_period = Dates.Millisecond(1000dt)
        if !isdivisible(nancheck_frequency_period, dt_period)
            @warn "Callback frequency ($(nancheck_frequency_period)) is not an integer multiple of dt $(dt_period)"
        end
    end

    cond = let schedule = schedule
        (u, t, integrator) -> schedule(integrator)
    end
    affect! = (integrator) -> count_nans_state(integrator.u; mask)

    SciMLBase.DiscreteCallback(cond, affect!)
end
