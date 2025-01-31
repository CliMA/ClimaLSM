module Rivers
using ClimaLand
using ClimaCore
using DocStringExtensions
import ClimaCore: Fields

import ClimaLand:
    AbstractExpModel,
    make_compute_exp_tendency,
    prognostic_vars,
    name,
    prognostic_types,
    prognostic_domain_names,
    auxiliary_vars,
    auxiliary_types,
    auxiliary_domain_names,
    make_update_boundary_fluxes,
    FTfromY

using ClimaLand.Domains
import ClimaLand.Domains:
    coordinates
export RiverModel

abstract type AbstractRiverModel{FT} <: AbstractExpModel{FT} end

abstract type AbstractRiverSimulationMode{FT<: AbstractFloat} end

"""
    RiverModel{FT, M, D, RD} <: AbstractSurfaceWaterModel{FT}

Interfacer for integrating ClimaLand.jl and ClimaRivers.jl.

$(DocStringExtensions.FIELDS)
"""
struct RiverModel{FT, M<: AbstractRiverMode, GD <: Domains.SphericalShell{FT}, BD} <: AbstractRiverModel{FT}
    "Mode - standalone or prescribed"
    mode::M
    "The ClimaCore (grid) domain for the land model; global only"
    grid_domain::GD
    "The river basin domain"
    basin_domain::BD
    "The river model timestep"
    Δt_river::FT
    "time bracket"
    t_bracket::NTuple{2,FT}
    "History length"
    hist_length::Int
end

ClimaLand.name(model::RiverModel) = :river

ClimaLand.prognostic_vars(model::RiverModel) = (:R_accum,)
ClimaLand.prognostic_types(model::RiverModel{FT}) where {FT} = (FT,)
ClimaLand.prognostic_domain_names(model::RiverModel) = (:grid,)
ClimaLand.auxiliary_vars(model::RiverModel) = (:R_accum_history,:R_inst, :state)
ClimaLand.auxiliary_types(model::RiverModel{FT}) where {FT} = (NTuple{model.hist_length, FT},FT, FT)
ClimaLand.auxiliary_domain_names(model::RiverModel) = (:basin,:grid, :basin)

function Domain.coordinates(basin_domain::Vector)
    return basin_domain
end

function Domain.coordinates(model::RiverModel)
    return (:grid = coordinates(model.grid_domain), :basin = coordinates(model.basin_domain))
end

function ClimaLand.make_compute_exp_tendency(model::RiverModel)
    function compute_exp_tendency!(dY, Y, p, t)
        R_inst = p.river.R_inst
        compute_instantaneous_runoff!(R_inst, model.runoff,Y,p,t)
        @. dY.river.R_accum = instantaneous_runoff(model.mode,Y,p,t)
    end
    return compute_exp_tendency!
end


struct Standalone{FT} <:  AbstractRiverSimulationMode{FT}
    "Forcing data"
    forcing_data::FT
end

struct Integrated{FT} <:  AbstractRiverSimulationMode{FT} end

function compute_instantaneous_runoff!(R_inst, mode::Standalone,Y,p,t)
    R_inst .= runoff.data(t) # on ClimaCore grid
end

function compute_instantaneous_runoff!(R_inst, mode::Integrated,Y,p,t)
    @. R_inst = p.soil.R_s + p.soil.R_ss # on ClimaCore grid
end

function river_callback!(Y,p,t, t_callback)
    # if we have stepped a full Δt_river from the previous step
    if t ≈ t_callback # = t_prev + Δt_river
        p.river.R_accum_history[1:end-1] .= p.river.R_accum_history[2:end]
        p.river.R_accum_history[end] .= grid_to_basin(Y.river.R_accum)
        Y.river.R_accum .= 0
        update_river_state!(p.river.state, p.river.R_accum_history) # on basin domain
    else
        nothing
    end
end
