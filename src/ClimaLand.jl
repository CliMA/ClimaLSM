module ClimaLand
using DocStringExtensions

using ClimaCore
import ClimaCore: Fields, Spaces

include("shared_utilities/Parameters.jl")
import .Parameters as LP

include("shared_utilities/Domains.jl")
import ClimaUtilities.TimeVaryingInputs
import ClimaUtilities.TimeVaryingInputs:
    TimeVaryingInput, AbstractTimeVaryingInput, evaluate!
export TimeVaryingInput, evaluate!
import ClimaUtilities.SpaceVaryingInputs
import ClimaUtilities.SpaceVaryingInputs: SpaceVaryingInput

import NCDatasets # Needed to load the ClimaUtilities.*VaryingInput
using .Domains
include("Artifacts.jl")
include("shared_utilities/checkpoints.jl")
include("shared_utilities/utils.jl")
include("shared_utilities/models.jl")
include("shared_utilities/drivers.jl")
include("shared_utilities/boundary_conditions.jl")
include("shared_utilities/sources.jl")
include("shared_utilities/implicit_timestepping.jl")
include("standalone/Bucket/Bucket.jl")

"""
     AbstractLandModel{FT} <: AbstractModel{FT}

An abstract type for all land model types, which are used
to simulated multiple land surface components as
a single system. Standalone component runs do not require
this interface and it should not be used for that purpose.

Many methods taking an argument of type `AbstractLandModel` are
extensions of functions defined for `AbstractModel`s.
There are default methods that apply for all `AbstractLandModel`s,
including `make_update_aux`, `make_exp_tendency`, `make_imp_tendency`,
`make_compute_exp_tendency`, `make_compute_imp_tendency`,
`initialize_prognostic`, `initialize_auxiliary`, `initialize`,
and `coordinates`.

Methods which dispatch on a specific type of AbstractLandModel
include any function involving interactions between components,
as these interactions depend on the components in the land model
and the versions of these component models being used.
"""
abstract type AbstractLandModel{FT} <: AbstractModel{FT} end

ClimaLand.name(::AbstractLandModel) = :land

"""
    Domains.coordinates(model::AbstractLandModel)

Returns a NamedTuple of the unique set of coordinates for the LSM
`model`, where the unique set is taken over the coordinates of all
of the subcomponents.

For example, an LSM with a single layer snow model, multi-layer
soil model, and canopy model would have a coordinate set corresponding
to the coordinates of the surface (snow), the subsurface coordinates (soil)
and the coordinates of the surface (canopy). This would return the coordinates
of the surface and subsurface. These are distinct because the subsurface
coordinates correspond to the centers of the layers, while the surface
corresponds to the top face of the domain.
"""
function Domains.coordinates(model::AbstractLandModel)
    components = land_components(model)
    coords_list = map(components) do (component)
        Domains.coordinates(getproperty(model, component))
    end
    unique_coords_list = merge(coords_list...)
    return unique_coords_list
end


function initialize_prognostic(
    model::AbstractLandModel{FT},
    coords::NamedTuple,
) where {FT}
    components = land_components(model)
    Y_state_list = map(components) do (component)
        submodel = getproperty(model, component)
        getproperty(initialize_prognostic(submodel, coords), component)
    end
    Y = ClimaCore.Fields.FieldVector(; NamedTuple{components}(Y_state_list)...)
    return Y
end

function initialize_auxiliary(
    model::AbstractLandModel{FT},
    coords::NamedTuple,
) where {FT}
    components = land_components(model)
    p_state_list = map(components) do (component)
        submodel = getproperty(model, component)
        getproperty(initialize_auxiliary(submodel, coords), component)
    end
    p_additional_aux = initialize_lsm_aux(model, coords)
    p = (; p_additional_aux..., NamedTuple{components}(p_state_list)...)
    domains_list = map(components) do (component)
        submodel = getproperty(model, component)
        getproperty(submodel, :domain)
    end
    for domain in unique(domains_list)
        p = add_dss_buffer_to_aux(p, domain)
    end
    return p
end

"""
    initialize_lsm_aux(land::AbstractLandModel) end

Initializes additional auxiliary variables required in integrated models, and
not existing in the individual component models auxiliary vars.

Additional auxiliary variables are specified by `lsm_aux_vars`, their types
by `lsm_aux_types`, and their domain names by `lsm_aux_domain_names`.
This function should be called during `initialize_auxiliary` step.
"""
function initialize_lsm_aux(land::AbstractLandModel, land_coords)
    vars = lsm_aux_vars(land)
    types = lsm_aux_types(land)
    domains = lsm_aux_domain_names(land)
    additional_aux = map(zip(types, domains)) do (T, domain)
        zero_instance = ClimaCore.RecursiveApply.rzero(T)
        map(_ -> zero_instance, getproperty(land_coords, domain))
    end
    return NamedTuple{vars}(additional_aux)
end


function make_imp_tendency(land::AbstractLandModel)
    components = land_components(land)
    compute_imp_tendency_list =
        map(x -> make_compute_imp_tendency(getproperty(land, x)), components)
    update_aux! = make_update_aux(land)
    update_boundary_fluxes! = make_update_boundary_fluxes(land)
    function imp_tendency!(dY, Y, p, t)
        update_aux!(p, Y, t)
        update_boundary_fluxes!(p, Y, t)
        for f! in compute_imp_tendency_list
            f!(dY, Y, p, t)
        end
    end
    return imp_tendency!
end

function make_exp_tendency(land::AbstractLandModel)
    components = land_components(land)
    compute_exp_tendency_list =
        map(x -> make_compute_exp_tendency(getproperty(land, x)), components)
    update_aux! = make_update_aux(land)
    update_boundary_fluxes! = make_update_boundary_fluxes(land)
    function exp_tendency!(dY, Y, p, t)
        update_aux!(p, Y, t)
        update_boundary_fluxes!(p, Y, t)
        for f! in compute_exp_tendency_list
            f!(dY, Y, p, t)
        end
    end
    return exp_tendency!
end

function make_update_aux(land::AbstractLandModel)
    components = land_components(land)
    update_aux_function_list =
        map(x -> make_update_aux(getproperty(land, x)), components)
    function update_aux!(p, Y, t)
        for f! in update_aux_function_list
            f!(p, Y, t)
        end
    end
    return update_aux!
end

function make_update_boundary_fluxes(land::AbstractLandModel)
    components = land_components(land)
    update_fluxes_function_list =
        map(x -> make_update_boundary_fluxes(getproperty(land, x)), components)
    function update_boundary_fluxes!(p, Y, t)
        for f! in update_fluxes_function_list
            f!(p, Y, t)
        end
    end
    return update_boundary_fluxes!
end

function make_compute_jacobian(land::AbstractLandModel)
    components = land_components(land)
    compute_jacobian_function_list =
        map(x -> make_compute_jacobian(getproperty(land, x)), components)
    function compute_jacobian!(jacobian, Y, p, dtγ, t)
        for f! in compute_jacobian_function_list
            f!(jacobian, Y, p, dtγ, t)
        end
    end
    return compute_jacobian!
end

"""
    land_components(land::AbstractLandModel)

Returns the component names of the `land` model, by calling
`propertynames(land)`.
"""
land_components(land::AbstractLandModel) = propertynames(land)

function prognostic_vars(land::AbstractLandModel)
    components = land_components(land)
    prognostic_list = map(components) do model
        prognostic_vars(getproperty(land, model))
    end
    return NamedTuple{components}(prognostic_list)
end

function prognostic_types(land::AbstractLandModel)
    components = land_components(land)
    prognostic_list = map(components) do model
        prognostic_types(getproperty(land, model))
    end
    return NamedTuple{components}(prognostic_list)
end

function prognostic_domain_names(land::AbstractLandModel)
    components = land_components(land)
    prognostic_list = map(components) do model
        prognostic_domain_names(getproperty(land, model))
    end
    return NamedTuple{components}(prognostic_list)
end


function auxiliary_vars(land::AbstractLandModel)
    components = land_components(land)
    auxiliary_list = map(components) do model
        auxiliary_vars(getproperty(land, model))
    end
    return NamedTuple{(components..., :lsm_aux)}((
        auxiliary_list...,
        lsm_aux_vars(land),
    ))
end

function auxiliary_types(land::AbstractLandModel)
    components = land_components(land)
    auxiliary_list = map(components) do model
        auxiliary_types(getproperty(land, model))
    end
    return NamedTuple{(components..., :lsm_aux)}((
        auxiliary_list...,
        lsm_aux_types(land),
    ))
end

function auxiliary_domain_names(land::AbstractLandModel)
    components = land_components(land)
    auxiliary_list = map(components) do model
        auxiliary_domain_names(getproperty(land, model))
    end
    return NamedTuple{(components..., :lsm_aux)}((
        auxiliary_list...,
        lsm_aux_domain_names(land),
    ))
end


"""
   lsm_aux_vars(m::AbstractLandModel)

Returns the additional aux variable symbols for the model in the form of a tuple.
"""
lsm_aux_vars(m::AbstractLandModel) = ()

"""
   lsm_aux_types(m::AbstractLandModel)

Returns the shared additional aux variable types for the model in the form of a tuple.
"""
lsm_aux_types(m::AbstractLandModel) = ()

"""
   lsm_aux_domain_names(m::AbstractLandModel)

Returns the additional domain symbols in the form of a tuple e.g. :surface or :subsurface.

This is only required for variables shared between land submodels, and only needed
for multi-component models, not standalone components. Component-specific variables
should be listed as prognostic or auxiliary variables which do not require this to
initialize.
"""
lsm_aux_domain_names(m::AbstractLandModel) = ()

# Methods extended by the LSM models we support
include("standalone/SurfaceWater/Pond.jl")
using .Pond
import .Pond: surface_runoff
include("standalone/Soil/Soil.jl")
using .Soil
import .Soil: soil_boundary_fluxes!, sublimation_source
import .Soil.Biogeochemistry: soil_temperature, soil_moisture
include("standalone/Snow/Snow.jl")
using .Snow
import .Snow: snow_boundary_fluxes!
include("standalone/Vegetation/Canopy.jl")
using .Canopy
using .Canopy.PlantHydraulics
import .Canopy.PlantHydraulics: root_water_flux_per_ground_area!
import .Canopy:
    ground_albedo_PAR,
    ground_albedo_NIR,
    canopy_radiant_energy_fluxes!,
    root_energy_flux_per_ground_area!,
    AbstractGroundConditions
### Concrete types of AbstractLandModels
### and associated methods
include("integrated/soil_energy_hydrology_biogeochemistry.jl")
include("integrated/pond_soil_model.jl")
include("integrated/soil_canopy_model.jl")
include("integrated/soil_snow_model.jl")
include("integrated/land.jl")
include("integrated/soil_canopy_root_interactions.jl")
include("integrated/land_radiation.jl")

# Diagnostics
include(joinpath("diagnostics", "Diagnostics.jl"))
import .Diagnostics: default_diagnostics

# Simulations
include(joinpath("simulations", "spatial_parameters.jl"))
include(joinpath("simulations", "initial_conditions.jl"))
include(joinpath("simulations", "domains.jl"))

end
