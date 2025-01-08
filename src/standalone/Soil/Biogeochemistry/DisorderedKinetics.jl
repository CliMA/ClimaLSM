module DisorderedKinetics
using Distributions
using ClimaLand
using DocStringExtensions
using ClimaCore
import ...Parameters as LP
import ClimaCore: Fields, Operators, Geometry, Spaces

import ClimaLand.Domains: AbstractDomain
import ClimaLand:
    AbstractExpModel,
    make_update_aux,
    make_compute_exp_tendency,
    make_update_boundary_fluxes,
    prognostic_vars,
    auxiliary_vars,
    name,
    prognostic_types,
    auxiliary_types,
    prognostic_domain_names,
    auxiliary_domain_names,
    TopBoundary,
    BottomBoundary,
    AbstractBC,
    boundary_flux!,
    AbstractSource,
    source!

export DisorderedKineticsModelParameters,
    DisorderedKineticsModel,
    LitterInput,
    AbstractSoilDriver,
    DisorderedKineticsDrivers


"""
DisorderedKineticsModelParameters{FT <: AbstractFloat, PSE}

A struct for storing parameters of the `DisorderedKineticsModel`.

All of these parameters are currently treated as global constants.
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct DisorderedKineticsModelParameters{FT <: AbstractFloat}
    "the mean of the log-normal distribution of the soil C decay rate []"
    mu::FT
    "the standard deviation of the log-normal distribution of the soil C decay rate []"
    sigma::FT
end


"""
AbstractSoilCarbonModel{FT} <: ClimaLand.AbstractExpModel{FT}

An abstract model type for soil biogeochemistry models.
"""
abstract type AbstractSoilCarbonModel{FT} <:
        ClimaLand.AbstractExpModel{FT} end

"""
DisorderedKineticsModel

A model for simulating soil organic carbon dynamics using a disordered kinetics model.

$(DocStringExtensions.FIELDS)
"""
struct DisorderedKineticsModel{FT, PS, N, D, S, DT} <:
    AbstractSoilCarbonModel{FT}
    "the parameter set"
    parameters::PS
    "the number of soil carbon pools for discretization"
    npools::N
    "the soil domain, using ClimaCore.Domains"
    domain::D
    "A tuple of sources, each of type AbstractSource"
    sources::S
    "Drivers"
    drivers::DT
end


"""
DisorderedKineticsModel{FT}(;
        parameters::DisorderedKineticsModelParameters{FT},
        npools::Int64,
        domain::ClimaLand.AbstractDomain,
        sources::Tuple,
        drivers::DT
    ) where {FT, BC}

A constructor for `DisorderedKineticsModel`.
"""
function DisorderedKineticsModel{FT}(;
    parameters::DisorderedKineticsModelParameters{FT},
    npools::Int64,
    domain::ClimaLand.AbstractDomain,
    sources::Tuple,
    drivers::DT,
) where {FT, DT}
    args = (parameters, npools, domain, sources, drivers)
    DisorderedKineticsModel{FT, typeof.(args)...}(args...)
end

ClimaLand.name(model::DisorderedKineticsModel) = :soilC
ClimaLand.prognostic_vars(::DisorderedKineticsModel) = (:C,)
ClimaLand.prognostic_types(model::DisorderedKineticsModel{FT}) where {FT} = (NTuple{model.npools, FT},)
# we use a vertical column with one layer to in the future interface with other variables like temperature and moisture that are depth dependent
ClimaLand.prognostic_domain_names(::DisorderedKineticsModel) = (:subsurface,)

ClimaLand.auxiliary_vars(model::DisorderedKineticsModel) = (:ks,:inputs)
ClimaLand.auxiliary_types(model::DisorderedKineticsModel{FT}) where {FT} = (NTuple{model.npools, FT}, FT)
# we use a vertical column with one layer to in the future interface with other variables like temperature and moisture that are depth dependent
ClimaLand.auxiliary_domain_names(model::DisorderedKineticsModel) = (:subsurface, :subsurface)

"""
    make_compute_exp_tendency(model::DisorderedKineticsModel)

An extension of the function `make_compute_exp_tendency`, for the disordered kinetics model.
This function creates and returns a function which computes the entire
right hand side of the ODE for `C`, and updates `dY.soilC.C` in place
with that value. These quantities will be stepped explicitly.

This has been written so as to work with Differential Equations.jl.
"""
function ClimaLand.make_compute_exp_tendency(model::DisorderedKineticsModel)
    function compute_exp_tendency!(dY, Y, p, t)
        
        @. dY.soilC.C = - p.soilC.ks * Y.soilC.C

        # Source terms are added in here
        for src in model.sources
            print(src)
            source!(dY, src, Y, p, model.parameters)
        end
    end
    return compute_exp_tendency!
end


"""
    LitterInput{FT} <: AbstractCarbonSource{FT}

Struct for the litter input of C into the SOC pool, appearing as a source
term in the differential equation.
"""
struct LitterInput{FT} <: ClimaLand.AbstractSource{FT} end

"""
    ClimaLand.source!(dY::ClimaCore.Fields.FieldVector,
                          src::AbstractSoilCarbonSource,
                          Y::ClimaCore.Fields.FieldVector,
                          p::NamedTuple,
                          params)

A method which extends the ClimaLand source! function for the
case of the disordered kinetics model.
"""
function ClimaLand.source!(
    dY::ClimaCore.Fields.FieldVector,
    src::LitterInput,
    Y::ClimaCore.Fields.FieldVector,
    p::NamedTuple,
    params::DisorderedKineticsModelParameters,
)
    @. dY.soilC.C += p.soilC.inputs * p_k(p.soilC.ks,params.mu,params.sigma);
end

"""
    p_k(k::N)

    a function to calculate the pdf of the lognormal distribution of each decay rate k
"""
function p_k(k,mu,sigma) @. (1/(k* sigma * sqrt(2*π))) * exp(-((log(k) - mu)^2)/(2*sigma^2)); end





"""
    SoilDrivers

A container which passes in the soil drivers to the biogeochemistry
model. These drivers are either of type Prescribed (for standalone mode)
or Prognostic (for running with a prognostic model for soil temp and moisture).

$(DocStringExtensions.FIELDS)
"""
struct DisorderedKineticsDrivers{
    FT,
    SOC_input <: PrescribedSOCInputs{FT},
}
    "Soil SOM driver - Prescribed only"
    soc_inputs::SOC_input
end



"""
    make_update_aux(model::DisorderedKineticsModel)

An extension of the function `make_update_aux`, for the disordered kinetics model.
This function creates and returns a function which updates the auxiliary
variables `p.soilC.variable` in place.
This has been written so as to work with Differential Equations.jl.

This function is empty because the auxiliary variable `ks` is not updated in time.
"""
function ClimaLand.make_update_aux(model::DisorderedKineticsModel)
    function update_aux!(p, Y, t)
        p.soilC.inputs .= model.drivers.soc.func.(t)
    end
    return update_aux!
end

Base.broadcastable(ps::DisorderedKineticsModelParameters) = tuple(ps)

end