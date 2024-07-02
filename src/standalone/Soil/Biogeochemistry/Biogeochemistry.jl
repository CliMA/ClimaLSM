module Biogeochemistry
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
    boundary_flux,
    AbstractSource,
    source!
export SoilCO2ModelParameters,
    SoilCO2Model,
    PrescribedMet,
    PrescribedSOC,
    MicrobeProduction,
    SoilCO2FluxBC,
    AtmosCO2StateBC,
    SoilCO2StateBC,
    AbstractSoilDriver,
    SoilDrivers

"""
    SoilCO2ModelParameters{FT <: AbstractFloat, PSE}

A struct for storing parameters of the `SoilCO2Model`.
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct SoilCO2ModelParameters{FT <: AbstractFloat, PSE}
    "Diffusion coefficient for CO₂ in air at standard temperature and pressure (m² s⁻¹)"
    D_ref::FT
    "Diffusivity of soil C substrate in liquid (unitless)"
    D_liq::FT
    # DAMM
    "Pre-exponential factor (kg C m-3 s-1)"
    α_sx::FT
    "Activation energy (J mol-1)"
    Ea_sx::FT
    "Michaelis constant (kg C m-3)"
    kM_sx::FT
    "Michaelis constant for O2 (m3 m-3)"
    kM_o2::FT
    "Volumetric fraction of O₂ in the soil air, dimensionless"
    O2_a::FT
    "Diffusion coefficient of oxygen in air, dimensionless"
    D_oa::FT
    "Fraction of soil carbon that is considered soluble, dimensionless"
    p_sx::FT
    "Physical constants used Clima-wide"
    earth_param_set::PSE
end

"""
    AbstractSoilBiogeochemistryModel{FT} <: ClimaLand.AbstractExpModel{FT}

An abstract model type for soil biogeochemistry models.
"""
abstract type AbstractSoilBiogeochemistryModel{FT} <:
              ClimaLand.AbstractExpModel{FT} end

"""
    SoilCO2Model

A model for simulating the production and transport of CO₂ in the soil with dynamic
source and diffusion terms.
$(DocStringExtensions.FIELDS)
"""
struct SoilCO2Model{FT, PS, D, BC, S, DT} <:
       AbstractSoilBiogeochemistryModel{FT}
    "the parameter set"
    parameters::PS
    "the soil domain, using ClimaCore.Domains"
    domain::D
    "the boundary conditions, of type NamedTuple"
    boundary_conditions::BC
    "A tuple of sources, each of type AbstractSource"
    sources::S
    "Drivers"
    driver::DT
end


"""
SoilCO2Model{FT}(;
        parameters::SoilCO2ModelParameters{FT},
        domain::ClimaLand.AbstractDomain,
        boundary_conditions::NamedTuple,
        sources::Tuple,
        drivers::DT,
    ) where {FT, BC, DT}

A constructor for `SoilCO2Model`.
"""
function SoilCO2Model{FT}(;
    parameters::SoilCO2ModelParameters{FT},
    domain::ClimaLand.AbstractDomain,
    boundary_conditions::BC,
    sources::Tuple,
    drivers::DT,
) where {FT, BC, DT}
    args = (parameters, domain, boundary_conditions, sources, drivers)
    SoilCO2Model{FT, typeof.(args)...}(args...)
end

ClimaLand.name(model::SoilCO2Model) = :soilco2
ClimaLand.prognostic_vars(::SoilCO2Model) = (:C,)
ClimaLand.prognostic_types(::SoilCO2Model{FT}) where {FT} = (FT,)
ClimaLand.prognostic_domain_names(::SoilCO2Model) = (:subsurface,)

ClimaLand.auxiliary_vars(model::SoilCO2Model) = (
    :D,
    :Sm,
    ClimaLand.boundary_vars(
        model.boundary_conditions.top,
        ClimaLand.TopBoundary(),
    )...,
    ClimaLand.boundary_vars(
        model.boundary_conditions.bottom,
        ClimaLand.BottomBoundary(),
    )...,
)


ClimaLand.auxiliary_types(model::SoilCO2Model{FT}) where {FT} = (
    FT,
    FT,
    ClimaLand.boundary_var_types(
        model,
        model.boundary_conditions.top,
        ClimaLand.TopBoundary(),
    )...,
    ClimaLand.boundary_var_types(
        model,
        model.boundary_conditions.bottom,
        ClimaLand.BottomBoundary(),
    )...,
)
ClimaLand.auxiliary_domain_names(model::SoilCO2Model) = (
    :subsurface,
    :subsurface,
    ClimaLand.boundary_var_domain_names(
        model.boundary_conditions.top,
        ClimaLand.TopBoundary(),
    )...,
    ClimaLand.boundary_var_domain_names(
        model.boundary_conditions.bottom,
        ClimaLand.BottomBoundary(),
    )...,
)

function make_update_boundary_fluxes(model::SoilCO2Model)
    function update_boundary_fluxes!(p, Y, t)
        Δz_top = model.domain.fields.Δz_top
        Δz_bottom = model.domain.fields.Δz_bottom
        p.soilco2.top_bc .= boundary_flux(
            model.boundary_conditions.top,
            TopBoundary(),
            Δz_top,
            Y,
            p,
            t,
        )
        p.soilco2.bottom_bc .= boundary_flux(
            model.boundary_conditions.bottom,
            BottomBoundary(),
            Δz_bottom,
            Y,
            p,
            t,
        )

    end
    return update_boundary_fluxes!
end

"""
    make_compute_exp_tendency(model::SoilCO2Model)

An extension of the function `make_compute_exp_tendency`, for the soilco2 equation.
This function creates and returns a function which computes the entire
right hand side of the PDE for `C`, and updates `dY.soil.C` in place
with that value. These quantities will be stepped explicitly.

This has been written so as to work with Differential Equations.jl.
"""
function ClimaLand.make_compute_exp_tendency(model::SoilCO2Model)
    function compute_exp_tendency!(dY, Y, p, t)
        top_flux_bc = p.soilco2.top_bc
        bottom_flux_bc = p.soilco2.bottom_bc

        interpc2f = ClimaCore.Operators.InterpolateC2F()
        gradc2f_C = ClimaCore.Operators.GradientC2F()
        divf2c_C = ClimaCore.Operators.DivergenceF2C(
            top = ClimaCore.Operators.SetValue(
                ClimaCore.Geometry.WVector.(top_flux_bc),
            ),
            bottom = ClimaCore.Operators.SetValue(
                ClimaCore.Geometry.WVector.(bottom_flux_bc),
            ),
        ) # -∇ ⋅ (-D∇C), where -D∇C is a flux of CO2. ∇C point in direction of increasing C, so the flux is - this.
        @. dY.soilco2.C =
            -divf2c_C(-interpc2f(p.soilco2.D) * gradc2f_C(Y.soilco2.C))

        # Source terms are added in here
        for src in model.sources
            source!(dY, src, Y, p, model.parameters)
        end

    end
    return compute_exp_tendency!
end

"""
    AbstractCarbonSource{FT} <: ClimaLand.AbstractSource{FT}

An abstract type for soil CO2 sources. There are two sources:
roots and microbes, in struct RootProduction and MicrobeProduction.
"""
abstract type AbstractCarbonSource{FT} <: ClimaLand.AbstractSource{FT} end

"""
    MicrobeProduction{FT} <: AbstractCarbonSource{FT}

Struct for the microbe production of CO2, appearing as a source
term in the differential equation.
"""
struct MicrobeProduction{FT} <: AbstractCarbonSource{FT} end

"""
    ClimaLand.source!(dY::ClimaCore.Fields.FieldVector,
                          src::MicrobeProduction,
                          Y::ClimaCore.Fields.FieldVector,
                          p::NamedTuple,
                          params)

A method which extends the ClimaLand source! function for the
case of microbe production of CO2 in soil.
"""
function ClimaLand.source!(
    dY::ClimaCore.Fields.FieldVector,
    src::MicrobeProduction,
    Y::ClimaCore.Fields.FieldVector,
    p::NamedTuple,
    params,
)
    dY.soilco2.C .+= p.soilco2.Sm
end


"""
    make_update_aux(model::SoilCO2Model)

An extension of the function `make_update_aux`, for the soilco2 equation.
This function creates and returns a function which updates the auxiliary
variables `p.soil.variable` in place.
This has been written so as to work with Differential Equations.jl.
"""
function ClimaLand.make_update_aux(model::SoilCO2Model)
    function update_aux!(p, Y, t)
        # get FT to enforce types of variables not stored directly in `p`
        FT = eltype(Y.soilco2.C)
        params = model.parameters
        z = model.domain.fields.z
        T_soil = soil_temperature(model.driver.soil, p, Y, t, z)
        θ_l = soil_moisture(model.driver.soil, p, Y, t, z)
        Csom = soil_SOM_C(model.driver.soil, p, Y, t, z)
        P_sfc = p.drivers.P
        θ_w = θ_l
        ν = model.driver.soil.ν
        θ_a100 = model.driver.soil.θ_a100
        b = model.driver.soil.b
        p.soilco2.D .=
            co2_diffusivity.(T_soil, θ_w, P_sfc, θ_a100, b, ν, params)
        p.soilco2.Sm .= microbe_source.(T_soil, θ_l, Csom, ν, params)
    end
    return update_aux!
end

# Boundary condition types/methods for Soil CO2
"""
    SoilCO2FluxBC <: ClimaLand.AbstractBC

A container holding the CO2 flux boundary condition,
which is a function `f(p,t)`, where `p` is the auxiliary state
vector.
"""
struct SoilCO2FluxBC{F <: Function} <: ClimaLand.AbstractBC
    bc::F
end

"""
    ClimaLand.boundary_flux(
        bc::SoilCO2FluxBC,
        boundary::ClimaLand.AbstractBoundary,
        Δz::ClimaCore.Fields.Field,
        Y::ClimaCore.Fields.FieldVector,
        p::NamedTuple,
        t,
    )::ClimaCore.Fields.Field

A method of ClimaLand.boundary_flux which returns the soilco2
flux (kg CO2 /m^2/s) in the case of a prescribed flux BC at either the top
or bottom of the domain.
"""
function ClimaLand.boundary_flux(
    bc::SoilCO2FluxBC,
    boundary::ClimaLand.AbstractBoundary,
    Δz::ClimaCore.Fields.Field,
    Y::ClimaCore.Fields.FieldVector,
    p::NamedTuple,
    t,
)::ClimaCore.Fields.Field
    FT = eltype(Δz)
    return FT.(bc.bc(p, t)) .+ FT.(ClimaCore.Fields.zeros(axes(Δz)))
end

"""
    SoilCO2StateBC <: ClimaLand.AbstractBC

A container holding the CO2 state boundary condition (kg CO2 m−3),
which is a function `f(p,t)`, where `p` is the auxiliary state
vector.
"""
struct SoilCO2StateBC{F <: Function} <: ClimaLand.AbstractBC
    bc::F
end

"""
    ClimaLand.boundary_flux(
    bc::SoilCO2StateBC,
    boundary::ClimaLand.TopBoundary,
    Δz::ClimaCore.Fields.Field,
    Y::ClimaCore.Fields.FieldVector,
    p::NamedTuple,
    t,
    )::ClimaCore.Fields.Field

A method of ClimaLand.boundary_flux which returns the soilco2
flux in the case of a prescribed state BC at
 top of the domain.
"""
function ClimaLand.boundary_flux(
    bc::SoilCO2StateBC,
    boundary::ClimaLand.TopBoundary,
    Δz::ClimaCore.Fields.Field,
    Y::ClimaCore.Fields.FieldVector,
    p::NamedTuple,
    t,
)::ClimaCore.Fields.Field
    FT = eltype(Δz)
    p_len = Spaces.nlevels(axes(p.soilco2.D))
    # We need to project center values onto the face space
    D_c = Fields.Field(
        Fields.field_values(Fields.level(p.soilco2.D, p_len)),
        axes(Δz),
    )
    C_c = Fields.Field(
        Fields.field_values(Fields.level(Y.soilco2.C, p_len)),
        axes(Δz),
    )
    C_bc = FT.(bc.bc(p, t))
    return ClimaLand.diffusive_flux(D_c, C_bc, C_c, Δz)
end

"""
    ClimaLand.boundary_flux(
        bc::SoilCO2StateBC,
        boundary::ClimaLand.BottomBoundary,
        Δz::ClimaCore.Fields.Field,
        Y::ClimaCore.Fields.FieldVector,
        p::NamedTuple,
        t,
    )::ClimaCore.Fields.Field

A method of ClimaLand.boundary_flux which returns the soilco2
flux in the case of a prescribed state BC at
bottom of the domain.
"""
function ClimaLand.boundary_flux(
    bc::SoilCO2StateBC,
    ::ClimaLand.BottomBoundary,
    Δz::ClimaCore.Fields.Field,
    Y::ClimaCore.Fields.FieldVector,
    p::NamedTuple,
    t,
)::ClimaCore.Fields.Field
    FT = eltype(Δz)
    # We need to project center values onto the face space
    D_c = Fields.Field(
        Fields.field_values(Fields.level(p.soilco2.D, 1)),
        axes(Δz),
    )
    C_c = Fields.Field(
        Fields.field_values(Fields.level(Y.soilco2.C, 1)),
        axes(Δz),
    )
    C_bc = FT.(bc.bc(p, t))
    return ClimaLand.diffusive_flux(D_c, C_c, C_bc, Δz)
end

"""
    AtmosCO2StateBC <: ClimaLand.AbstractBC

Set the CO2 concentration to the atmospheric one.
"""
struct AtmosCO2StateBC <: ClimaLand.AbstractBC end

"""
    ClimaLand.boundary_flux(
    bc::AtmosCO2StateBC,
    boundary::ClimaLand.TopBoundary,
    Δz::ClimaCore.Fields.Field,
    Y::ClimaCore.Fields.FieldVector,
    p::NamedTuple,
    t,
    )::ClimaCore.Fields.Field

A method of ClimaLand.boundary_flux which returns the soilco2 flux in the case when the
atmospheric CO2 is ued at top of the domain.
"""
function ClimaLand.boundary_flux(
    bc::AtmosCO2StateBC,
    boundary::ClimaLand.TopBoundary,
    Δz::ClimaCore.Fields.Field,
    Y::ClimaCore.Fields.FieldVector,
    p::NamedTuple,
    t,
)::ClimaCore.Fields.Field
    p_len = Spaces.nlevels(axes(p.soilco2.D))
    # We need to project center values onto the face space
    D_c = Fields.Field(
        Fields.field_values(Fields.level(p.soilco2.D, p_len)),
        axes(Δz),
    )
    C_c = Fields.Field(
        Fields.field_values(Fields.level(Y.soilco2.C, p_len)),
        axes(Δz),
    )
    C_bc = p.drivers.c_co2
    return ClimaLand.diffusive_flux(D_c, C_bc, C_c, Δz)
end

function ClimaLand.get_drivers(model::SoilCO2Model)
    return (model.driver.atmos, nothing)
end
Base.broadcastable(ps::SoilCO2ModelParameters) = tuple(ps)

include("./co2_parameterizations.jl")

end # module
