import ClimaLand:
    AbstractBoundary,
    AbstractBC,
    boundary_flux!,
    boundary_vars,
    boundary_var_domain_names,
    boundary_var_types
using ClimaLand: Domains
using ClimaCore: Geometry
export TemperatureStateBC,
    MoistureStateBC,
    FreeDrainage,
    HeatFluxBC,
    WaterFluxBC,
    AtmosDrivenFluxBC,
    RichardsAtmosDrivenFluxBC,
    WaterHeatBC,
    sublimation_source


# New BC type for Richards Equation (AbstractWaterBC)
"""
    AbstractWaterBC <: ClimaLand.AbstractBC

An abstract type for boundary conditions for Richards equation.
"""
abstract type AbstractWaterBC <: ClimaLand.AbstractBC end

"""
   MoistureStateBC <: AbstractWaterBC

A simple concrete type of boundary condition, which enforces a
state boundary condition ϑ_l = f(p,t) at either the top or bottom of the domain.
"""
struct MoistureStateBC{F <: Function} <: AbstractWaterBC
    bc::F
end

"""
   WaterFluxBC <: AbstractWaterBC

A simple concrete type of boundary condition, which enforces a
normal flux boundary condition f(p,t) at either the top or bottom of the domain.
"""
struct WaterFluxBC{F <: Function} <: AbstractWaterBC
    bc::F
end

"""
    FreeDrainage <: AbstractWaterBC
A concrete type of soil boundary condition, for use at
the BottomBoundary only, where the flux is set to be
`F = -K∇h = -K`.
"""
struct FreeDrainage <: AbstractWaterBC end


"""
   RichardsAtmosDrivenFluxBC{F <: PrescribedPrecipitation, R <: AbstractRunoffModel} <: AbstractWaterBC

A concrete type of boundary condition intended only for use with the RichardsModel,
which uses a prescribed precipitation rate (m/s) to compute the infiltration
into the soil.

A runoff model is used
to simulate surface and subsurface runoff and this is accounted
for when setting boundary conditions. In order to run the simulation
*without* runoff, choose runoff = NoRunoff() - this is also the default.

If you wish to simulate precipitation and runoff in the full `EnergyHydrology` model,
you must use the `AtmosDrivenFluxBC` type.
$(DocStringExtensions.FIELDS)
"""
struct RichardsAtmosDrivenFluxBC{
    F <: PrescribedPrecipitation,
    R <: AbstractRunoffModel,
} <: AbstractWaterBC
    "The prescribed liquid water precipitation rate f(t) (m/s); Negative by convention."
    precip::F
    "The runoff model. The default is no runoff."
    runoff::R
end

function RichardsAtmosDrivenFluxBC(
    precip::PrescribedPrecipitation;
    runoff = NoRunoff(),
)
    if typeof(runoff) <: NoRunoff
        @info("Warning: No runoff model was provided; zero runoff generated.")
    end

    return RichardsAtmosDrivenFluxBC{typeof(precip), typeof(runoff)}(
        precip,
        runoff,
    )
end

# Methods
"""
    boundary_vars(::RichardsAtmosDrivenFluxBC{<:PrescribedPrecipitation,
                                              <:Runoff.AbstractRunoffModel,
                                              }, ::ClimaLand.TopBoundary)

An extension of the `boundary_vars` method for RichardsAtmosDrivenFluxBC with
runoff.

These variables are updated in place in `boundary_flux!`.
"""
boundary_vars(
    bc::RichardsAtmosDrivenFluxBC{
        <:PrescribedPrecipitation,
        <:Runoff.AbstractRunoffModel,
    },
    ::ClimaLand.TopBoundary,
) = (:top_bc, :top_bc_wvec, Runoff.runoff_vars(bc.runoff)...)

"""
    boundary_var_domain_names(::RichardsAtmosDrivenFluxBC{<:PrescribedPrecipitation,
                                              <:Runoff.AbstractRunoffModel,
                                              },
                              ::ClimaLand.TopBoundary)

An extension of the `boundary_var_domain_names` method for RichardsAtmosDrivenFluxBC
with runoff.
"""
boundary_var_domain_names(
    bc::RichardsAtmosDrivenFluxBC{
        <:PrescribedPrecipitation,
        <:Runoff.AbstractRunoffModel,
    },
    ::ClimaLand.TopBoundary,
) = (:surface, :surface, Runoff.runoff_var_domain_names(bc.runoff)...)
"""
    boundary_var_types(::RichardsModel{FT},
                        ::RichardsAtmosDrivenFluxBC{<:PrescribedPrecipitation,
                                                    <: Runoff.AbstractRunoffModel,
                                                  },
                        ::ClimaLand.TopBoundary,
                        ) where {FT}

An extension of the `boundary_var_types` method for RichardsAtmosDrivenFluxBC
with runoff.
"""
boundary_var_types(
    model::RichardsModel{FT},
    bc::RichardsAtmosDrivenFluxBC{
        <:PrescribedPrecipitation,
        <:Runoff.AbstractRunoffModel,
    },
    ::ClimaLand.TopBoundary,
) where {FT} = (
    FT,
    ClimaCore.Geometry.WVector{FT},
    Runoff.runoff_var_types(bc.runoff, FT)...,
)


"""
    boundary_flux!(bc_field, bc::WaterFluxBC,  _...)

A method of boundary fluxes which updates the desired flux.
"""
function boundary_flux!(
    bc_field,
    bc::WaterFluxBC,
    boundary::ClimaLand.AbstractBoundary,
    model,
    Δz::ClimaCore.Fields.Field,
    Y::ClimaCore.Fields.FieldVector,
    p::NamedTuple,
    t,
)
    FT = eltype(Δz)
    bc_field .= FT.(bc.bc(p, t))
end


"""
    boundary_flux!(bc_field, bc::RichardsAtmosDrivenFluxBC,
                           boundary::ClimaLand.AbstractBoundary,
                           model::RichardsModel{FT},
                           Δz::ClimaCore.Fields.Field,
                           Y::ClimaCore.Fields.FieldVector,
                           p::NamedTuple,
                           t,
                           ) where {FT}

A method of boundary fluxes which returns the desired water volume flux for
the RichardsModel, at the top of the domain, in the case of a prescribed
precipitation flux.

If `model.runoff` is not of type `NoRunoff`, surface runoff is accounted for
when computing the infiltration.
"""
function boundary_flux!(
    bc_field,
    bc::RichardsAtmosDrivenFluxBC,
    boundary::ClimaLand.AbstractBoundary,
    model::RichardsModel,
    Δz::ClimaCore.Fields.Field,
    Y::ClimaCore.Fields.FieldVector,
    p::NamedTuple,
    t,
)
    update_runoff!(p, bc.runoff, p.drivers.P_liq, Y, t, model)
    bc_field .= p.soil.infiltration
end

"""
    boundary_flux!(bc_field, rre_bc::MoistureStateBC,
                           ::ClimaLand.TopBoundary,
                           model::AbstractSoilModel,
                           Δz::ClimaCore.Fields.Field,
                           Y::ClimaCore.Fields.FieldVector,
                           p::NamedTuple,
                           t,
                           )

A method of boundary fluxes which converts a state boundary condition on θ_l at the top of the
domain into a flux of liquid water.
"""
function boundary_flux!(
    bc_field,
    rre_bc::MoistureStateBC,
    ::ClimaLand.TopBoundary,
    model::AbstractSoilModel,
    Δz::ClimaCore.Fields.Field,
    Y::ClimaCore.Fields.FieldVector,
    p::NamedTuple,
    t,
)
    FT = eltype(Δz)
    # First extract the value of the top layer of the pressure head
    # and cast onto the face space
    ψ_c = ClimaLand.Domains.top_center_to_surface(p.soil.ψ)

    # Calculate pressure head using boundary condition on ϑ_l = θ_bc
    # We first need to extract the parameters of the soil in the top layer
    # Again, we need to cast them onto the face space
    (; hydrology_cm, θ_r, ν, S_s) = model.parameters
    hcm_bc = ClimaLand.Domains.top_center_to_surface(hydrology_cm)
    θ_r_bc = ClimaLand.Domains.top_center_to_surface(θ_r)
    ν_bc = ClimaLand.Domains.top_center_to_surface(ν)
    S_s_bc = ClimaLand.Domains.top_center_to_surface(S_s)

    θ_bc = FT.(rre_bc.bc(p, t))

    # Lastly, we need an effective conductivity to compute the flux that results from
    # the gradient in pressure.
    # currently we approximate this as equal to the center value at the top layer (K_c)
    # More accurate would be to compute the mean between K_c and K evaluated at the boundary
    # condition.
    K_eff = ClimaLand.Domains.top_center_to_surface(p.soil.K)

    # Pass in (ψ_bc .+ Δz) as to account for contribution of gravity (∂(ψ+z)/∂z
    @. bc_field = ClimaLand.diffusive_flux(
        K_eff,
        pressure_head(hcm_bc, θ_r_bc, θ_bc, ν_bc, S_s_bc) + Δz,
        ψ_c,
        Δz,
    )
end

"""
    boundary_flux!(bc_field, rre_bc::MoistureStateBC,
                           ::ClimaLand.BottomBoundary,
                           model::AbstractSoilModel,
                           Δz::ClimaCore.Fields.Field,
                           Y::ClimaCore.Fields.FieldVector,
                           p::NamedTuple,
                           t,
                           )

A method of boundary fluxes which converts a state boundary condition on θ_l at the bottom of the
domain into a flux of liquid water.
"""
function boundary_flux!(
    bc_field,
    rre_bc::MoistureStateBC,
    ::ClimaLand.BottomBoundary,
    model::AbstractSoilModel,
    Δz::ClimaCore.Fields.Field,
    Y::ClimaCore.Fields.FieldVector,
    p::NamedTuple,
    t,
)
    FT = eltype(Δz)
    # First extract the value of the bottom layer of the pressure head
    # and cast onto the face space
    ψ_c = ClimaLand.Domains.bottom_center_to_surface(p.soil.ψ)


    # Calculate pressure head using boundary condition on ϑ_l = θ_bc
    # We first need to extract the parameters of the soil in the bottom layer
    # Again, we need to cast them onto the face space
    (; hydrology_cm, θ_r, ν, S_s) = model.parameters
    hcm_bc = ClimaLand.Domains.bottom_center_to_surface(hydrology_cm)
    θ_r_bc = ClimaLand.Domains.bottom_center_to_surface(θ_r)
    ν_bc = ClimaLand.Domains.bottom_center_to_surface(ν)
    S_s_bc = ClimaLand.Domains.bottom_center_to_surface(S_s)

    θ_bc = FT.(rre_bc.bc(p, t))

    # Lastly, we need an effective conductivity to compute the flux that results from
    # the gradient in pressure.
    # currently we approximate this as equal to the center value at the top layer (K_c)
    # More accurate would be to compute the mean between K_c and K evaluated at the boundary
    # condition.
    K_eff = ClimaLand.Domains.bottom_center_to_surface(p.soil.K)

    # At the bottom boundary, ψ_c is at larger z than ψ_bc
    #  so we swap their order in the derivative calc
    # Pass in (ψ_c .+ Δz)  to account for contribution of gravity  (∂(ψ+z)/∂z
    @. bc_field = ClimaLand.diffusive_flux(
        K_eff,
        ψ_c + Δz,
        pressure_head(hcm_bc, θ_r_bc, θ_bc, ν_bc, S_s_bc),
        Δz,
    )
end

"""
    boundary_flux!(bc_field, bc::FreeDrainage,
                           boundary::ClimaLand.BottomBoundary,
                           model::AbstractSoilModel,
                           Δz::ClimaCore.Fields.Field,
                           Y::ClimaCore.Fields.FieldVector,
                           p::NamedTuple,
                           t,
                           )

A method of boundary fluxes which enforces free drainage at the bottom
of the domain.
"""
function boundary_flux!(
    bc_field,
    bc::FreeDrainage,
    boundary::ClimaLand.BottomBoundary,
    model::AbstractSoilModel,
    Δz::ClimaCore.Fields.Field,
    Y::ClimaCore.Fields.FieldVector,
    p::NamedTuple,
    t,
)
    FT = eltype(Δz)
    K_c = Fields.level(p.soil.K, 1)
    @. bc_field = -1 * K_c
end

"""
    ClimaLand.set_dfluxBCdY!(
        model::RichardsModel,
        ::MoistureStateBC,
        boundary::ClimaLand.TopBoundary,
        Δz,
        Y,
        p,
        t,
)

Computes the derivative of the flux in the top layer (due to the
boundary condition), with respect to the state variable in the top layer.
This value is then updated in-place in the cache.

For Richards equation (a diffusion equation with a single state variable),
this is given by `∂F_bc/∂Y_N= -K_N (∂ψ_bc/∂ϑ_N) / Δz`,
where `N` indicates the top layer cell index and
`ψ_bc` is the pressure head at the boundary condition.
"""
function ClimaLand.set_dfluxBCdY!(
    model::RichardsModel,
    ::MoistureStateBC,
    boundary::ClimaLand.TopBoundary,
    Δz,
    Y,
    p,
    t,
)
    (; ν, hydrology_cm, S_s, θ_r) = model.parameters

    # Copy center variables to top face space
    KN = Domains.top_center_to_surface(p.soil.K)
    hydrology_cmN = Domains.top_center_to_surface(hydrology_cm)
    ϑ_lN = Domains.top_center_to_surface(Y.soil.ϑ_l)
    νN = Domains.top_center_to_surface(ν)
    θ_rN = Domains.top_center_to_surface(θ_r)
    S_sN = Domains.top_center_to_surface(S_s)


    # Get the local geometry of the face space, then extract the top level
    levels = ClimaCore.Spaces.nlevels(Domains.obtain_face_space(axes(p.soil.K)))
    local_geometry_faceN = ClimaCore.Fields.level(
        Fields.local_geometry_field(Domains.obtain_face_space(axes(p.soil.K))),
        levels - ClimaCore.Utilities.half,
    )

    # Update dfluxBCdY at the top boundary in place
    # Calculate the value and convert it to a Covariant3Vector
    @. p.soil.dfluxBCdY =
        covariant3_unit_vector(local_geometry_faceN) *
        (KN * dψdϑ(hydrology_cmN, ϑ_lN, νN, θ_rN, S_sN) / Δz)
    return nothing
end

"""
    covariant3_unit_vector(local_geometry)

A function to compute the unit vector in the direction of the normal
to the surface.

Adapted from ClimaAtmos.jl's unit_basis_vector_data function.
"""
function covariant3_unit_vector(local_geometry)
    FT = Geometry.undertype(typeof(local_geometry))
    data =
        FT(1) / Geometry._norm(Geometry.Covariant3Vector(FT(1)), local_geometry)
    return Geometry.Covariant3Vector(data)
end

# BC type for the soil heat equation
"""
    AbstractHeatBC <: ClimaLand.AbstractBC

An abstract type for boundary conditions for the soil heat equation.
"""
abstract type AbstractHeatBC <: ClimaLand.AbstractBC end

"""
   TemperatureStateBC <: AbstractHeatBC

A simple concrete type of boundary condition, which enforces a
state boundary condition T = f(p,t) at either the top or bottom of the domain.
"""
struct TemperatureStateBC{F <: Function} <: AbstractHeatBC
    bc::F
end

"""
   HeatFluxBC <: AbstractHeatBC

A simple concrete type of boundary condition, which enforces a
normal flux boundary condition f(p,t) at either the top or bottom of the domain.
"""
struct HeatFluxBC{F <: Function} <: AbstractHeatBC
    bc::F
end


# Methods

"""
    boundary_flux!(bc_field, bc::HeatFluxBC,  _...)

A method of boundary fluxes which updates the desired flux.

"""
function boundary_flux!(
    bc_field,
    bc::HeatFluxBC,
    boundary::ClimaLand.AbstractBoundary,
    model,
    Δz::ClimaCore.Fields.Field,
    Y::ClimaCore.Fields.FieldVector,
    p::NamedTuple,
    t,
)
    FT = eltype(Δz)
    bc_field .= FT.(bc.bc(p, t))
end



"""
    boundary_flux!(bc_field, heat_bc::TemperatureStateBC,
                           ::ClimaLand.TopBoundary,
                           model::EnergyHydrology,
                           Δz::ClimaCore.Fields.Field,
                           Y::ClimaCore.Fields.FieldVector,
                           p::NamedTuple,
                           t,
                           ):

A method of boundary fluxes which converts a state boundary condition on temperature at the top of the
domain into a flux of energy.
"""
function boundary_flux!(
    bc_field,
    heat_bc::TemperatureStateBC,
    ::ClimaLand.TopBoundary,
    model::EnergyHydrology,
    Δz::ClimaCore.Fields.Field,
    Y::ClimaCore.Fields.FieldVector,
    p::NamedTuple,
    t,
)
    FT = eltype(Δz)
    # Approximate κ_bc ≈ κ_c (center closest to the boundary)
    # We need to project the center values onto the face space.
    T_c = ClimaLand.Domains.top_center_to_surface(p.soil.T)
    κ_c = ClimaLand.Domains.top_center_to_surface(p.soil.κ)

    T_bc = FT.(heat_bc.bc(p, t))
    @. bc_field = ClimaLand.diffusive_flux(κ_c, T_bc, T_c, Δz)
end

"""
    boundary_flux!(bc_field, heat_bc::TemperatureStateBC,
                           ::ClimaLand.BottomBoundary,
                           model::EnergyHydrology,
                           Δz::ClimaCore.Fields.Field,
                           Y::ClimaCore.Fields.FieldVector,
                           p::NamedTuple,
                           t,
                           )

A method of boundary fluxes which converts a state boundary condition on temperature at the bottom of the
domain into a flux of energy.
"""
function boundary_flux!(
    bc_field,
    heat_bc::TemperatureStateBC,
    ::ClimaLand.BottomBoundary,
    model::EnergyHydrology,
    Δz::ClimaCore.Fields.Field,
    Y::ClimaCore.Fields.FieldVector,
    p::NamedTuple,
    t,
)
    FT = eltype(Δz)
    # Approximate κ_bc ≈ κ_c (center closest to the boundary)
    # We need to project the center values onto the face space.
    T_c = ClimaLand.Domains.bottom_center_to_surface(p.soil.T)
    κ_c = ClimaLand.Domains.bottom_center_to_surface(p.soil.κ)
    T_bc = FT.(heat_bc.bc(p, t))
    @. bc_field = ClimaLand.diffusive_flux(κ_c, T_c, T_bc, Δz)
end


# BC type for the combined energy-hydrology equations
"""
    AbstractEnergyHydrologyBC <: ClimaLand.AbstractBC

An abstract type for boundary conditions for the combined energy
and hydrology equations.

In general these can consist of independent boundary conditions for
water (<: AbstractWaterBC) and for heat (<: AbstractHeatBC) independently,
, or a boundary condition type which sets both using the same parameterizations.
"""
abstract type AbstractEnergyHydrologyBC <: ClimaLand.AbstractBC end

"""
    WaterHeatBC{W <: AbstractWaterBC, H <: AbstractHeatBC} <:
       AbstractEnergyHydrologyBC

A general struct used to store the boundary conditions for Richards
and the soil heat equations separately; useful when the boundary
conditions for each component are independent of each other.
"""
struct WaterHeatBC{W <: AbstractWaterBC, H <: AbstractHeatBC} <:
       AbstractEnergyHydrologyBC
    water::W
    heat::H
end
function WaterHeatBC(; water, heat)
    return WaterHeatBC{typeof(water), typeof(heat)}(water, heat)
end

"""
    AtmosDrivenFluxBC{
        A <: AbstractAtmosphericDrivers,
        B <: AbstractRadiativeDrivers,
        R <: AbstractRunoffModel,
        C::Tuple
    } <: AbstractEnergyHydrologyBC

A concrete type of soil boundary condition for use at the top
of the domain. This holds the conditions for the atmosphere
`AbstractAtmosphericDrivers`, for the radiation state
`AbstractRadiativeDrivers`. This is only supported for the
`EnergyHydrology` model.

This choice indicates the Monin-Obukhov Surface Theory will
be used to compute the sensible and latent heat fluxes, as
well as evaporation,
 and that the net radiation and precipitation
will also be computed. The net energy and water fluxes
are used as boundary conditions.

A runoff model is used
to simulate surface and subsurface runoff and this is accounted
for when setting boundary conditions. The default is to have no runoff
accounted for.

Finally, because this same boundary condition type is used for the soil
in integrated land surface models, we also provide a tuple of symbols
indicating the prognostic land components present, as these affect how the boundary
conditions are computed. The default is
a tuple containing only (:soil,), indicating a standalone soil run.

For more information on the allowed values, please see
the [documentation](https://clima.github.io/ClimaLand.jl/dev/generated/integrated/handling_soil_fluxes)
$(DocStringExtensions.FIELDS)
"""
struct AtmosDrivenFluxBC{
    A <: AbstractAtmosphericDrivers,
    B <: AbstractRadiativeDrivers,
    R <: AbstractRunoffModel,
    C <: Tuple,
} <: AbstractEnergyHydrologyBC
    "The atmospheric conditions driving the model"
    atmos::A
    "The radiative fluxes driving the model"
    radiation::B
    "The runoff model. The default is no runoff."
    runoff::R
    "Prognostic land components present"
    prognostic_land_components::C
end

function AtmosDrivenFluxBC(
    atmos,
    radiation,
    runoff;
    prognostic_land_components = (:soil,),
)
    args = (atmos, radiation, runoff, prognostic_land_components)
    return AtmosDrivenFluxBC{typeof.(args)...}(args...)
end

function AtmosDrivenFluxBC(
    atmos,
    radiation;
    runoff = NoRunoff(),
    prognostic_land_components = (:soil,),
)
    if typeof(runoff) <: NoRunoff
        @info("Warning: No runoff model was provided; zero runoff generated.")
    end
    args = (atmos, radiation, runoff, prognostic_land_components)
    return AtmosDrivenFluxBC{typeof.(args)...}(args...)
end

# Special methods for EnergyHydrology BC
"""
    boundary_var_types(::Soil.EnergyHydrology{FT}, ::AbstractEnergyHydrologyBC, ::ClimaLand.AbstractBoundary) where {FT}

The list of domain names for additional variables added to the
EnergyHydrology model auxiliary state, which defaults to adding storage for the
 boundary flux field.

Because we supply boundary conditions for water and heat, we found it convenient to
have these stored as a NamedTuple under the names `top_bc` and `bottom_bc`.
"""
function boundary_var_types(
    ::Soil.EnergyHydrology{FT},
    ::AbstractEnergyHydrologyBC,
    ::ClimaLand.AbstractBoundary,
) where {FT}
    (NamedTuple{(:water, :heat), Tuple{FT, FT}}, ClimaCore.Geometry.WVector{FT})
end

"""
    soil_boundary_fluxes!(bc::WaterHeatBC, boundary::AbstractBoundary, model, Δz, Y, p, t)

updates the boundary fluxes for ϑ_l and ρe_int.
"""
function soil_boundary_fluxes!(
    bc::WaterHeatBC,
    boundary::AbstractBoundary,
    model,
    Δz,
    Y,
    p,
    t,
)
    params = model.parameters
    name = ClimaLand.bc_name(boundary)
    water_bc = getproperty(p.soil, name).water
    heat_bc = getproperty(p.soil, name).heat
    boundary_flux!(water_bc, bc.water, boundary, model, Δz, Y, p, t)
    boundary_flux!(heat_bc, bc.heat, boundary, model, Δz, Y, p, t)
end

"""
    boundary_vars(::AtmosDrivenFluxBC{<:AbstractAtmosphericDrivers,
                                    <:AbstractRadiativeDrivers,
                                    <:AbstractRunoffModel,
                                    }, ::ClimaLand.TopBoundary)

An extension of the `boundary_vars` method for AtmosDrivenFluxBC. This
adds the surface conditions (SHF, LHF, evaporation, and resistance) and the
net radiation to the auxiliary variables.

These variables are updated in place in `soil_boundary_fluxes!`.
"""
boundary_vars(bc::AtmosDrivenFluxBC, ::ClimaLand.TopBoundary) = (
    :turbulent_fluxes,
    :R_n,
    :top_bc,
    :top_bc_wvec,
    :sfc_scratch,
    :PAR_albedo,
    :NIR_albedo,
    :sfc_S_e,
    :sub_sfc_scratch,
    Runoff.runoff_vars(bc.runoff)...,
)

"""
    boundary_var_domain_names(::AtmosDrivenFluxBC,
                              ::ClimaLand.TopBoundary)

An extension of the `boundary_var_domain_names` method for AtmosDrivenFluxBC. This
specifies the part of the domain on which the additional variables should be
defined.
"""
boundary_var_domain_names(bc::AtmosDrivenFluxBC, ::ClimaLand.TopBoundary) = (
    :surface,
    :surface,
    :surface,
    :surface,
    :surface,
    :surface,
    :surface,
    :surface,
    :subsurface,
    Runoff.runoff_var_domain_names(bc.runoff)...,
)
"""
    boundary_var_types(
        ::EnergyHydrology{FT},
        ::AtmosDrivenFluxBC,
        ::ClimaLand.TopBoundary,
    ) where {FT}

An extension of the `boundary_var_types` method for AtmosDrivenFluxBC. This
specifies the type of the additional variables.
"""
boundary_var_types(
    model::EnergyHydrology{FT},
    bc::AtmosDrivenFluxBC,
    ::ClimaLand.TopBoundary,
) where {FT} = (
    NamedTuple{
        (:lhf, :shf, :vapor_flux_liq, :r_ae, :vapor_flux_ice),
        Tuple{FT, FT, FT, FT, FT},
    },
    FT,
    NamedTuple{(:water, :heat), Tuple{FT, FT}},
    ClimaCore.Geometry.WVector{FT},
    FT,
    FT,
    FT,
    FT,
    FT,
    Runoff.runoff_var_types(bc.runoff, FT)...,
)

"""
    soil_boundary_fluxes!(
        bc::AtmosDrivenFluxBC{
            <:PrescribedAtmosphere,
            <:PrescribedRadiativeFluxes,
        },
        boundary::ClimaLand.TopBoundary,
        model::EnergyHydrology,
        Δz,
        Y,
        p,
        t,
    )

Returns the net volumetric water flux (m/s) and net energy
flux (W/m^2) for the soil `EnergyHydrology` model at the top
of the soil domain.

This function calls the `turbulent_fluxes` and `net_radiation`
functions, which use the soil surface conditions as well as
the atmos and radiation conditions in order to
compute the surface fluxes using Monin Obukhov Surface Theory.
It also accounts for the presence of other components, if run as
part of an integrated land model, and their effect on boundary conditions.
"""
function soil_boundary_fluxes!(
    bc::AtmosDrivenFluxBC,
    boundary::ClimaLand.TopBoundary,
    soil::EnergyHydrology{FT},
    Δz,
    Y,
    p,
    t,
) where {FT}
    soil_boundary_fluxes!(bc, Val(bc.prognostic_land_components), soil, Y, p, t)
end


"""
    soil_boundary_fluxes!(
        bc::AtmosDrivenFluxBC{
            <:PrescribedAtmosphere,
            <:PrescribedRadiativeFluxes,
        },
        prognostic_land_components::Val{(:soil,)},
        model::EnergyHydrology,
        Y,
        p,
        t,
    )

Returns the net volumetric water flux (m/s) and net energy
flux (W/m^2) for the soil `EnergyHydrology` model at the top
of the soil domain.

Here, the soil boundary fluxes are computed as if the soil is run
in standalone mode.
"""
function soil_boundary_fluxes!(
    bc::AtmosDrivenFluxBC{<:PrescribedAtmosphere, <:PrescribedRadiativeFluxes},
    prognostic_land_components::Val{(:soil,)},
    model::EnergyHydrology,
    Y,
    p,
    t,
)
    p.soil.turbulent_fluxes .= turbulent_fluxes(bc.atmos, model, Y, p, t)
    p.soil.R_n .= net_radiation(bc.radiation, model, Y, p, t)
    # influx = maximum possible rate of infiltration given precip, snowmelt, evaporation/condensation
    # but if this exceeds infiltration capacity of the soil, runoff will
    # be generated.
    # Use top_bc.water as temporary storage to avoid allocation
    influx = p.soil.top_bc.water
    @. influx = p.drivers.P_liq + p.soil.turbulent_fluxes.vapor_flux_liq
    # The update_runoff! function computes how much actually infiltrates
    # given influx and our runoff model bc.runoff, and updates
    # p.soil.infiltration in place
    update_runoff!(p, bc.runoff, influx, Y, t, model)
    # We do not model the energy flux from infiltration.
    @. p.soil.top_bc.water = p.soil.infiltration
    @. p.soil.top_bc.heat =
        p.soil.R_n + p.soil.turbulent_fluxes.lhf + p.soil.turbulent_fluxes.shf
end

"""
    boundary_vars(::MoistureStateBC, ::ClimaLand.TopBoundary)

An extension of the `boundary_vars` method for MoistureStateBC at the
top boundary.

These variables are updated in place in `boundary_flux!`.
"""
boundary_vars(bc::MoistureStateBC, ::ClimaLand.TopBoundary) =
    (:top_bc, :top_bc_wvec, :dfluxBCdY)

"""
    boundary_var_domain_names(::MoistureStateBC, ::ClimaLand.TopBoundary)

An extension of the `boundary_var_domain_names` method for MoistureStateBC at the
top boundary.
"""
boundary_var_domain_names(bc::MoistureStateBC, ::ClimaLand.TopBoundary) =
    (:surface, :surface, :surface)
"""
    boundary_var_types(::RichardsModel{FT},
                        ::MoistureStateBC,
                        ::ClimaLand.TopBoundary,
                        ) where {FT}

An extension of the `boundary_var_types` method for MoistureStateBC at the
    top boundary.
"""
boundary_var_types(
    model::RichardsModel{FT},
    bc::MoistureStateBC,
    ::ClimaLand.TopBoundary,
) where {FT} = (
    FT,
    ClimaCore.Geometry.WVector{FT},
    ClimaCore.Geometry.Covariant3Vector{FT},
)


function sublimation_source(::Val{(:soil,)}, FT)
    return SoilSublimation{FT}()
end
