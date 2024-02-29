"""
    RichardsParameters{F <: Union{<: AbstractFloat, ClimaCore.Fields.Field}, C <: AbstractSoilHydrologyClosure}

A struct for storing parameters of the `RichardModel`.
$(DocStringExtensions.FIELDS)
"""
struct RichardsParameters{
    F <: Union{<:AbstractFloat, ClimaCore.Fields.Field},
    C,
}
    "The porosity of the soil (m^3/m^3)"
    ν::F
    "The hydrology closure model: vanGenuchten or BrooksCorey"
    hydrology_cm::C
    "The saturated hydraulic conductivity (m/s)"
    K_sat::F
    "The specific storativity (1/m)"
    S_s::F
    "The residual water fraction (m^3/m^3"
    θ_r::F
end

function RichardsParameters(;
    hydrology_cm::C,
    ν::F,
    K_sat::F,
    S_s::F,
    θ_r::F,
) where {F <: Union{<:AbstractFloat, ClimaCore.Fields.Field}, C}
    return RichardsParameters{F, typeof(hydrology_cm)}(
        ν,
        hydrology_cm,
        K_sat,
        S_s,
        θ_r,
    )
end

"""
    RichardsModel

A model for simulating the flow of water in a porous medium
by solving the Richardson-Richards Equation.

A variety of boundary condition types are supported, including
FluxBC, RichardsAtmosDrivenFluxBC, MoistureStateBC, and FreeDrainage
(only for the bottom of the domain).

If you wish to
simulate soil hydrology under the context of a prescribed precipitation
volume flux (m/s) as a function of time, the RichardsAtmosDrivenFluxBC
type should be chosen. Please see the documentation for more details.

$(DocStringExtensions.FIELDS)
"""
struct RichardsModel{FT, PS, D, BC, S} <: AbstractSoilModel{FT}
    "the parameter set"
    parameters::PS
    "the soil domain, using ClimaCore.Domains"
    domain::D
    "the boundary conditions, of type AbstractSoilBoundaryConditions"
    boundary_conditions::BC
    "A tuple of sources, each of type AbstractSoilSource"
    sources::S
    "A boolean flag which, when false, turns off the horizontal flow of water"
    lateral_flow::Bool
end

"""
    RichardsModel{FT}(;
        parameters::RichardsParameters,
        domain::D,
        boundary_conditions::NamedTuple,
        sources::Tuple,
        lateral_flow::Bool = true
    ) where {FT, D}

A constructor for a `RichardsModel`, which sets the
default value of `lateral_flow` to be true.
"""
function RichardsModel{FT}(;
    parameters::RichardsParameters,
    domain::D,
    boundary_conditions::NamedTuple,
    sources::Tuple,
    lateral_flow::Bool = true,
) where {FT, D}
    top_bc = boundary_conditions.top
    if typeof(top_bc) <: RichardsAtmosDrivenFluxBC
        # If the top BC indicates precipitation is driving the model,
        # add baseflow as a sink/source term
        subsurface_source = subsurface_runoff_source(top_bc.runoff)
        sources = append_source(subsurface_source, sources)
    end
    args = (parameters, domain, boundary_conditions, sources)
    RichardsModel{FT, typeof.(args)...}(args..., lateral_flow)
end

function make_update_boundary_fluxes(model::RichardsModel)
    function update_boundary_fluxes!(p, Y, t)
        z = ClimaCore.Fields.coordinate_field(model.domain.space.subsurface).z
        Δz_top, Δz_bottom = get_Δz(z)
        p.soil.top_bc .= boundary_flux(
            model.boundary_conditions.top,
            TopBoundary(),
            model,
            Δz_top,
            Y,
            p,
            t,
        )
        p.soil.bottom_bc .= boundary_flux(
            model.boundary_conditions.bottom,
            BottomBoundary(),
            model,
            Δz_bottom,
            Y,
            p,
            t,
        )
    end
    return update_boundary_fluxes!
end

"""
    make_compute_imp_tendency(model::RichardsModel)

An extension of the function `make_compute_imp_tendency`, for the Richardson-
Richards equation.

This function creates and returns a function which computes the entire
right hand side of the PDE for `ϑ_l`, and updates `dY.soil.ϑ_l` in place
with that value.
"""
function ClimaLand.make_compute_imp_tendency(model::RichardsModel)
    function compute_imp_tendency!(dY, Y, p, t)
        z = ClimaCore.Fields.coordinate_field(model.domain.space.subsurface).z
        top_flux_bc = p.soil.top_bc
        bottom_flux_bc = p.soil.bottom_bc

        interpc2f = Operators.InterpolateC2F()
        gradc2f_water = Operators.GradientC2F()

        # We are setting a boundary value on a flux, which is a gradient of a scalar
        # Therefore, we should set boundary conditions in terms of a covariant vector
        # We set the third component first - supply a Covariant3Vector

        # Without topography only
        # In Cartesian coordinates, W (z^) = Cov3 (z^) = Contra3 (n^ = z^)
        # In spherical coordinates, W (r^) = Cov3 (r^) = Contra3 (n^ = r^)

        # It appears that the WVector is converted internally to a Covariant3Vector for the gradient value
        # at the boundary. Offline tests indicate that you get the same thing if
        # the bc is WVector(F) or Covariant3Vector(F*Δr) or Contravariant3Vector(F/Δr)

        divf2c_water = Operators.DivergenceF2C(
            top = Operators.SetValue(Geometry.WVector.(top_flux_bc)),
            bottom = Operators.SetValue(Geometry.WVector.(bottom_flux_bc)),
        )

        # GradC2F returns a Covariant3Vector, so no need to convert.
        @. dY.soil.ϑ_l =
            -(divf2c_water(-interpc2f(p.soil.K) * gradc2f_water(p.soil.ψ + z)))
    end
    return compute_imp_tendency!
end

"""
    make_explicit_tendency(model::Soil.RichardsModel)

An extension of the function `make_compute_imp_tendency`, for the Richardson-
Richards equation.

Construct the tendency computation function for the explicit terms of the RHS,
which are horizontal components and source/sink terms.
"""
function ClimaLand.make_compute_exp_tendency(model::Soil.RichardsModel)
    # Currently, boundary conditions in the horizontal conditions
    # are restricted to be periodic. In this case, the explicit tendency
    # does not depend on boundary fluxes, and we do not need to update
    # the boundary_var variables prior to evaluation.
    function compute_exp_tendency!(dY, Y, p, t)
        # set dY before updating it
        dY.soil.ϑ_l .= eltype(dY.soil.ϑ_l)(0)
        z = ClimaCore.Fields.coordinate_field(model.domain.space.subsurface).z

        horizontal_components!(
            dY,
            model.domain,
            Val(model.lateral_flow),
            model,
            p,
            z,
        )

        # Source terms
        for src in model.sources
            ClimaLand.source!(dY, src, Y, p, model)
        end
    end
    return compute_exp_tendency!
end


"""
   horizontal_components!(dY::ClimaCore.Fields.FieldVector,
                          domain::Union{HybridBox, SphericalShell},
                          lateral_flow::Val{true},
                          model::RichardsModel,
                          p::NamedTuple)

Updates dY in place by adding in the tendency terms resulting from
horizontal derivative operators for the RichardsModel,
 in the case of a hybrid box or
spherical shell domain with the model
`lateral_flag` set to true.

The horizontal contributions are
computed using the WeakDivergence and Gradient operators.
"""
function horizontal_components!(
    dY::ClimaCore.Fields.FieldVector,
    domain::Union{HybridBox, SphericalShell},
    lateral_flow::Val{true},
    model::RichardsModel,
    p::NamedTuple,
    z::ClimaCore.Fields.Field,
)
    hdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()
    # The flux is already covariant, from hgrad, so no need to convert.
    @. dY.soil.ϑ_l += -hdiv(-p.soil.K * hgrad(p.soil.ψ + z))
end

"""
    prognostic_vars(soil::RichardsModel)

A function which returns the names of the prognostic variables
of `RichardsModel`.
"""
ClimaLand.prognostic_vars(soil::RichardsModel) = (:ϑ_l,)
ClimaLand.prognostic_types(soil::RichardsModel{FT}) where {FT} = (FT,)
ClimaLand.prognostic_domain_names(soil::RichardsModel) = (:subsurface,)

"""
    auxiliary_vars(soil::RichardsModel)

A function which returns the names of the auxiliary variables
of `RichardsModel`.
"""
function ClimaLand.auxiliary_vars(soil::RichardsModel)
    return (
        :K,
        :ψ,
        boundary_vars(soil.boundary_conditions.top, ClimaLand.TopBoundary())...,
        boundary_vars(
            soil.boundary_conditions.bottom,
            ClimaLand.BottomBoundary(),
        )...,
    )
end

"""
    auxiliary_domain_names(soil::RichardsModel)

A function which returns the names of the auxiliary domain names
of `RichardsModel`.
"""
function ClimaLand.auxiliary_domain_names(soil::RichardsModel)
    return (
        :subsurface,
        :subsurface,
        boundary_var_domain_names(
            soil.boundary_conditions.top,
            ClimaLand.TopBoundary(),
        )...,
        boundary_var_domain_names(
            soil.boundary_conditions.bottom,
            ClimaLand.BottomBoundary(),
        )...,
    )
end

"""
    auxiliary_types(soil::RichardsModel)

A function which returns the names of the auxiliary types
of `RichardsModel`.
"""
function ClimaLand.auxiliary_types(soil::RichardsModel{FT}) where {FT}
    return (
        FT,
        FT,
        boundary_var_types(
            soil,
            soil.boundary_conditions.top,
            ClimaLand.TopBoundary(),
        )...,
        boundary_var_types(
            soil,
            soil.boundary_conditions.bottom,
            ClimaLand.BottomBoundary(),
        )...,
    )
end

"""
    make_update_aux(model::RichardsModel)

An extension of the function `make_update_aux`, for the Richardson-
Richards equation.

This function creates and returns a function which updates the auxiliary
variables `p.soil.variable` in place.

This has been written so as to work with Differential Equations.jl.
"""
function ClimaLand.make_update_aux(model::RichardsModel)
    function update_aux!(p, Y, t)
        (; ν, hydrology_cm, K_sat, S_s, θ_r) = model.parameters
        @. p.soil.K = hydraulic_conductivity(
            hydrology_cm,
            K_sat,
            effective_saturation(ν, Y.soil.ϑ_l, θ_r),
        )
        @. p.soil.ψ = pressure_head(hydrology_cm, θ_r, Y.soil.ϑ_l, ν, S_s)
    end
    return update_aux!
end


"""
    RichardsTridiagonalW{R, J, W, T} <: ClimaLand.AbstractTridiagonalW

A struct containing the necessary information for constructing a tridiagonal
Jacobian matrix (`W`) for solving Richards equation, treating only the vertical
diffusion term implicitly.

Note that the diagonal, upper diagonal, and lower diagonal entry values
are stored in this struct and updated in place.
$(DocStringExtensions.FIELDS)
"""
struct RichardsTridiagonalW{R, J, JA, T, A} <: ClimaLand.AbstractTridiagonalW
    "Reference to dtγ, which is specified by the ODE solver"
    dtγ_ref::R
    "Diagonal entries of the Jacobian stored as a ClimaCore.Fields.Field"
    ∂ϑₜ∂ϑ::J
    "Array of tridiagonal matrices containing W for each column"
    W_column_arrays::JA
    "An allocated cache used to evaluate ldiv!"
    temp1::T
    "An allocated cache used to evaluate ldiv!"
    temp2::T
    "A flag indicating whether this struct is used to compute Wfact_t or Wfact"
    transform::Bool
    "A pre-allocated cache storing ones on the face space"
    ones_face_space::A
end

"""
    RichardsTridiagonalW(
        Y::ClimaCore.Fields.FieldVector;
        transform::Bool = false
)

Outer constructor for the RichardsTridiagonalW Jacobian
matrix struct.

Initializes all variables to zeros.
"""
function RichardsTridiagonalW(
    Y::ClimaCore.Fields.FieldVector;
    transform::Bool = false,
)
    FT = eltype(Y.soil.ϑ_l)
    center_space = axes(Y.soil.ϑ_l)
    N = Spaces.nlevels(center_space)

    tridiag_type = Operators.StencilCoefs{-1, 1, NTuple{3, FT}}
    ∂ϑₜ∂ϑ = Fields.Field(tridiag_type, center_space)

    fill!(parent(∂ϑₜ∂ϑ), FT(0))

    W_column_arrays = [
        LinearAlgebra.Tridiagonal(
            zeros(FT, N - 1) .+ FT(0),
            zeros(FT, N) .+ FT(0),
            zeros(FT, N - 1) .+ FT(0),
        ) for _ in 1:Threads.nthreads()
    ]
    dtγ_ref = Ref(FT(0))
    temp1 = similar(Y.soil.ϑ_l)
    temp1 .= FT(0)
    temp2 = similar(Y.soil.ϑ_l)
    temp2 .= FT(0)

    face_space = ClimaLand.Domains.obtain_face_space(center_space)
    ones_face_space = ones(face_space)

    return RichardsTridiagonalW(
        dtγ_ref,
        ∂ϑₜ∂ϑ,
        W_column_arrays,
        temp1,
        temp2,
        transform,
        ones_face_space,
    )
end


"""
    ClimaLand.make_update_jacobian(model::RichardsModel{FT}) where {FT}

Creates and returns the update_jacobian! function for RichardsModel.

Using this Jacobian with a backwards Euler timestepper is equivalent
to using the modified Picard scheme of Celia et al. (1990).
"""
function ClimaLand.make_update_jacobian(model::RichardsModel{FT}) where {FT}
    update_aux! = make_update_aux(model)
    update_boundary_fluxes! = make_update_boundary_fluxes(model)
    function update_jacobian!(W::RichardsTridiagonalW, Y, p, dtγ, t)
        (; dtγ_ref, ∂ϑₜ∂ϑ) = W
        dtγ_ref[] = dtγ
        (; ν, hydrology_cm, S_s, θ_r) = model.parameters
        update_aux!(p, Y, t)
        update_boundary_fluxes!(p, Y, t)
        divf2c_op = Operators.DivergenceF2C(
            top = Operators.SetValue(Geometry.WVector.(FT(0))),
            bottom = Operators.SetValue(Geometry.WVector.(FT(0))),
        )
        divf2c_stencil = Operators.Operator2Stencil(divf2c_op)
        gradc2f_op = Operators.GradientC2F(
            top = Operators.SetGradient(Geometry.WVector.(FT(0))),
            bottom = Operators.SetGradient(Geometry.WVector.(FT(0))),
        )
        gradc2f_stencil = Operators.Operator2Stencil(gradc2f_op)
        interpc2f_op = Operators.InterpolateC2F(
            bottom = Operators.Extrapolate(),
            top = Operators.Extrapolate(),
        )
        compose = Operators.ComposeStencils()

        @. ∂ϑₜ∂ϑ = compose(
            divf2c_stencil(Geometry.Covariant3Vector(W.ones_face_space)),
            (
                interpc2f_op(p.soil.K) * ClimaLand.to_scalar_coefs(
                    gradc2f_stencil(
                        ClimaLand.Soil.dψdϑ(
                            hydrology_cm,
                            Y.soil.ϑ_l,
                            ν,
                            θ_r,
                            S_s,
                        ),
                    ),
                )
            ),
        )
        # Hardcoded for single column: FIX!
        # The derivative of the tendency may eventually live in boundary vars and be updated there. but TBD
        z = Fields.coordinate_field(axes(Y.soil.ϑ_l)).z
        Δz_top, Δz_bottom = get_Δz(z)
        ∂T_bc∂YN = ClimaLand.∂tendencyBC∂Y(
            model,
            model.boundary_conditions.top.water,
            ClimaLand.TopBoundary(),
            Δz_top,
            Y,
            p,
            t,
        )
        #TODO: allocate space in W? See how final implementation of stencils with boundaries works out
        N = ClimaCore.Spaces.nlevels(axes(Y.soil.ϑ_l))
        parent(ClimaCore.Fields.level(∂ϑₜ∂ϑ.coefs.:2, N)) .=
            parent(ClimaCore.Fields.level(∂ϑₜ∂ϑ.coefs.:2, N)) .+
            parent(∂T_bc∂YN.soil.ϑ_l)

    end
    return update_jacobian!
end

"""
    ClimaLand.get_drivers(model::RichardsModel)

Returns the driver variable symbols for the RichardsModel; these
depend on the boundary condition type and currently only are required
for the RichardsAtmosDrivenFluxBC, which is driven by
a prescribed time and space varying precipitation.
"""
function ClimaLand.get_drivers(model::RichardsModel)
    bc = model.boundary_conditions.top
    if typeof(bc) <: RichardsAtmosDrivenFluxBC{
        <:PrescribedPrecipitation,
        <:AbstractRunoffModel,
    }
        return (bc.precip, nothing)
    else
        return (nothing, nothing)
    end
end
