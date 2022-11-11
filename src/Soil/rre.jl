"""
    RichardsParameters{FT <: AbstractFloat}

A struct for storing parameters of the `RichardModel`.
$(DocStringExtensions.FIELDS)
"""
struct RichardsParameters{FT <: AbstractFloat}
    "The porosity of the soil (m^3/m^3)"
    ν::FT
    "The van Genuchten parameter α (1/m)"
    vg_α::FT
    "The van Genuchten parameter n"
    vg_n::FT
    "The van Genuchten parameter m"
    vg_m::FT
    "The saturated hydraulic conductivity (m/s)"
    K_sat::FT
    "The specific storativity (1/m)"
    S_s::FT
    "The residual water fraction (m^3/m^3"
    θ_r::FT
end

function RichardsParameters(;
    ν::FT,
    vg_α::FT,
    vg_n::FT,
    vg_m::FT,
    K_sat::FT,
    S_s::FT,
    θ_r::FT,
) where {FT}
    return RichardsParameters{FT}(ν, vg_α, vg_n, vg_m, K_sat, S_s, θ_r)
end

"""
    RichardsModel

A model for simulating the flow of water in a porous medium
by solving the Richardson-Richards Equation.

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
end

"""
    RichardsModel{FT}(;
        parameters::RichardsParameters{FT},
        domain::D,
        boundary_conditions::RREBoundaryConditions{FT},
        sources::Tuple,
    ) where {FT, D}

A constructor for a `RichardsModel`.
"""
function RichardsModel{FT}(;
    parameters::RichardsParameters{FT},
    domain::D,
    boundary_conditions::NamedTuple,
    sources::Tuple,
) where {FT, D}
    args = (parameters, domain, boundary_conditions, sources)
    RichardsModel{FT, typeof.(args)...}(args...)
end


"""
    make_rhs(model::RichardsModel)

An extension of the function `make_rhs`, for the Richardson-
Richards equation. 

This function creates and returns a function which computes the entire
right hand side of the PDE for `ϑ_l`, and updates `dY.soil.ϑ_l` in place
with that value.

This has been written so as to work with Differential Equations.jl.
"""
function ClimaLSM.make_rhs(model::RichardsModel)
    function rhs!(dY, Y, p, t)
        @unpack ν, vg_α, vg_n, vg_m, K_sat, S_s, θ_r = model.parameters
        z = ClimaCore.Fields.coordinate_field(model.domain.space).z
        Δz_top, Δz_bottom = get_Δz(z)

        top_flux_bc = boundary_flux(
            model.boundary_conditions.water.top,
            TopBoundary(),
            Δz_top,
            p,
            model.parameters,
        )
        bot_flux_bc = boundary_flux(
            model.boundary_conditions.water.bottom,
            BottomBoundary(),
            Δz_bottom,
            p,
            model.parameters,
        )

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
            bottom = Operators.SetValue(Geometry.WVector.(bot_flux_bc)),
        )

        # GradC2F returns a Covariant3Vector, so no need to convert.
        @. dY.soil.ϑ_l =
            -(divf2c_water(-interpc2f(p.soil.K) * gradc2f_water(p.soil.ψ + z)))
        # Horizontal contributions
        horizontal_components!(dY, model.domain, model, p, z)

        # Source terms
        for src in model.sources
            ClimaLSM.source!(dY, src, Y, p)
        end

        # This has to come last
        dss!(dY, model.domain)
    end
    return rhs!
end


"""
   horizontal_components!(dY::ClimaCore.Fields.FieldVector,
                          domain::HybridBox,
                          model::RichardsModel,
                          p::ClimaCore.Fields.FieldVector)
Updates dY in place by adding in the tendency terms resulting from
horizontal derivative operators.

In the case of a hybrid box domain, the horizontal contributions are
computed using the WeakDivergence and Gradient operators.
"""
function horizontal_components!(
    dY::ClimaCore.Fields.FieldVector,
    domain::Union{HybridBox, SphericalShell},
    model::RichardsModel,
    p::ClimaCore.Fields.FieldVector,
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
ClimaLSM.prognostic_vars(soil::RichardsModel) = (:ϑ_l,)
ClimaLSM.prognostic_types(soil::RichardsModel{FT}) where {FT} = (FT,)
"""
    auxiliary_vars(soil::RichardsModel)

A function which returns the names of the auxiliary variables 
of `RichardsModel`.

Note that auxiliary variables are not needed for such a simple model.
We could instead compute the conductivity and matric potential within the
rhs function explicitly, rather than compute and store them in the 
auxiliary vector `p`. We did so in this case as a demonstration.
"""
ClimaLSM.auxiliary_vars(soil::RichardsModel) = (:K, :ψ)
ClimaLSM.auxiliary_types(soil::RichardsModel{FT}) where {FT} = (FT, FT)
"""
    make_update_aux(model::RichardsModel)

An extension of the function `make_update_aux`, for the Richardson-
Richards equation. 

This function creates and returns a function which updates the auxiliary
variables `p.soil.variable` in place.

This has been written so as to work with Differential Equations.jl.
"""
function ClimaLSM.make_update_aux(model::RichardsModel)
    function update_aux!(p, Y, t)
        @unpack ν, vg_α, vg_n, vg_m, K_sat, S_s, θ_r = model.parameters
        @. p.soil.K = hydraulic_conductivity(
            K_sat,
            vg_m,
            effective_saturation(ν, Y.soil.ϑ_l, θ_r),
        )
        @. p.soil.ψ = pressure_head(vg_α, vg_n, vg_m, θ_r, Y.soil.ϑ_l, ν, S_s)
    end
    return update_aux!
end
