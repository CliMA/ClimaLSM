module Bucket
using DocStringExtensions
using Thermodynamics
using Dates
using NCDatasets
import ClimaUtilities.TimeVaryingInputs
import ClimaUtilities.DataHandling
import ClimaUtilities.SpaceVaryingInputs: SpaceVaryingInput
import ClimaUtilities.TimeVaryingInputs:
    TimeVaryingInput, AbstractTimeVaryingInput
import Interpolations
using ClimaCore
using ClimaCore.Fields: coordinate_field, level, FieldVector
using ClimaCore.Operators: InterpolateC2F, DivergenceF2C, GradientC2F, SetValue
using ClimaCore.Geometry: WVector
using ClimaComms

using ClimaLand
import ..Parameters as LP
import Thermodynamics.Parameters as TP
import ClimaLand.Domains: coordinates, SphericalShell
using ClimaLand:
    AbstractAtmosphericDrivers,
    AbstractRadiativeDrivers,
    turbulent_fluxes,
    net_radiation,
    compute_ρ_sfc,
    AbstractExpModel,
    heaviside,
    PrescribedAtmosphere,
    add_dss_buffer_to_aux
import ClimaLand:
    make_update_aux,
    make_compute_exp_tendency,
    prognostic_vars,
    auxiliary_vars,
    name,
    prognostic_types,
    auxiliary_types,
    prognostic_domain_names,
    auxiliary_domain_names,
    initialize_vars,
    initialize,
    initialize_auxiliary,
    surface_temperature,
    surface_air_density,
    surface_evaporative_scaling,
    surface_albedo,
    surface_emissivity,
    surface_height,
    get_drivers
export BucketModelParameters,
    BucketModel,
    PrescribedBaregroundAlbedo,
    PrescribedSurfaceAlbedo,
    surface_albedo,
    partition_snow_surface_fluxes


abstract type AbstractBucketModel{FT} <: AbstractExpModel{FT} end

abstract type AbstractBucketAlbedoModel{FT <: AbstractFloat} end

"""
    PrescribedBaregroundAlbedo{FT, F <: ClimaCore.Fields.Field} <: AbstractBucketAlbedoModel

An albedo model where the static snow-free bareground albedo is prescribed as a
function of space or using data from a file, and the land surface albedo is
computed each timestep as a linear combination of the snow albedo and the bareground
albedo, following the SLIM model (Lague et al 2019).
"""
struct PrescribedBaregroundAlbedo{FT, F <: ClimaCore.Fields.Field} <:
       AbstractBucketAlbedoModel{FT}
    α_snow::FT
    α_bareground::F
end

"""
     PrescribedBaregroundAlbedo{FT}(α_snow::FT,
                                    α_bareground_func::Function,
                                    surface_space::ClimaCore.Spaces.AbstractSpace
                                    ) where {FT}

An outer constructor for the PrescribedBaregroundAlbedo model which uses an analytic function
of the coordinates to compute α_bareground on the model `surface_space`.

This particular method can be used with site level or global runs.
"""
function PrescribedBaregroundAlbedo{FT}(
    α_snow::FT,
    α_bareground_func::Function,
    surface_space::ClimaCore.Spaces.AbstractSpace,
) where {FT}
    α_bareground = SpaceVaryingInput(α_bareground_func, surface_space)
    return PrescribedBaregroundAlbedo{FT, typeof(α_bareground)}(
        α_snow,
        α_bareground,
    )
end


"""
     PrescribedBaregroundAlbedo{FT}(α_snow::FT,
                                    surface_space::ClimaCore.Spaces.AbstractSpace;
                                    varnames = ["sw_alb"],
                                    albedo_file_path::AbstractString = ClimaLand.Artifacts.bareground_albedo_dataset_path(),
                                    ) where{FT}

An outer constructor for the PrescribedBaregroundAlbedo model which uses data
from a file obtained from a net cdf file for the bareground albedo.

This particular method can only be used with global runs.
"""
function PrescribedBaregroundAlbedo{FT}(
    α_snow::FT,
    surface_space::ClimaCore.Spaces.AbstractSpace;
    varnames = ["sw_alb"],
    albedo_file_path::AbstractString = ClimaLand.Artifacts.bareground_albedo_dataset_path(),
    regridder_type = :InterpolationsRegridder,
) where {FT}
    if surface_space isa ClimaCore.Spaces.PointSpace
        error("Using an albedo map requires a global run.")
    end
    # Albedo file only has one variable, so access first `varname`
    α_bareground = SpaceVaryingInput(
        albedo_file_path,
        varnames[begin],
        surface_space;
        regridder_type,
    )
    return PrescribedBaregroundAlbedo{FT, typeof(α_bareground)}(
        α_snow,
        α_bareground,
    )
end

"""
    PrescribedSurfaceAlbedo{FT, TV <: AbstractTimeVaryingInput}
                       <: AbstractBucketAlbedoModel

An albedo model where the albedo of different surface types
is specified. Albedo is specified via a NetCDF file which is a function
of time and covers all surface types (soil, vegetation, snow, etc).
This albedo type changes over time according to the input file.

Note that this option should only be used with global simulations,
i.e. with a `ClimaLand.LSMSphericalShellDomain.`
"""
struct PrescribedSurfaceAlbedo{FT, TV <: AbstractTimeVaryingInput} <:
       AbstractBucketAlbedoModel{FT}
    albedo::TV
end

"""
    PrescribedSurfaceAlbedo{FT}(
        start_date::Union{DateTime, DateTimeNoLeap},
        Space::ClimaCore.Spaces.AbstractSpace;
        get_infile = ClimaLand.Artifacts.cesm2_albedo_dataset_path,
        varname = "sw_alb"
    ) where {FT}

Constructor for the PrescribedSurfaceAlbedo struct.
The `varname` must correspond to the name of the variable in the NetCDF
file retrieved by the `get_infile` function.
`get_infile` uses ArtifactWrappers.jl to return a path to the data file
and download the data if it doesn't already exist on the machine.
The input data file must have a time component.
"""
function PrescribedSurfaceAlbedo{FT}(
    start_date::Union{DateTime, DateTimeNoLeap},
    space::ClimaCore.Spaces.AbstractSpace;
    albedo_file_path = ClimaLand.Artifacts.cesm2_albedo_dataset_path(),
    varname = "sw_alb",
    regridder_type = :InterpolationsRegridder,
) where {FT}
    # Verify inputs
    if typeof(space) <: ClimaCore.Spaces.PointSpace
        error("Using an albedo map requires a global run.")
    end

    data_handler = DataHandling.DataHandler(
        albedo_file_path,
        varname,
        space;
        reference_date = start_date,
        regridder_type,
    )

    # Construct object containing info to read in surface albedo over time
    albedo = TimeVaryingInput(data_handler)
    return PrescribedSurfaceAlbedo{FT, typeof(albedo)}(albedo)
end



ClimaLand.name(::AbstractBucketModel) = :bucket

"""
    struct BucketModelParameters{
        FT <: AbstractFloat,
        PSE,
    }

Container for holding the parameters of the bucket model.

$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct BucketModelParameters{
    FT <: AbstractFloat,
    AAM <: AbstractBucketAlbedoModel,
    PSE,
}
    "Conductivity of the soil (W/K/m); constant"
    κ_soil::FT
    "Volumetric heat capacity of the soil (J/m^3/K); constant"
    ρc_soil::FT
    "Albedo Model"
    albedo::AAM
    "Critical σSWE amount (m) where surface transitions from to snow-covered"
    σS_c::FT
    "Fraction of critical amount of snow at which sublimation β begins to decay to zero (unitless)"
    f_snow::FT
    "Capacity of the land bucket (m)"
    W_f::FT
    "Fraction of bucket capacity at which evaporation β begins to decay to zero (unitless)"
    f_bucket::FT
    "Exponent used in β decay (unitless)"
    p::FT
    "Roughness length for momentum (m)"
    z_0m::FT
    "Roughness length for scalars (m)"
    z_0b::FT
    "τc timescale on which snow melts"
    τc::FT
    "Earth Parameter set; physical constants, etc"
    earth_param_set::PSE
end

"""

    struct BucketModel{
         FT,
         PS <: BucketModelParameters{FT},
         ATM <: AbstractAtmosphericDrivers{FT},
         RAD <: AbstractRadiativeDrivers{FT},
         D,
     } <: AbstractBucketModel{FT}

Concrete type for the BucketModel, which store the model
domain and parameters, as well as the necessary atmosphere
and radiation fields for driving the model.
$(DocStringExtensions.FIELDS)
"""
struct BucketModel{
    FT,
    PS <: BucketModelParameters{FT},
    ATM <: AbstractAtmosphericDrivers{FT},
    RAD <: AbstractRadiativeDrivers{FT},
    D,
} <: AbstractBucketModel{FT}
    "Parameters required by the bucket model"
    parameters::PS
    "The atmospheric drivers: Prescribed or Coupled"
    atmos::ATM
    "The radiation drivers: Prescribed or Coupled"
    radiation::RAD
    "The domain of the model"
    domain::D
end

"""
   BucketModel(; parameters::BucketModelParameters{FT, PSE},
                 domain::D,
                 atmosphere::ATM,
                 radiation::RAD,
               ) where {FT, PSE, ATM, RAD, D<: ClimaLand.Domains.AbstractDomain}

An outer constructor for the `BucketModel`, which enforces the
constraints:
1. The bucket model domain is of type <: ClimaLand.Domains.AbstractDomain
2. Using an albedo read from a lat/lon file requires a global run.
"""
function BucketModel(;
    parameters::BucketModelParameters{FT, PSE},
    domain::D,
    atmosphere::ATM,
    radiation::RAD,
) where {FT, PSE, ATM, RAD, D <: ClimaLand.Domains.AbstractDomain}
    if parameters.albedo isa PrescribedSurfaceAlbedo
        typeof(domain) <: SphericalShell ? nothing :
        error("Using an albedo map requires a global run.")
    end

    args = (parameters, atmosphere, radiation, domain)
    BucketModel{FT, typeof.(args)...}(args...)
end



prognostic_types(::BucketModel{FT}) where {FT} = (FT, FT, FT, FT)
prognostic_vars(::BucketModel) = (:W, :T, :Ws, :σS)
prognostic_domain_names(::BucketModel) =
    (:surface, :subsurface, :surface, :surface)

auxiliary_types(::BucketModel{FT}) where {FT} = (
    FT,
    NamedTuple{(:lhf, :shf, :vapor_flux, :r_ae), Tuple{FT, FT, FT, FT}},
    FT,
    FT,
    FT,
    FT,
    FT,
    FT,
    NamedTuple{(:F_melt, :F_into_snow, :G_under_snow), Tuple{FT, FT, FT}},
    FT,
    FT,
    FT,
)
auxiliary_vars(::BucketModel) = (
    :q_sfc,
    :turbulent_fluxes,
    :R_n,
    :T_sfc,
    :α_sfc,
    :ρ_sfc,
    :snow_cover_fraction,
    :F_sfc,
    :partitioned_fluxes,
    :G,
    :snow_melt,
    :infiltration,
)
auxiliary_domain_names(::BucketModel) = (
    :surface,
    :surface,
    :surface,
    :surface,
    :surface,
    :surface,
    :surface,
    :surface,
    :surface,
    :surface,
    :surface,
    :surface,
)


"""
    make_compute_exp_tendency(model::BucketModel{FT}) where {FT}

Creates the compute_exp_tendency! function for the bucket model.
"""
function make_compute_exp_tendency(model::BucketModel{FT}) where {FT}
    function compute_exp_tendency!(dY, Y, p, t)
        (; κ_soil, ρc_soil, σS_c) = model.parameters

        # Temperature profile of soil.
        gradc2f = ClimaCore.Operators.GradientC2F()
        divf2c = ClimaCore.Operators.DivergenceF2C(
            top = ClimaCore.Operators.SetValue(
                ClimaCore.Geometry.WVector.(p.bucket.G),
            ),
            bottom = ClimaCore.Operators.SetValue(
                ClimaCore.Geometry.WVector.(FT(0.0)),
            ),
        )

        @. dY.bucket.T = -1 / ρc_soil * (divf2c(-κ_soil * gradc2f(Y.bucket.T))) # Simple heat equation, Eq 10

        # Positive infiltration -> net (negative) flux into soil
        @. dY.bucket.W = -p.bucket.infiltration # Equation (2) of the text.

        liquid_precip = p.drivers.P_liq
        snow_precip = p.drivers.P_snow

        dY.bucket.Ws = @. -(
            liquid_precip +
            p.bucket.snow_cover_fraction * p.bucket.snow_melt +
            (1 - p.bucket.snow_cover_fraction) *
            p.bucket.turbulent_fluxes.vapor_flux - p.bucket.infiltration
        ) # Equation (3) of the text

        dY.bucket.σS = @. -(
            snow_precip +
            p.bucket.snow_cover_fraction *
            p.bucket.turbulent_fluxes.vapor_flux -
            p.bucket.snow_cover_fraction * p.bucket.snow_melt
        ) # Equation (6)
    end
    return compute_exp_tendency!
end

"""
    make_update_aux(model::BucketModel{FT}) where {FT}

Creates the update_aux! function for the BucketModel.
"""
function make_update_aux(model::BucketModel{FT}) where {FT}
    function update_aux!(p, Y, t)
        p.bucket.T_sfc .= ClimaLand.Domains.top_center_to_surface(Y.bucket.T)
        p.bucket.ρ_sfc .=
            surface_air_density(model.atmos, model, Y, p, t, p.bucket.T_sfc)

        # This relies on the surface specific humidity being computed
        # entirely over snow or over soil. i.e. the snow cover fraction must be a heaviside
        # here, otherwise we would need two values of q_sfc!
        p.bucket.q_sfc .=
            saturation_specific_humidity.(
                p.bucket.T_sfc,
                p.bucket.ρ_sfc,
                Ref(
                    LP.thermodynamic_parameters(
                        model.parameters.earth_param_set,
                    ),
                ),
            )
        # Compute turbulent surface fluxes
        p.bucket.turbulent_fluxes .=
            turbulent_fluxes(model.atmos, model, Y, p, t)

        # Radiative fluxes
        p.bucket.R_n .= net_radiation(model.radiation, model, Y, p, t)

        # Surface albedo
        next_albedo!(
            p.bucket.α_sfc,
            model.parameters.albedo,
            model.parameters,
            Y,
            p,
            t,
        )

        #Currently, the entire surface is assumed to be
        # snow covered entirely or not at all.
        p.bucket.snow_cover_fraction .= heaviside.(Y.bucket.σS)

        # In this case, there is just one set of surface fluxes to compute.
        # Since q_sfc itself has already been modified to account for
        # snow covered regions, and since the surface temperature is
        # assumed to be the same for snow and underlying land,
        # the returned fluxes are computed correctly for the cell
        # regardless of snow-coverage.

        # The below must be adjusted to compute F_sfc over snow and over soil
        # if we want the snow cover fraction to be intermediate between 0 and 1.
        @. p.bucket.F_sfc = (
            p.bucket.turbulent_fluxes.shf .+ p.bucket.turbulent_fluxes.lhf -
            p.bucket.R_n
        ) # Eqn (21)
        _T_freeze = LP.T_freeze(model.parameters.earth_param_set)
        _LH_f0 = LP.LH_f0(model.parameters.earth_param_set)
        _ρ_liq = LP.ρ_cloud_liq(model.parameters.earth_param_set)
        _ρLH_f0 = _ρ_liq * _LH_f0 # Latent heat per unit volume.
        # partition energy fluxes for snow covered area
        p.bucket.partitioned_fluxes .=
            partition_snow_surface_fluxes.(
                Y.bucket.σS,
                p.bucket.T_sfc,
                model.parameters.τc,
                p.bucket.snow_cover_fraction,
                p.bucket.turbulent_fluxes.vapor_flux,
                p.bucket.F_sfc,
                _ρLH_f0,
                _T_freeze,
            )
        @. p.bucket.G =
            p.bucket.F_sfc * (1 - p.bucket.snow_cover_fraction) +
            p.bucket.partitioned_fluxes.G_under_snow *
            p.bucket.snow_cover_fraction # Equation 22

        # Partition water fluxes
        liquid_precip = p.drivers.P_liq
        # F_melt is negative as it is a downward welling flux warming the snow
        @.p.bucket.snow_melt = p.bucket.partitioned_fluxes.F_melt / _ρLH_f0 # defined after Equation (22)

        @. p.bucket.infiltration = infiltration_at_point(
            Y.bucket.W,
            p.bucket.snow_cover_fraction * p.bucket.snow_melt,
            liquid_precip,
            (1 - p.bucket.snow_cover_fraction) *
            p.bucket.turbulent_fluxes.vapor_flux,
            model.parameters.W_f,
        ) # Equation (2) of the text.

    end
    return update_aux!
end

"""
    next_albedo!(next_α_sfc,
                 model_albedo::PrescribedBaregroundAlbedo{FT},
                 parameters, Y, p, t)

Update the surface albedo for time `t`: the albedo is calculated by
linearly interpolating between the albedo
of snow and of the bareground surface, based on the snow water equivalent `S` relative to
the parameter `S_c`. The linear interpolation is taken from Lague et al 2019.

Note that if our snow_cover_fraction function was smoothly varying, the albedo
would simply be σα_snow + (1-σ)α_bareground. Since we cannot support snow cover
fractions that are not a heaviside function, we have a small inconsistency
for 0 < σS < eps(FT) where the snow cover fraction is zero, but there is a small
contribution of snow to the albedo.
"""
function next_albedo!(
    next_α_sfc,
    model_albedo::PrescribedBaregroundAlbedo{FT},
    parameters,
    Y,
    p,
    t,
) where {FT}
    (; α_snow, α_bareground) = model_albedo
    (; σS_c) = parameters
    σS = max.(Y.bucket.σS, FT(0))# if σS is small and negative, clip to zero for albedo estimation
    @. next_α_sfc =
        ((1 - σS / (σS + σS_c)) * α_bareground + σS / (σS + σS_c) * α_snow)
end

"""
    next_albedo!(next_α_sfc, model_albedo::PrescribedSurfaceAlbedo{FT}, parameters, Y, p, t)

Update the surface albedo for time `t`: for a file containing surface albedo
information over time, this reads in the value for time t.
"""
function next_albedo!(
    next_α_sfc,
    model_albedo::PrescribedSurfaceAlbedo{FT},
    parameters,
    Y,
    p,
    t,
) where {FT}
    TimeVaryingInputs.evaluate!(next_α_sfc, model_albedo.albedo, t)
end

function ClimaLand.get_drivers(model::BucketModel)
    return (model.atmos, model.radiation)
end

include("./bucket_parameterizations.jl")

end
