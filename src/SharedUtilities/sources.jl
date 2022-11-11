export source!, AbstractSource

"""
    AbstractSource{FT <: AbstractFloat}
An abstract type for types of source terms.
"""
abstract type AbstractSource{FT <: AbstractFloat} end

"""
     source!(dY::ClimaCore.Fields.FieldVector,
             src::AbstractSource,
             Y::ClimaCore.Fields.FieldVector,
             p::ClimaCore.Fields.FieldVector
             )::ClimaCore.Fields.Field
A stub function, which is extended by ClimaLSM.
"""
function source!(
    dY::ClimaCore.Fields.FieldVector,
    src::AbstractSource,
    Y::ClimaCore.Fields.FieldVector,
    p::ClimaCore.Fields.FieldVector,
    _...,
)::ClimaCore.Fields.Field end
