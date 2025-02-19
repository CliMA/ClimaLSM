"""
    set_eff_land_radiation_properties!(p, earth_param_set)

Sets the values of the effective land broadband albedo p.α_sfc and emissiviity p.ϵ_sfc,
and the corresponding temperature of blackbody p.T_sfc in place, using downwelling
and upwelling SW radiation, and LW upwelling.
"""
function set_eff_land_radiation_properties!(p, earth_param_set)
    # Effective (radiative) land properties
    _σ = LP.Stefan(earth_param_set)
    FT = typeof(_σ)
    @. p.α_sfc = p.SW_u / max(p.drivers.SW_d, eps(FT))
    @. p.ϵ_sfc = 1
    @. p.T_sfc = (p.LW_u / (p.ϵ_sfc * _σ))^(1 / 4)
end


"""
    Canopy.canopy_radiant_energy_fluxes!(p::NamedTuple,
                                         s::Union{PrognosticGroundConditions,PrognosticSoilConditions},
                                         canopy,
                                         radiation::PrescribedRadiativeFluxes,
                                         earth_param_set::PSE,
                                         Y::ClimaCore.Fields.FieldVector,
                                         t,
                                        ) where {PSE}

In standalone mode, this function computes and stores the net
long and short wave radition, in W/m^2,
absorbed by the canopy.

In integrated mode, we have already computed those quantities in
`lsm_radiant_energy_fluxes!`, so this method does nothing additional.

LW and SW net radiation are stored in `p.canopy.radiative_transfer.LW_n`
and `p.canopy.radiative_transfer.SW_n`.
"""
function Canopy.canopy_radiant_energy_fluxes!(
    p::NamedTuple,
    s::Union{PrognosticGroundConditions, PrognosticSoilConditions},
    canopy,
    radiation::AbstractRadiativeDrivers,
    earth_param_set::PSE,
    Y::ClimaCore.Fields.FieldVector,
    t,
) where {PSE}
    nothing
end
