using Test
import ClimaComms
ClimaComms.@import_required_backends
using ClimaLand
import ClimaLand
import ClimaParams as CP
import ClimaLand.Parameters as LP
using Dates
FT = Float32
earth_param_set = LP.LandParameters(FT)
domain = ClimaLand.Domains.Column(; zlim = FT.((-100.0, 0.0)), nelements = 10)
# Radiation
start_date = DateTime(2005)
SW_d = (t) -> 500
LW_d = (t) -> 5.67e-8 * 285.0^4.0
radiation = PrescribedRadiativeFluxes(
    FT,
    TimeVaryingInput(SW_d),
    TimeVaryingInput(LW_d),
    start_date,
)
# Atmos
precip = (t) -> 1e-7
precip_snow = (t) -> 0
T_atmos = (t) -> 285
u_atmos = (t) -> 3
q_atmos = (t) -> 0.005
h_atmos = FT(10)
P_atmos = (t) -> 101325
atmos = PrescribedAtmosphere(
    TimeVaryingInput(precip),
    TimeVaryingInput(precip_snow),
    TimeVaryingInput(T_atmos),
    TimeVaryingInput(u_atmos),
    TimeVaryingInput(q_atmos),
    TimeVaryingInput(P_atmos),
    start_date,
    h_atmos,
    earth_param_set,
)

# Required to make soil model
ν = FT(0.495)
K_sat = FT(0.0443 / 3600 / 100) # m/s
S_s = FT(1e-3) #inverse meters
vg_n = FT(2.0)
vg_α = FT(2.6) # inverse meters
vg_m = FT(1) - FT(1) / vg_n
hcm = ClimaLand.Soil.vanGenuchten{FT}(; α = vg_α, n = vg_n)
θ_r = FT(0.1)
ν_ss_om = FT(0.0)
ν_ss_quartz = FT(1.0)
ν_ss_gravel = FT(0.0)
emissivity = FT(0.99)
PAR_albedo_dry = FT(0.2)
NIR_albedo_dry = FT(0.4)
PAR_albedo_wet = FT(0.1)
NIR_albedo_wet = FT(0.2)
z_0m = FT(0.001)
z_0b = z_0m
soil_params = ClimaLand.Soil.EnergyHydrologyParameters(
    FT;
    ν,
    ν_ss_om,
    ν_ss_quartz,
    ν_ss_gravel,
    hydrology_cm = hcm,
    K_sat,
    S_s,
    θ_r,
    PAR_albedo_dry,
    NIR_albedo_dry,
    PAR_albedo_wet,
    NIR_albedo_wet,
    emissivity,
    z_0m,
    z_0b,
)
# Required snow model parameters
Δt = FT(180.0)
snow_params =
    ClimaLand.Snow.SnowParameters{FT}(Δt; earth_param_set = earth_param_set)

# Land model
land_model = ClimaLand.LandHydrologyModel{FT}(;
    land_args = (atmos = atmos, radiation = radiation, domain = domain),
    snow_model_type = ClimaLand.Snow.SnowModel,
    snow_args = (parameters = snow_params,),
    soil_model_type = ClimaLand.Soil.EnergyHydrology,
    soil_args = (parameters = soil_params,),
)

Y, p, coords = ClimaLand.initialize(land_model)
p_soil_alone = deepcopy(p)
for lsm_aux_var in (
    :excess_water_flux,
    :excess_heat_flux,
    :atmos_energy_flux,
    :atmos_water_flux,
    :ground_heat_flux,
    :effective_soil_sfc_T,
    :sfc_scratch,
    :subsfc_scratch,
    :effective_soil_sfc_depth,
)
    @test lsm_aux_var ∈ propertynames(p)
end
@test typeof(land_model.soil.boundary_conditions.top) <:
      ClimaLand.Soil.AtmosDrivenFluxBC
@test land_model.soil.boundary_conditions.top.prognostic_land_components ==
      (:snow, :soil)
src = ClimaLand.SoilSublimationwithSnow{FT}()
@test src ∈ land_model.soil.sources
@test ClimaLand.get_drivers(land_model) == (atmos, radiation)
# Set initial conditions for a case with *no snow on ground*
function init_soil!(Y, z, params)
    ν = params.ν
    FT = eltype(ν)
    Y.soil.ϑ_l .= ν / 2
    Y.soil.θ_i .= 0.05
    T = FT(273)
    ρc_s = ClimaLand.Soil.volumetric_heat_capacity(
        ν / 2,
        FT(0),
        params.ρc_ds,
        params.earth_param_set,
    )
    Y.soil.ρe_int =
        ClimaLand.Soil.volumetric_internal_energy.(
            FT(0),
            ρc_s,
            T,
            params.earth_param_set,
        )
end
function init_snow!(Y, S)
    Y.snow.S .= S
    Y.snow.Sl .= 0
    @. Y.snow.U =
        ClimaLand.Snow.energy_from_q_l_and_swe(Y.snow.S, 1.0f0, snow_params)
end

t = Float64(0)
init_soil!(Y, coords.subsurface.z, land_model.soil.parameters)
init_snow!(Y, 0.0f0)

set_initial_cache! = make_set_initial_cache(land_model)
set_initial_cache!(p, Y, t)
# Make sure snow boundary fluxes are zero
@test all(parent(p.snow.total_energy_flux) .≈ 0)
@test all(parent(p.snow.total_water_flux) .≈ 0)
# Make sure the boundary conditions match bare soil result
set_soil_initial_cache! = make_set_initial_cache(land_model.soil)
set_soil_initial_cache!(p_soil_alone, Y, t)
@test p.soil.top_bc == p_soil_alone.soil.top_bc
dY_soil_snow = deepcopy(Y) .* 0
dY_soil_alone = deepcopy(Y) .* 0
ClimaLand.source!(
    dY_soil_alone,
    ClimaLand.Soil.SoilSublimation{FT}(),
    Y,
    p_soil_alone,
    land_model.soil,
)
ClimaLand.source!(dY_soil_snow, src, Y, p, land_model.soil)
@test dY_soil_alone.soil == dY_soil_snow.soil


# Repeat now with snow cover fraction of 1
init_snow!(Y, 1.0f0)
set_initial_cache!(p, Y, t)
# Make sure the boundary conditions for soil are correct since that is what the LandHydrology methods
# affect
@test all(
    parent(p.soil.top_bc.water) .≈
    parent(p.excess_water_flux .+ p.snow.water_runoff),
)
@test all(
    parent(p.soil.top_bc.heat) .≈ parent(
        p.excess_heat_flux .+ p.snow.snow_cover_fraction .* p.ground_heat_flux,
    ),
)
dY_soil_snow = deepcopy(Y) .* 0
ClimaLand.source!(dY_soil_snow, src, Y, p, land_model.soil)
@test all(parent(dY_soil_snow.soil.θ_i) .≈ 0)

# Check conservation
@test p.atmos_water_flux == @. p.drivers.P_snow +
         p.drivers.P_liq +
         (1 - p.snow.snow_cover_fraction) * (
             p.soil.turbulent_fluxes.vapor_flux_liq +
             p.soil.turbulent_fluxes.vapor_flux_ice
         ) +
         p.snow.snow_cover_fraction * p.snow.turbulent_fluxes.vapor_flux

_LH_f0 = FT(LP.LH_f0(earth_param_set))
_ρ_liq = FT(LP.ρ_cloud_liq(earth_param_set))
ρe_falling_snow = -_LH_f0 * _ρ_liq # per unit vol of liquid water
@test p.atmos_energy_flux == @. (1 - p.snow.snow_cover_fraction) * (
             p.soil.turbulent_fluxes.lhf +
             p.soil.turbulent_fluxes.shf +
             p.soil.R_n
         ) +
         p.snow.snow_cover_fraction * (
             p.snow.turbulent_fluxes.lhf +
             p.snow.turbulent_fluxes.shf +
             p.snow.R_n
         ) +
         p.drivers.P_snow * ρe_falling_snow
# Make sure soil boundary flux method also worked
G = deepcopy(p.ground_heat_flux)
p_snow_alone = deepcopy(p)
ClimaLand.Snow.snow_boundary_fluxes!(
    land_model.snow.boundary_conditions,
    Val((:snow,)),
    land_model.snow,
    Y,
    p_snow_alone,
    t,
)
ClimaLand.Snow.snow_boundary_fluxes!(
    land_model.snow.boundary_conditions,
    Val((:snow, :soil)),
    land_model.snow,
    Y,
    p,
    t,
)
@test all(
    parent(p_snow_alone.snow.total_energy_flux .- p.snow.total_energy_flux) .≈
    parent(p.snow.snow_cover_fraction .* p.ground_heat_flux),
)
