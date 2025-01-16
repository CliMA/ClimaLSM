Snow Model

```@meta
CurrentModule = ClimaLand.Snow
```
## Snow Parameters

```@docs
ClimaLand.Snow.SnowParameters
```

## Snow Functions of State

```@docs
ClimaLand.Snow.specific_heat_capacity
ClimaLand.Snow.snow_surface_temperature
ClimaLand.Snow.snow_depth
ClimaLand.Snow.snow_thermal_conductivity
ClimaLand.Snow.snow_bulk_temperature
ClimaLand.Snow.maximum_liquid_mass_fraction
ClimaLand.Snow.runoff_timescale
ClimaLand.Snow.compute_water_runoff	
ClimaLand.Snow.energy_from_q_l_and_swe
ClimaLand.Snow.energy_from_T_and_swe
ClimaLand.Snow.snow_cover_fraction
```

## Computing fluxes for snow

```@docs
ClimaLand.Snow.snow_boundary_fluxes!
ClimaLand.Snow.phase_change_flux
ClimaLand.Snow.AtmosDrivenSnowBC
```