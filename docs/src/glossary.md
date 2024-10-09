# Definition of variables and terms

## Sign and Units conventions

### Basic units
All variables are in standard units: `kg` for mass, `m` for length, and
seconds `s` for time.  Zenith angle is reported in radians. We sometimes
use the units of moles and number of photons in the Canopy model where noted.

Currently, dates are used only for interfacing with a calendar
(for forcing data, for output, for the insolation), and the simulation time
is in units of seconds from a reference date.

### Units for fluxes
- Energy fluxes have units of W/m^2.
- Water fluxes have units of m^3/m^2/s = m/s. We are planning to change this to a mass flux.
- Carbon fluxes have units of moles CO2/m^2/s or kg C/m^2/s, depending on the context. We are planning to change this to be consistent throughout.

### Sign conventions for fluxes
The majority of fluxes follow the convention that upwards is positive and
downwards is negative. So, we have that:
- Sensible, latent, and vapor fluxes are positive if towards the atmosphere.
- Snowmelt is negative.
- Root extraction is positive if the canopy is taking water from the soil.
- The ground heat flux between soil and snow is positive if the snow is warming.

Some fluxes, however, are represented only with scalars, with a direction
built in, such as:
- Precipitation is positive by definition, but always is downwards.
- Radiative fluxes marked with `_d` are downwelling and positive by
definition.
- Radiative fluxes marked with `_u` are upwelling and positive by definition.
- Net radiation is defined to be `R_n = SW_d + LW_d - SW_u - LW_u`, where
`SW` and `LW indicate the total short and longwave fluxes. Therefore, the
ClimaLand convention is that `R_n` is positive if the land is gaining energy.
- Runoff is defined to be positive.