"""
    define_diagnostics!(land_model)

Calls `add_diagnostic_variable!` for all available variables specializing the
compute function for `land_model`.
"""
function define_diagnostics!(land_model)

    ### BucketModel ###

    ## Stored in p (diagnostics variables stored in the cache) ##

    # Shortwave Albedo
    add_diagnostic_variable!(
        short_name = "swa",
        long_name = "Shortwave Albedo",
        standard_name = "sw_albedo",
        units = "",
        comments = "The fraction of downwelling shortwave radiation reflected by the land surface.",
        compute! = (out, Y, p, t) ->
            compute_sw_albedo!(out, Y, p, t, land_model),
    )

    # Net radiation
    add_diagnostic_variable!(
        short_name = "rn",
        long_name = "Net Radiation",
        standard_name = "net_radiation",
        units = "W m^-2",
        comments = "Difference between downwelling and upwelling shortwave and longwave radiation at the land surface.",
        compute! = (out, Y, p, t) ->
            compute_net_radiation!(out, Y, p, t, land_model),
    )

    # Bucket Surface temperature
    add_diagnostic_variable!(
        short_name = "tsfc",
        long_name = "Bucket Surface Temperature",
        standard_name = "surface_temperature",
        units = "K",
        comments = "Temperature of the bucket-land surface.",
        compute! = (out, Y, p, t) ->
            compute_surface_temperature!(out, Y, p, t, land_model),
    )

    # Latent heat flux
    add_diagnostic_variable!(
        short_name = "lhf",
        long_name = "Latent Heat Flux",
        standard_name = "latent_heat_flux",
        units = "W m^-2",
        comments = "Exchange of energy at the land-atmosphere interface due to water evaporation or sublimation.",
        compute! = (out, Y, p, t) ->
            compute_latent_heat_flux!(out, Y, p, t, land_model),
    )

    # Aerodynamic resistance
    add_diagnostic_variable!(
        short_name = "rae",
        long_name = "Aerodynamic Resistance",
        standard_name = "aerodynamic_resistance",
        units = "m s^-1",
        comments = "Effiency of turbulent transport controlling the land-atmosphere exchange of sensible and latent heat.",
        compute! = (out, Y, p, t) ->
            compute_aerodynamic_resistance!(out, Y, p, t, land_model),
    )

    # Sensible heat flux
    add_diagnostic_variable!(
        short_name = "shf",
        long_name = "Sensible Heat Flux",
        standard_name = "sensible_heat_flux",
        units = "W m^-2",
        comments = "Exchange of energy at the land-atmosphere interface due to temperature difference.",
        compute! = (out, Y, p, t) ->
            compute_sensible_heat_flux!(out, Y, p, t, land_model),
    )

    # Vapor flux
    add_diagnostic_variable!(
        short_name = "vflux",
        long_name = "Liquid water evaporation",
        standard_name = "vapor_flux",
        units = "m s^-1",
        comments = "Flux of water from the land surface to the atmosphere. E.g., evaporation or sublimation.",
        compute! = (out, Y, p, t) ->
            compute_vapor_flux!(out, Y, p, t, land_model),
    )

    # Surface air density
    add_diagnostic_variable!(
        short_name = "rhosfc",
        long_name = "Surface Air Density",
        standard_name = "surface_air_density",
        units = "kg m^−3",
        comments = "Density of air at the land-atmosphere interface.",
        compute! = (out, Y, p, t) ->
            compute_surface_air_density!(out, Y, p, t, land_model),
    )

    ## Stored in Y (prognostic or state variables) ##

    # Soil temperature (3D) at depth
    add_diagnostic_variable!(
        short_name = "tsoil",
        long_name = "Soil temperature",
        standard_name = "soil_temperature",
        units = "K",
        comments = "Soil temperature at multiple soil depth. (depth resolved)",
        compute! = (out, Y, p, t) ->
            compute_soil_temperature!(out, Y, p, t, land_model),
    )

    # Surbsurface water storage
    add_diagnostic_variable!(
        short_name = "wsoil",
        long_name = "subsurface Water Storage",
        standard_name = "subsurface_water_storage",
        units = "m",
        comments = "Soil water content.",
        compute! = (out, Y, p, t) ->
            compute_subsurface_water_storage!(out, Y, p, t, land_model),
    )

    # Surface water content
    add_diagnostic_variable!(
        short_name = "wsfc",
        long_name = "Surface Water Content",
        standard_name = "surface_water_content",
        units = "m",
        comments = "Water at the soil surface.",
        compute! = (out, Y, p, t) ->
            compute_surface_water_content!(out, Y, p, t, land_model),
    )

    # Surface snow water content
    add_diagnostic_variable!(
        short_name = "ssfc",
        long_name = "Snow Water Equivalent",
        standard_name = "snow_water_equivalent",
        units = "m",
        comments = "Snow at the soil surface, expressed in water equivalent.",
        compute! = (out, Y, p, t) ->
            compute_snow_water_equivalent!(out, Y, p, t, land_model),
    )

    ###### SoilCanopyModel ######

    ## stored in p (diagnostics variables stored in the cache) ##

    ## Canopy Module ##

    ### Canopy - Solar Induced Fluorescence
    # Solar Induced Fluorescence
    add_diagnostic_variable!(
        short_name = "sif",
        long_name = "Solar Induced Fluorescence",
        standard_name = "solar_induced_fluorescence",
        units = "W m^-2",
        comments = "The fluorescence of leaves induced by solar radiation at 755nm. This quantity is correlated with photosynthesis activity.",
        compute! = (out, Y, p, t) ->
            compute_solar_induced_fluorescence!(out, Y, p, t, land_model),
    )

    ### Canopy - Autotrophic respiration
    # Autotrophic respiration
    add_diagnostic_variable!(
        short_name = "ra",
        long_name = "Autotrophic Respiration",
        standard_name = "autotrophic_respiration",
        units = "kg C m^-2 s^-1",
        comments = "The canopy autotrophic respiration, the sum of leaves, stems and roots respiration.",
        compute! = (out, Y, p, t) ->
            compute_autotrophic_respiration!(out, Y, p, t, land_model),
    )

    ### Canopy - Conductance
    # Stomatal conductance
    add_diagnostic_variable!(
        short_name = "gs",
        long_name = "Stomatal Conductance",
        standard_name = "stomatal_conductance",
        units = "mol H2O m^-2 s^-1", # do we want this in kg h2o?
        comments = "The conductance of leaves. This depends on stomatal opening. It varies with factors such as soil moisture or atmospheric water demand.",
        compute! = (out, Y, p, t) ->
            compute_stomatal_conductance!(out, Y, p, t, land_model),
    )

    # Canopy transpiration
    add_diagnostic_variable!(
        short_name = "trans",
        long_name = "Canopy Transpiration",
        standard_name = "canopy_transpiration",
        units = "m s^-1",
        comments = "The water evaporated from the canopy due to leaf transpiration (flux of water volume, m^3 of water per m^2 of ground).",
        compute! = (out, Y, p, t) ->
            compute_canopy_transpiration!(out, Y, p, t, land_model),
    )

    ### Canopy - Energy

    # Canopy latent heat flux
    add_diagnostic_variable!(
        short_name = "clhf",
        long_name = "Canopy Latent Heat Flux",
        standard_name = "canopy_latent_heat_flux",
        units = "W m^-2",
        comments = "The energy used for canopy transpiration.",
        compute! = (out, Y, p, t) ->
            compute_canopy_latent_heat_flux!(out, Y, p, t, land_model),
    )

    # Canopy sensible heat flux
    add_diagnostic_variable!(
        short_name = "cshf",
        long_name = "Canopy Sensible Heat Flux",
        standard_name = "canopy_sensible_heat_flux",
        units = "W m^-2",
        comments = "The energy used for canopy temperature change.",
        compute! = (out, Y, p, t) ->
            compute_canopy_sensible_heat_flux!(out, Y, p, t, land_model),
    )

    ### Canopy - Hydraulics
    # Leaf water potential


    add_diagnostic_variable!(
        short_name = "lwp",
        long_name = "Leaf Water Potential",
        standard_name = "leaf_water_potential",
        units = "m",
        comments = "The water potential of a leaf.",
        compute! = (out, Y, p, t) ->
            compute_leaf_water_potential!(out, Y, p, t, land_model),
    )
    #=
        # Flux per ground area
        add_diagnostic_variable!(
            short_name = "fa",
            long_name = "Flux Per Ground Area",
            standard_name = "flux_per_ground_area",
            units = "m s^-1",
            comments = "Flux of water volume per m^2 of plant per second, multiplied by the area index (plant area/ground area).",
            compute! = (out, Y, p, t) ->
                compute_flux_per_ground_area!(out, Y, p, t, land_model),
        )
        =#

    # Root flux per ground area
    add_diagnostic_variable!(
        short_name = "far",
        long_name = "Root flux per ground area",
        standard_name = "root_flux_per_ground_area",
        units = "m s^-1",
        comments = "Flux of water volume per m^2 of root per second, multiplied by the area index (root area/ground area).",
        compute! = (out, Y, p, t) ->
            compute_root_flux_per_ground_area!(out, Y, p, t, land_model),
    )

    # Leaf area index
    add_diagnostic_variable!(
        short_name = "lai",
        long_name = "Leaf area Index",
        standard_name = "leaf_area_index",
        units = "m^2 m^-2",
        comments = "The area index of leaves, expressed in surface area of leaves per surface area of ground.",
        compute! = (out, Y, p, t) ->
            compute_leaf_area_index!(out, Y, p, t, land_model),
    )

    # Moisture stress factor
    add_diagnostic_variable!(
        short_name = "msf",
        long_name = "Moisture Stress Factor",
        standard_name = "moisture_stress_factor",
        units = "",
        comments = "Sensitivity of plants conductance to soil water content. Unitless",
        compute! = (out, Y, p, t) ->
            compute_moisture_stress_factor!(out, Y, p, t, land_model),
    )

    # Root area index
    add_diagnostic_variable!(
        short_name = "rai",
        long_name = "Root area Index",
        standard_name = "root_area_index",
        units = "m^2 m^-2",
        comments = "The area index of roots, expressed in surface area of roots per surface area of ground.",
        compute! = (out, Y, p, t) ->
            compute_root_area_index!(out, Y, p, t, land_model),
    )

    # Stem area index
    add_diagnostic_variable!(
        short_name = "sai",
        long_name = "Stem area Index",
        standard_name = "stem_area_index",
        units = "m^2 m^-2",
        comments = "The area index of stems, expressed in surface area of stems per surface area of ground.",
        compute! = (out, Y, p, t) ->
            compute_stem_area_index!(out, Y, p, t, land_model),
    )

    ### Canopy - Photosynthesis
    # GPP - Gross Primary Productivity
    add_diagnostic_variable!(
        short_name = "gpp",
        long_name = "Gross Primary Productivity",
        standard_name = "gross_primary_productivity",
        units = "kg C m^-2 s^-1",
        comments = "Gross photosynthesis (carbon assimilation) of the canopy.", # Discuss, is this net leaf assimilation scaled? therefore, is it gpp?
        compute! = (out, Y, p, t) ->
            compute_photosynthesis_gross_canopy!(out, Y, p, t, land_model),
    )

    # Leaf net photosynthesis
    add_diagnostic_variable!(
        short_name = "an",
        long_name = "Leaf Net Photosynthesis",
        standard_name = "leaf_net_photosynthesis",
        units = "kg C m^-2 s^-1",
        comments = "Net photosynthesis (carbon assimilation) of a leaf, computed for example by the Farquhar model.",
        compute! = (out, Y, p, t) ->
            compute_photosynthesis_net_leaf!(out, Y, p, t, land_model),
    )

    # Leaf respiration
    add_diagnostic_variable!(
        short_name = "rd",
        long_name = "Leaf Respiration",
        standard_name = "leaf_dark_respiration",
        units = "kg C m^-2 s^-1",
        comments = "Leaf respiration, called dark respiration because usually measured in the abscence of radiation.",
        compute! = (out, Y, p, t) ->
            compute_respiration_leaf!(out, Y, p, t, land_model),
    )

    # Vcmax25
    add_diagnostic_variable!(
        short_name = "vcmax25", # isn't this a parameter? why do we return it?
        long_name = "Vcmax25",
        standard_name = "vcmax25",
        units = "mol CO2 m^-2 s^-1",
        comments = "The parameter vcmax at 25 degree celsius. Important for the Farquhar model of leaf photosynthesis.",
        compute! = (out, Y, p, t) -> compute_vcmax25!(out, Y, p, t, land_model),
    )

    ### Canopy - Radiative Transfer
    # NIR - near infrared radiaton
    add_diagnostic_variable!(
        short_name = "nir",
        long_name = "Near Infrared Radiation",
        standard_name = "near_infrared_radiation",
        units = "mol photons m^-2 s^-1", # Do we want all radiation fluxes in W m-2?
        comments = "The amount of near infrared radiation reaching the canopy.",
        compute! = (out, Y, p, t) ->
            compute_near_infrared_radiation_down!(out, Y, p, t, land_model),
    )

    # ANIR - absorbed near infrared radiation
    add_diagnostic_variable!(
        short_name = "anir",
        long_name = "Absorbed Near Infrared Radiation",
        standard_name = "absorbed_near_infrared_radiation",
        units = "mol photons m^-2 s^-1",
        comments = "The amount of near infrared radiation absorbed by the canopy.",
        compute! = (out, Y, p, t) ->
            compute_near_infrared_radiation_absorbed!(out, Y, p, t, land_model),
    )

    # RNIR - reflected near infrared radiation
    add_diagnostic_variable!(
        short_name = "rnir",
        long_name = "Reflected Near Infrared Radiation",
        standard_name = "reflected_near_infrared_radiation",
        units = "mol photons m^-2 s^-1",
        comments = "The amount of near infrared radiation reflected by the canopy.",
        compute! = (out, Y, p, t) ->
            compute_near_infrared_radiation_reflected!(
                out,
                Y,
                p,
                t,
                land_model,
            ),
    )

    # TNIR - transmitted near infrared radiation
    add_diagnostic_variable!(
        short_name = "tnir",
        long_name = "Transmitted Near Infrared Radiation",
        standard_name = "transmitted_near_infrared_radiation",
        units = "mol photons m^-2 s^-1",
        comments = "The amount of near infrared radiation transmitted by the canopy.",
        compute! = (out, Y, p, t) ->
            compute_near_infrared_radiation_transmitted!(
                out,
                Y,
                p,
                t,
                land_model,
            ),
    )

    # PAR - photosynthetically active radiation
    add_diagnostic_variable!(
        short_name = "par",
        long_name = "Photosynthetically Active Radiation",
        standard_name = "photosynthetically_active_radiation",
        units = "mol photons m^-2 s^-1",
        comments = "The subset of total radiation that activates photosynthesis reaching the canopy.",
        compute! = (out, Y, p, t) ->
            compute_photosynthetically_active_radiation_down!(
                out,
                Y,
                p,
                t,
                land_model,
            ),
    )

    # APAR - absorbed photosynthetically active radiation
    add_diagnostic_variable!(
        short_name = "apar",
        long_name = "Absorbed Photosynthetically Active Radiation",
        standard_name = "absorbed_photosynthetically_active_radiation",
        units = "mol photons m^-2 s^-1",
        comments = "The amount of photosynthetically active radiation absorbed by the leaf. The rest if reflected or transmitted.",
        compute! = (out, Y, p, t) ->
            compute_photosynthetically_active_radiation_absorbed!(
                out,
                Y,
                p,
                t,
                land_model,
            ),
    )

    # RPAR - reflected photosynthetically active radiation
    add_diagnostic_variable!(
        short_name = "rpar",
        long_name = "Reflected Photosynthetically Active Radiation",
        standard_name = "reflected_photosynthetically_active_radiation",
        units = "mol photons m^-2 s^-1",
        comments = "The amount of photosynthetically active radiation reflected by leaves.",
        compute! = (out, Y, p, t) ->
            compute_photosynthetically_active_radiation_reflected!(
                out,
                Y,
                p,
                t,
                land_model,
            ),
    )

    # TPAR - transmitted photosynthetically active radiation
    add_diagnostic_variable!(
        short_name = "tpar",
        long_name = "Transmitted Photosynthetically Active Radiation",
        standard_name = "transmitted_photosynthetically_active_radiation",
        units = "mol photons m^-2 s^-1",
        comments = "The amount of photosynthetically active radiation transmitted by leaves.",
        compute! = (out, Y, p, t) ->
            compute_photosynthetically_active_radiation_transmitted!(
                out,
                Y,
                p,
                t,
                land_model,
            ),
    )

    # Net longwave radiation
    add_diagnostic_variable!(
        short_name = "lwn",
        long_name = "Net Longwave Radiation",
        standard_name = "net_longwave_radiation",
        units = "W m^-2",
        comments = "The net (down minus up) longwave radiation at the surface.",
        compute! = (out, Y, p, t) ->
            compute_radiation_longwave_net!(out, Y, p, t, land_model),
    )

    # Net shortwave radiation
    add_diagnostic_variable!(
        short_name = "swn",
        long_name = "Net Shortwave Radiation",
        standard_name = "net_shortwave_radiation",
        units = "W m^-2",
        comments = "The net (down minus up) shortwave radiation at the surface.",
        compute! = (out, Y, p, t) ->
            compute_radiation_shortwave_net!(out, Y, p, t, land_model),
    )

    ## Drivers Module ##
    # Soil organic carbon
    add_diagnostic_variable!(
        short_name = "soc",
        long_name = "Soil organic carbon",
        standard_name = "soil_organic_carbon",
        units = "kg C m^-3",
        comments = "Mass of organic carbon per volume of soil. (depth resolved)",
        compute! = (out, Y, p, t) ->
            compute_soil_organic_carbon!(out, Y, p, t, land_model),
    )

    # Air pressure
    add_diagnostic_variable!(
        short_name = "airp",
        long_name = "Air pressure",
        standard_name = "air_pressure",
        units = "Pa",
        comments = "The air pressure.",
        compute! = (out, Y, p, t) ->
            compute_pressure!(out, Y, p, t, land_model),
    )

    # Rainfall
    add_diagnostic_variable!(
        short_name = "rain",
        long_name = "Rainfall",
        standard_name = "rainfall",
        units = "m s^-1",
        comments = "Precipitation of liquid water volume (m^3 of water per m^2 of ground per second).",
        compute! = (out, Y, p, t) ->
            compute_rainfall!(out, Y, p, t, land_model),
    )

    # Net longwave radiation
    add_diagnostic_variable!(
        short_name = "lwd",
        long_name = "Down Longwave Radiation",
        standard_name = "down_longwave_radiation",
        units = "W m^-2",
        comments = "The downwelling longwave radiation at the surface.",
        compute! = (out, Y, p, t) ->
            compute_radiation_longwave_down!(out, Y, p, t, land_model),
    )

    # Net shortwave radiation
    add_diagnostic_variable!(
        short_name = "swd",
        long_name = "Short Longwave Radiation",
        standard_name = "short_longwave_radiation",
        units = "W m^-2",
        comments = "The downwelling shortwave radiation at the surface.",
        compute! = (out, Y, p, t) ->
            compute_radiation_shortwave_down!(out, Y, p, t, land_model),
    )

    # Snowfall
    add_diagnostic_variable!(
        short_name = "snow",
        long_name = "Snowfall",
        standard_name = "snowfall",
        units = "m s^-1",
        comments = "The precipitation of snow in liquid water volume (m^3 of water per m^2 of ground per second).",
        compute! = (out, Y, p, t) ->
            compute_snowfall!(out, Y, p, t, land_model),
    )

    # Solar zenith angle
    add_diagnostic_variable!(
        short_name = "sza",
        long_name = "Solar Zenith Angle",
        standard_name = "solar_zenith_angle",
        units = "",
        comments = "Solar zenith angle.",
        compute! = (out, Y, p, t) ->
            compute_solar_zenith_angle!(out, Y, p, t, land_model),
    )

    # Specific humidity
    add_diagnostic_variable!(
        short_name = "qsfc",
        long_name = "Surface Specific Humidity",
        standard_name = "surface_specific_humidity",
        units = "",
        comments = "Ratio of water vapor mass to total moist air parcel mass.",
        compute! = (out, Y, p, t) ->
            compute_specific_humidity!(out, Y, p, t, land_model),
    )

    # Wind speed
    add_diagnostic_variable!(
        short_name = "ws",
        long_name = "Wind Speed",
        standard_name = "wind_speed",
        units = "m s^-1",
        comments = "The average wind speed.",
        compute! = (out, Y, p, t) ->
            compute_wind_speed!(out, Y, p, t, land_model),
    )

    ## Soil Module ##
    # Infiltration
    add_diagnostic_variable!(
        short_name = "infil",
        long_name = "Infiltration",
        standard_name = "infiltration",
        units = "m s^-1",
        comments = "The flux of liquid water volume into the soil (m^3 of water per m^2 of ground per second).",
        compute! = (out, Y, p, t) ->
            compute_infiltration!(out, Y, p, t, land_model),
    )

    # Soil albedo
    add_diagnostic_variable!(
        short_name = "salb",
        long_name = "Soil Albedo",
        standard_name = "surface albedo",
        units = "",
        comments = "The mean of PAR and NIR albedo, which are calculated as α_soil_band = α_band_dry * (1 - S_e) + α_band_wet * S_e.",
        compute! = (out, Y, p, t) ->
            compute_soil_albedo!(out, Y, p, t, land_model),
    )


    # Soil hydraulic conductivity
    add_diagnostic_variable!(
        short_name = "shc",
        long_name = "Soil Hydraulic Conductivity",
        standard_name = "soil_hydraulic_conductivity",
        units = "m s^-1",
        comments = "Soil hydraulic conductivity. (depth resolved)",
        compute! = (out, Y, p, t) ->
            compute_soil_hydraulic_conductivity!(out, Y, p, t, land_model),
    )

    # Soil thermal conductivity
    add_diagnostic_variable!(
        short_name = "stc",
        long_name = "Soil Thermal Conductivity",
        standard_name = "soil_thermal_conductivity",
        units = "W m^-1 K^-1",
        comments = "Soil thermal conductivity. (depth resolved)",
        compute! = (out, Y, p, t) ->
            compute_soil_thermal_conductivity!(out, Y, p, t, land_model),
    )

    # Soil Water Potential
    add_diagnostic_variable!(
        short_name = "swp",
        long_name = "Soil Water Potential",
        standard_name = "soil_water_potential",
        units = "Pa",
        comments = "Soil water potential. (depth resolved)",
        compute! = (out, Y, p, t) ->
            compute_soil_water_potential!(out, Y, p, t, land_model),
    )

    # Soil net radiation
    add_diagnostic_variable!(
        short_name = "soilrn",
        long_name = "Soil Net Radiation",
        standard_name = "soil_net_radiation",
        units = "W m^-2",
        comments = "Net radiation at the soil surface.",
        compute! = (out, Y, p, t) ->
            compute_soil_net_radiation!(out, Y, p, t, land_model),
    )

    ### Soil - Turbulent Fluxes

    # Soil latent heat flux
    add_diagnostic_variable!(
        short_name = "soillhf",
        long_name = "Soil Latent Heat Flux",
        standard_name = "soil_Latent_Heat_Flux",
        units = "W m^-2",
        comments = "Soil latent heat flux, the amount of liquid water evaporated by the soil, expressed in energy units (W m^-2).",
        compute! = (out, Y, p, t) ->
            compute_soil_latent_heat_flux!(out, Y, p, t, land_model),
    )

    # Soil sensible heat flux
    add_diagnostic_variable!(
        short_name = "soilshf",
        long_name = "Soil Sensible Heat Flux",
        standard_name = "soil_sensible_Heat_Flux",
        units = "W m^-2",
        comments = "Soil sensible heat flux, the amount of energy exchanged between the soil and atmosphere to change the temperature of the soil.",
        compute! = (out, Y, p, t) ->
            compute_soil_sensible_heat_flux!(out, Y, p, t, land_model),
    )

    ### Soil - SoilCO2
    # Heterotrophic respiration
    add_diagnostic_variable!(
        short_name = "hr",
        long_name = "Heterotrophic Respiration",
        standard_name = "heterotrophic_respiration",
        units = "kg C m^-2 s^-1",
        comments = "The CO2 efflux at the soil surface due to microbial decomposition of soil organic matter. This is not necessarily equal to CO2 production by microbes, as co2 diffusion through the soil pores takes time.",
        compute! = (out, Y, p, t) ->
            compute_heterotrophic_respiration!(out, Y, p, t, land_model),
    )

    # Soil CO2 diffusivity
    add_diagnostic_variable!(
        short_name = "scd",
        long_name = "Soil CO2 Diffusivity",
        standard_name = "soil_co2_diffusivity",
        units = "m^2 s^-1",
        comments = "The diffusivity of CO2 in the porous phase of the soil. Depends on soil texture, moisture, and temperature. (depth resolved)",
        compute! = (out, Y, p, t) ->
            compute_soilco2_diffusivity!(out, Y, p, t, land_model),
    )

    # Soil CO2 microbial source
    add_diagnostic_variable!(
        short_name = "scms",
        long_name = "Soil CO2 Microbial Source",
        standard_name = "soil_co2_microbial_source",
        units = "kg C m^-3 s^-1",
        comments = "The production of CO2 by microbes in the soil. Vary by layers of soil depth. (depth resolved)",
        compute! = (out, Y, p, t) ->
            compute_soilco2_source_microbe!(out, Y, p, t, land_model),
    )

    ## Other ##
    # Longwave out
    add_diagnostic_variable!(
        short_name = "lwu",
        long_name = "Longwave Radiation Up",
        standard_name = "longwave_radiation_up",
        units = "W m^-2",
        comments = "Upwelling longwave radiation.",
        compute! = (out, Y, p, t) -> compute_lw_up!(out, Y, p, t, land_model),
    )

    # Shortwave out
    add_diagnostic_variable!(
        short_name = "swu",
        long_name = "Shortwave Radiation Up",
        standard_name = "shortwave_radiation_up",
        units = "W m^-2",
        comments = "Upwelling shortwave radiation",
        compute! = (out, Y, p, t) -> compute_sw_up!(out, Y, p, t, land_model),
    )

    # Evapotranspiration
    add_diagnostic_variable!(
        short_name = "et",
        long_name = "Evapotranspiration",
        standard_name = "evapotranspiration",
        units = "kg m^-2 s^-1",
        comments = "Total flux of water mass out of the surface.",
        compute! = (out, Y, p, t) ->
            compute_evapotranspiration!(out, Y, p, t, land_model),
    )

    # Ecosystem respiration
    add_diagnostic_variable!(
        short_name = "er",
        long_name = "Ecosystem Respiration",
        standard_name = "ecosystem respiration",
        units = "kg C m^-2 s^-1",
        comments = "Total respiration flux out of the surface.",
        compute! = (out, Y, p, t) ->
            compute_total_respiration!(out, Y, p, t, land_model),
    )

    # Surface runoff
    add_diagnostic_variable!(
        short_name = "sr",
        long_name = "Surface Runoff",
        standard_name = "surface_runoff",
        units = "m s^-1",
        comments = "Water runoff at the surface, this is the water flowing horizontally above the ground.",
        compute! = (out, Y, p, t) ->
            compute_surface_runoff!(out, Y, p, t, land_model),
    )

    # Ground heat flux
    add_diagnostic_variable!(
        short_name = "ghf",
        long_name = "Ground Heat Flux",
        standard_name = "ground_heat_flux",
        units = "W m^-2",
        comments = "Transfer of heat between the surface and deeper soil layers.",
        compute! = (out, Y, p, t) ->
            compute_ground_heat_flux!(out, Y, p, t, land_model),
    )

    ## Stored in Y (prognostic or state variables) ##

    # Canopy temperature
    add_diagnostic_variable!(
        short_name = "ct",
        long_name = "Canopy Temperature",
        standard_name = "canopy_temperature",
        units = "K",
        comments = "Canopy temperature.",
        compute! = (out, Y, p, t) ->
            compute_canopy_temperature!(out, Y, p, t, land_model),
    )

    # Soil CO2
    add_diagnostic_variable!(
        short_name = "sco2",
        long_name = "Soil CO2",
        standard_name = "soil_co2",
        units = "kg C m^3",
        comments = "Concentration of CO2 in the porous air space of the soil. (depth resolved)",
        compute! = (out, Y, p, t) -> compute_soilco2!(out, Y, p, t, land_model),
    )

    # Soil water content
    add_diagnostic_variable!(
        short_name = "swc",
        long_name = "Soil Water Content",
        standard_name = "soil_water_content",
        units = "m^3 m^-3",
        comments = "The volume of soil water per volume of soil. (depth resolved)",
        compute! = (out, Y, p, t) ->
            compute_soil_water_content!(out, Y, p, t, land_model),
    )

    # Plant water content

    #=
    add_diagnostic_variable!(
        short_name = "pwc",
        long_name = "Plant Water Content",
        standard_name = "plant_water_content",
        units = "m^3 m^-3",
        comments = "The volume of plant water per volume of plant.",
        compute! = (out, Y, p, t) ->
            compute_plant_water_content!(out, Y, p, t, land_model),
    )
    =#
    # return a Tuple

    # Soil ice
    add_diagnostic_variable!(
        short_name = "si",
        long_name = "Soil Ice",
        standard_name = "soil_ice",
        units = "m^3 m^-3",
        comments = "The volume of soil ice per volume of soil. (depth resolved)",
        compute! = (out, Y, p, t) ->
            compute_soil_ice_content!(out, Y, p, t, land_model),
    )

    # Soil internal energy
    add_diagnostic_variable!(
        short_name = "sie",
        long_name = "Soil Internal Energy",
        standard_name = "soil_internal_energy",
        units = "W m^-2",
        comments = "The energy per volume of soil. (depth resolved)",
        compute! = (out, Y, p, t) ->
            compute_soil_internal_energy!(out, Y, p, t, land_model),
    )

    # SWE
    add_diagnostic_variable!(
        short_name = "swe",
        long_name = "Snow water equivalent",
        standard_name = "snow_water_equivalent",
        units = "m",
        comments = "The height of liquid water if all snow melted",
        compute! = (out, Y, p, t) ->
            compute_snow_water_equivalent!(out, Y, p, t, land_model),
    )

    # Snow depth
    add_diagnostic_variable!(
        short_name = "snd",
        long_name = "Snow depth",
        standard_name = "snow_depth",
        units = "m",
        comments = "The snow depth",
        compute! = (out, Y, p, t) ->
            compute_snow_depth!(out, Y, p, t, land_model),
    )

end
