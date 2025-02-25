"""
This file contains the PFT struct used to prescribe vegetation parameters to the 
model when running using PFTs, as well as the default PFT scheme. ClimaLand 
supports running using default PFTs, user defined PFTs, or fully prescribed 
vegetation parameters.

The default PFTs used in the model use parameter values drawn from different 
sources in the literature. The sources are listed here with abbreviations that
are used in the comments in the code below.

    Chen, J. M., G. Mo, J. Pisek, J. Liu, F. Deng, M. Ishizawa, 
    and D. Chan (2012), Effects of foliage clumping on the estimation of global
    terrestrial gross primary productivity, Global Biogeochem. Cycles, 26, 
    GB1019, doi:10.1029/2010GB003996.
    abb. as Chen et al. 2012

    CLM5.0 Technical Note
    abb. as CLM5.0

    CLM2 PFT Data
    https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata/lnd/clm2/pftdata/
    abb. as CLM2 PFT Data

    Ma, X., Jin, J., Liu, J. et al. An improved vegetation emissivity scheme for
    land surface modeling and its impact on snow cover simulations. Clim Dyn 53,
    6215–6226 (2019). https://doi.org/10.1007/s00382-019-04924-9
    abb. as Ma et al. 2019

    Bonan, G. B., Patton, E. G., Harman, I. N., Oleson, K. W., Finnigan, J. J.,
    Lu, Y., and Burakowski, E. A.: Modeling canopy-induced turbulence in the
    Earth system: a unified parameterization of turbulent exchange within plant
    canopies and the roughness sublayer (CLM-ml v0), Geosci. Model Dev., 11,
    1467–1496, https://doi.org/10.5194/gmd-11-1467-2018, 2018.
    abb. as Bonan et al. 2018
"""

export Pft, default_pfts, params_from_pfts, pft_param_list

# List of parameters that PFTs need to define in order to be valid
pft_param_list = [
    # TwoStreamModel parameters
    :Ω,          # Clumping Index
    :α_PAR_leaf, # PAR Leaf Reflectance
    :α_NIR_leaf, # Near-infrared Leaf Reflectance
    :τ_PAR_leaf, # PAR Leaf Transmittance
    :τ_NIR_leaf, # Near-infrared Leaf Transmittance
    :ϵ_canopy,   # Canopy Emissivity
    :χl,         # Leaf/Stem orientation Index
    :ac_canopy,  # Canopy Specific Heat per Area

    # Conductance Model
    :g1, # Medlyn CO2 Sensitivity

    # Photosynthesis model
    :Vcmax25, # Max rate of carboxylation of Rubisco
    # When running optimality Farquhar model, this param is not used

    # Plant Hydraulics and general plant parameters
    :f_root_to_shoot, # Root to Shoot Ratio
    :K_sat_plant,     # Hydraulic Conductivity at Saturation
    :ψ63,             # Xylem water potential at 63% loss of conductivity
    :plant_ν,         # Plant porosity
    :rooting_depth,   # Plant Rooting Depth
]


"""
Define a PFT type that can be used to store the parameters for each PFT.
Each PFT has a name and a list of parameters. The parameters are stored as 
NamedTuples mapping parameter names to values. The inner constructor checks to
ensure all required parameters are defined for the PFT.
"""
struct Pft{NT <: NamedTuple}
    name::String
    parameters::NT

    # Inner constructor to verify all required parameters are defined
    function Pft(name, parameters)
        defined_param_names = keys(parameters)
        for param in pft_param_list
            if param ∉ defined_param_names
                error("PFT $name is missing parameter $param")
            end
        end
        new{typeof(parameters)}(name, parameters)
    end
end


NET_Temp = Pft(
    "Temperate Needleleaf Evergreen Tree",
    (
        # Radiative Transfer Model
        Ω = 0.74,          # Unitless ⋅ Chen et al. 2012 Table 1
        α_PAR_leaf = 0.07, # Unitless ⋅ CLM5.0 Table 2.3.1
        α_NIR_leaf = 0.35, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_PAR_leaf = 0.05, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_NIR_leaf = 0.10, # Unitless ⋅ CLM5.0 Table 2.3.1
        ϵ_canopy = 0.982,  # Unitless ⋅ Ma et al. 2019
        χl = 0.01,         # Unitless ⋅ CLM2 PFT Data
        ac_canopy = 2234,  # Jm^-2K^-1 ⋅ Bonan et al. 2018

        # Stomatal Conductance Model
        g1 = 74.31, # kPa^(1/2) ⋅ CLM5.0 Table 2.9.1

        # Photosynthesis Model
        Vcmax25 = 5.1e-5, # mol CO2/m^2/s ⋅ CLM2 PFT Data

        # Plant Hydraulics and general plant parameters
        f_root_to_shoot = 1.0, # Unitless ⋅ CLM2 PFT Data
        K_sat_plant = 5e-9,    # m/s
        ψ63 = -4 / 0.0098,     # / MPa to m
        plant_ν = 2e-4,        # Unitless
        rooting_depth = 3.90,  # m ⋅ Bonan Table 2.3
    ),
)

NET_Bor = Pft(
    "Boreal Needleleaf Evergreen Tree",
    (
        # Radiative Transfer Model
        Ω = 0.74,          # Unitless ⋅ Chen et al. 2012 Table 1
        α_PAR_leaf = 0.07, # Unitless ⋅ CLM5.0 Table 2.3.1
        α_NIR_leaf = 0.35, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_PAR_leaf = 0.05, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_NIR_leaf = 0.10, # Unitless ⋅ CLM5.0 Table 2.3.1
        ϵ_canopy = 0.982,  # Unitless ⋅ Ma et al. 2019
        χl = 0.01,         # Unitless ⋅ CLM2 PFT Data
        ac_canopy = 2792,  # Jm^-2K^-1 ⋅ Bonan et al. 2018

        # Stomatal Conductance Model
        g1 = 74.31, # kPa^(1/2) ⋅ CLM5.0 Table 2.9.1

        # Photosynthesis Model
        Vcmax25 = 4.3e-5, # mol CO2/m^2/s ⋅ CLM2 PFT Data

        # Plant Hydraulics and general plant parameters
        f_root_to_shoot = 1.0, # Unitless ⋅ CLM2 PFT Data
        K_sat_plant = 5e-9,    # m/s
        ψ63 = -4 / 0.0098,     # / MPa to m
        plant_ν = 2e-4,        # Unitless
        rooting_depth = 2.0,   # m ⋅ Bonan Table 2.3
    ),
)

NDT_Bor = Pft(
    "Boreal Needleleaf Deciduous Tree",
    (
        # Radiative Transfer Model
        Ω = 0.78,          # Unitless ⋅ Chen et al. 2012 Table 1
        α_PAR_leaf = 0.07, # Unitless ⋅ CLM5.0 Table 2.3.1
        α_NIR_leaf = 0.35, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_PAR_leaf = 0.05, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_NIR_leaf = 0.10, # Unitless ⋅ CLM5.0 Table 2.3.1
        ϵ_canopy = 0.985,  # Unitless ⋅ Ma et al. 2019
        χl = 0.01,         # Unitless ⋅ CLM2 PFT Data
        ac_canopy = 2792,  # Jm^-2K^-1 ⋅ Bonan et al. 2018

        # Stomatal Conductance Model
        g1 = 74.31, # kPa^(1/2) ⋅ CLM5.0 Table 2.9.1

        # Photosynthesis Model
        Vcmax25 = 5.1e-5, # mol CO2/m^2/s ⋅ CLM2 PFT Data

        # Plant Hydraulics and general plant parameters
        f_root_to_shoot = 1.0, # Unitless ⋅ CLM2 PFT Data
        K_sat_plant = 5e-9,    # m/s
        ψ63 = -4 / 0.0098,     # / MPa to m
        plant_ν = 2e-4,        # Unitless
        rooting_depth = 2.0,   # m ⋅ Bonan Table 2.3
    ),
)

BET_Trop = Pft(
    "Tropical Broadleaf Evergreen Tree",
    (
        # Radiative Transfer Model
        Ω = 0.66,          # Unitless ⋅ Chen et al. 2012 Table 1
        α_PAR_leaf = 0.10, # Unitless ⋅ CLM5.0 Table 2.3.1
        α_NIR_leaf = 0.45, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_PAR_leaf = 0.05, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_NIR_leaf = 0.25, # Unitless ⋅ CLM5.0 Table 2.3.1
        ϵ_canopy = 0.978,  # Unitless ⋅ Ma et al. 2019
        χl = 0.1,          # Unitless ⋅ CLM2 PFT Data
        ac_canopy = 2.5e3, # Jm^-2K^-1 ⋅ Bonan et al. 2018

        # Stomatal Conductance Model
        g1 = 130.29, # kPa^(1/2) ⋅ CLM5.0 Table 2.9.1

        # Photosynthesis Model
        Vcmax25 = 7.5e-5, # mol CO2/m^2/s ⋅ CLM2 PFT Data

        # Plant Hydraulics and general plant parameters
        f_root_to_shoot = 1.0, # Unitless ⋅ CLM2 PFT Data
        K_sat_plant = 5e-9,    # m/s
        ψ63 = -4 / 0.0098,     # / MPa to m
        plant_ν = 2e-4,        # Unitless
        rooting_depth = 7.30,  # m ⋅ Bonan Table 2.3
    ),
)

BET_Temp = Pft(
    "Temperate Broadleaf Evergreen Tree",
    (
        # Radiative Transfer Model
        Ω = 0.66,          # Unitless ⋅ Chen et al. 2012 Table 1
        α_PAR_leaf = 0.10, # Unitless ⋅ CLM5.0 Table 2.3.1
        α_NIR_leaf = 0.45, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_PAR_leaf = 0.05, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_NIR_leaf = 0.25, # Unitless ⋅ CLM5.0 Table 2.3.1
        ϵ_canopy = 0.981,  # Unitless ⋅ Ma et al. 2019
        χl = 0.1,          # Unitless ⋅ CLM2 PFT Data
        ac_canopy = 2.5e3, # Jm^-2K^-1 ⋅ Bonan et al. 2018

        # Stomatal Conductance Model
        g1 = 130.29, # kPa^(1/2) ⋅ CLM5.0 Table 2.9.1

        # Photosynthesis Model
        Vcmax25 = 6.9e-5, # mol CO2/m^2/s ⋅ CLM2 PFT Data

        # Plant Hydraulics and general plant parameters
        f_root_to_shoot = 1.0, # Unitless ⋅ CLM2 PFT Data
        K_sat_plant = 5e-9,    # m/s
        ψ63 = -4 / 0.0098,     # / MPa to m
        plant_ν = 2e-4,        # Unitless
        rooting_depth = 2.90,  # m ⋅ Bonan Table 2.3
    ),
)

BDT_Trop = Pft(
    "Tropical Broadleaf Deciduous Tree",
    (
        # Radiative Transfer Model
        Ω = 0.70,          # Unitless ⋅ Chen et al. 2012 Table 1
        α_PAR_leaf = 0.10, # Unitless ⋅ CLM5.0 Table 2.3.1
        α_NIR_leaf = 0.45, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_PAR_leaf = 0.05, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_NIR_leaf = 0.25, # Unitless ⋅ CLM5.0 Table 2.3.1
        ϵ_canopy = 0.982,  # Unitless ⋅ Ma et al. 2019
        χl = 0.01,         # Unitless ⋅ CLM2 PFT Data
        ac_canopy = 2.5e3, # Jm^-2K^-1 ⋅ Bonan et al. 2018

        # Stomatal Conductance Model
        g1 = 140.72, # kPa^(1/2) ⋅ CLM5.0 Table 2.9.1

        # Photosynthesis Model
        Vcmax25 = 4e-5, # mol CO2/m^2/s ⋅ CLM2 PFT Data

        # Plant Hydraulics and general plant parameters
        f_root_to_shoot = 1.0, # Unitless ⋅ CLM2 PFT Data
        K_sat_plant = 5e-9,    # m/s
        ψ63 = -4 / 0.0098,     # / MPa to m
        plant_ν = 2e-4,        # Unitless
        rooting_depth = 3.70,  # m ⋅ Bonan Table 2.3
    ),
)

BDT_Temp = Pft(
    "Temperate Broadleaf Deciduous Tree",
    (
        # Radiative Transfer Model
        Ω = 0.70,          # Unitless ⋅ Chen et al. 2012 Table 1
        α_PAR_leaf = 0.10, # Unitless ⋅ CLM5.0 Table 2.3.1
        α_NIR_leaf = 0.45, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_PAR_leaf = 0.05, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_NIR_leaf = 0.25, # Unitless ⋅ CLM5.0 Table 2.3.1
        ϵ_canopy = 0.970,  # Unitless ⋅ Ma et al. 2019
        χl = 0.25,         # Unitless ⋅ CLM2 PFT Data
        ac_canopy = 2.5e3, # Jm^-2K^-1 ⋅ Bonan et al. 2018

        # Stomatal Conductance Model
        g1 = 140.72, # kPa^(1/2) ⋅ CLM5.0 Table 2.9.1

        # Photosynthesis Model
        Vcmax25 = 5.1e-5, # mol CO2/m^2/s ⋅ CLM2 PFT Data

        # Plant Hydraulics and general plant parameters
        f_root_to_shoot = 1.0, # Unitless ⋅ CLM2 PFT Data
        K_sat_plant = 5e-9,    # m/s
        ψ63 = -4 / 0.0098,     # / MPa to m
        plant_ν = 2e-4,        # Unitless
        rooting_depth = 2.90,  # m ⋅ Bonan Table 2.3
    ),
)

BDT_Bor = Pft(
    "Boreal Broadleaf Deciduous Tree",
    (
        # Radiative Transfer Model
        Ω = 0.70,          # Unitless ⋅ Chen et al. 2012 Table 1
        α_PAR_leaf = 0.10, # Unitless ⋅ CLM5.0 Table 2.3.1
        α_NIR_leaf = 0.45, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_PAR_leaf = 0.05, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_NIR_leaf = 0.25, # Unitless ⋅ CLM5.0 Table 2.3.1
        ϵ_canopy = 0.968,  # Unitless ⋅ Ma et al. 2019
        χl = 0.25,         # Unitless ⋅ CLM2 PFT Data
        ac_canopy = 745,   # Jm^-2K^-1 ⋅ Bonan et al. 2018

        # Stomatal Conductance Model
        g1 = 140.72, # kPa^(1/2) ⋅ CLM5.0 Table 2.9.1

        # Photosynthesis Model
        Vcmax25 = 5.1e-5, # mol CO2/m^2/s ⋅ CLM2 PFT Data

        # Plant Hydraulics and general plant parameters
        f_root_to_shoot = 1.0, # Unitless ⋅ CLM2 PFT Data
        K_sat_plant = 5e-9,    # m/s
        ψ63 = -4 / 0.0098,     # / MPa to m
        plant_ν = 2e-4,        # Unitless
        rooting_depth = 2.00,  # m ⋅ Bonan Table 2.3
    ),
)

BES_Temp = Pft(
    "Temperate Broadleaf Evergreen Shrub",
    (
        # Radiative Transfer Model
        Ω = 0.75,          # Unitless ⋅ Chen et al. 2012 Table 1
        α_PAR_leaf = 0.07, # Unitless ⋅ CLM5.0 Table 2.3.1
        α_NIR_leaf = 0.35, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_PAR_leaf = 0.05, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_NIR_leaf = 0.10, # Unitless ⋅ CLM5.0 Table 2.3.1
        ϵ_canopy = 0.987,  # Unitless ⋅ Ma et al. 2019
        χl = 0.01,         # Unitless ⋅ CLM2 PFT Data
        ac_canopy = 745,   # Jm^-2K^-1 ⋅ Bonan et al. 2018

        # Stomatal Conductance Model
        g1 = 148.63, # kPa^(1/2) ⋅ CLM5.0 Table 2.9.1

        # Photosynthesis Model
        Vcmax25 = 1.7e-5, # mol CO2/m^2/s ⋅ CLM2 PFT Data

        # Plant Hydraulics and general plant parameters
        f_root_to_shoot = 1.0, # Unitless ⋅ CLM2 PFT Data
        K_sat_plant = 5e-9,    # m/s
        ψ63 = -4 / 0.0098,     # / MPa to m
        plant_ν = 5e-4,        # Unitless
        rooting_depth = 5.20,  # m ⋅ Bonan Table 2.3
    ),
)

BDS_Temp = Pft(
    "Temperate Broadleaf Deciduous Shrub",
    (
        # Radiative Transfer Model
        Ω = 0.75,          # Unitless ⋅ Chen et al. 2012 Table 1
        α_PAR_leaf = 0.10, # Unitless ⋅ CLM5.0 Table 2.3.1
        α_NIR_leaf = 0.45, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_PAR_leaf = 0.05, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_NIR_leaf = 0.25, # Unitless ⋅ CLM5.0 Table 2.3.1
        ϵ_canopy = 0.987,  # Unitless ⋅ Ma et al. 2019
        χl = 0.25,         # Unitless ⋅ CLM2 PFT Data
        ac_canopy = 745,   # Jm^-2K^-1 ⋅ Bonan et al. 2018

        # Stomatal Conductance Model
        g1 = 148.63, # kPa^(1/2) ⋅ CLM5.0 Table 2.9.1

        # Photosynthesis Model
        Vcmax25 = 1.7e-5, # mol CO2/m^2/s ⋅ CLM2 PFT Data

        # Plant Hydraulics and general plant parameters
        f_root_to_shoot = 1.0, # Unitless ⋅ CLM2 PFT Data
        K_sat_plant = 5e-9,    # m/s
        ψ63 = -4 / 0.0098,     # / MPa to m
        plant_ν = 5e-4,        # Unitless
        rooting_depth = 5.20,  # m ⋅ Bonan Table 2.3
    ),
)

BDS_Bor = Pft(
    "Boreal Broadleaf Deciduous Shrub",
    (
        # Radiative Transfer Model
        Ω = 0.75,          # Unitless ⋅ Chen et al. 2012 Table 1
        α_PAR_leaf = 0.10, # Unitless ⋅ CLM5.0 Table 2.3.1
        α_NIR_leaf = 0.45, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_PAR_leaf = 0.05, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_NIR_leaf = 0.25, # Unitless ⋅ CLM5.0 Table 2.3.1
        ϵ_canopy = 0.987,  # Unitless ⋅ Ma et al. 2019
        χl = 0.25,         # Unitless ⋅ CLM2 PFT Data
        ac_canopy = 745,   # Jm^-2K^-1 ⋅ Bonan et al. 2018

        # Stomatal Conductance Model
        g1 = 148.63, # kPa^(1/2) ⋅ CLM5.0 Table 2.9.1

        # Photosynthesis Model
        Vcmax25 = 3.3e-5, # mol CO2/m^2/s ⋅ CLM2 PFT Data

        # Plant Hydraulics and general plant parameters
        f_root_to_shoot = 1.0, # Unitless ⋅ CLM2 PFT Data
        K_sat_plant = 5e-9,    # m/s
        ψ63 = -4 / 0.0098,     # / MPa to m
        plant_ν = 5e-4,        # Unitless
        rooting_depth = 5.20,  # m ⋅ Bonan Table 2.3
    ),
)

C3G_A = Pft(
    "C3 Arctic Grass",
    (
        # Radiative Transfer Model
        Ω = 0.76,          # Unitless ⋅ Chen et al. 2012 Table 1
        α_PAR_leaf = 0.11, # Unitless ⋅ CLM5.0 Table 2.3.1
        α_NIR_leaf = 0.35, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_PAR_leaf = 0.05, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_NIR_leaf = 0.34, # Unitless ⋅ CLM5.0 Table 2.3.1
        ϵ_canopy = 0.978,  # Unitless ⋅ Ma et al. 2019
        χl = -0.3,         # Unitless ⋅ CLM2 PFT Data
        ac_canopy = 745,   # Jm^-2K^-1 ⋅ Bonan et al. 2018

        # Stomatal Conductance Model
        g1 = 70.2, # kPa^(1/2) ⋅ CLM5.0 Table 2.9.1

        # Photosynthesis Model
        Vcmax25 = 4.3e-5, # mol CO2/m^2/s ⋅ CLM2 PFT Data

        # Plant Hydraulics and general plant parameters
        f_root_to_shoot = 3.0,   # Unitless ⋅ CLM2 PFT Data
        K_sat_plant = 5e-9,      # m/s
        ψ63 = -2.7 / 0.0098,     # / MPa to m
        plant_ν = 8e-4,          # Unitless
        rooting_depth = 2.60,    # m ⋅ Bonan Table 2.3
    ),
)

C3G_NA = Pft(
    "Non-arctic C3 Grass",
    (
        # Radiative Transfer Model
        Ω = 0.76,          # Unitless ⋅ Chen et al. 2012 Table 1
        α_PAR_leaf = 0.11, # Unitless ⋅ CLM5.0 Table 2.3.1
        α_NIR_leaf = 0.35, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_PAR_leaf = 0.05, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_NIR_leaf = 0.34, # Unitless ⋅ CLM5.0 Table 2.3.1
        ϵ_canopy = 0.978,  # Unitless ⋅ Ma et al. 2019
        χl = -0.3,         # Unitless ⋅ CLM2 PFT Data
        ac_canopy = 745,   # Jm^-2K^-1 ⋅ Bonan et al. 2018

        # Stomatal Conductance Model
        g1 = 166.02, # kPa^(1/2) ⋅ CLM5.0 Table 2.9.1

        # Photosynthesis Model
        Vcmax25 = 4.3e-5, # mol CO2/m^2/s ⋅ CLM2 PFT Data

        # Plant Hydraulics and general plant parameters
        f_root_to_shoot = 3.0,   # Unitless ⋅ CLM2 PFT Data
        K_sat_plant = 5e-9,      # m/s
        ψ63 = -2.7 / 0.0098,     # / MPa to m
        plant_ν = 8e-4,          # Unitless
        rooting_depth = 2.60,    # m ⋅ Bonan Table 2.3
    ),
)

C4G = Pft(
    "C4 Grass",
    (
        # Radiative Transfer Model
        Ω = 0.75,          # Unitless ⋅ Chen et al. 2012 Table 1
        α_PAR_leaf = 0.11, # Unitless ⋅ CLM5.0 Table 2.3.1
        α_NIR_leaf = 0.35, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_PAR_leaf = 0.05, # Unitless ⋅ CLM5.0 Table 2.3.1
        τ_NIR_leaf = 0.34, # Unitless ⋅ CLM5.0 Table 2.3.1
        ϵ_canopy = 0.978,  # Unitless ⋅ Ma et al. 2019
        χl = -0.3,         # Unitless ⋅ CLM2 PFT Data
        ac_canopy = 745,   # Jm^-2K^-1 ⋅ Bonan et al. 2018

        # Stomatal Conductance Model
        g1 = 51.22, # kPa^(1/2) ⋅ CLM5.0 Table 2.9.1

        # Photosynthesis Model
        Vcmax25 = 2.4e-5, # mol CO2/m^2/s ⋅ CLM2 PFT Data

        # Plant Hydraulics and general plant parameters
        f_root_to_shoot = 3.0,   # Unitless ⋅ CLM2 PFT Data
        K_sat_plant = 5e-9,      # m/s
        ψ63 = -2.7 / 0.0098,     # / MPa to m
        plant_ν = 8e-4,          # Unitless
        rooting_depth = 2.60,    # m ⋅ Bonan Table 2.3
    ),
)

# Lists of defaults to pass to parameter constructing function
default_pfts = [
    NET_Temp,
    NET_Bor,
    NDT_Bor,
    BET_Trop,
    BET_Temp,
    BDT_Trop,
    BDT_Temp,
    BDT_Bor,
    BES_Temp,
    BDS_Temp,
    BDT_Bor,
    C3G_A,
    C3G_NA,
    C4G,
]

function params_from_pfts(
    pft_cover::Vector{Float64},
    pfts::Vector{Pft} = default_pfts,
)
    """
    Takes in a vector of PFT cover percentages and returns the correct parameter set
    corresponding to the most dominant PFT by cover percentage.

    May optionally take in a vector of PFTs to use instead of the default PFTs.
    In this case, the pfts_cover vector must be the same length and in the
    corresponding order to the PFTs in the pfts vector. This allows a user to
    define their own PFT scheme and plug it into the model.
    """
    # Find the index of the dominant PFT by cover percentage
    max_ind = argmax(pft_cover)

    # Extract the parameter set for the dominant PFT
    param_set = []
    dominant_pft_params = pfts[max_ind].parameters

    # Collect the parameters in the correct order
    for param in pft_param_list
        append!(param_set, getfield(dominant_pft_params, param))
    end

    return param_set
end
