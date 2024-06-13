using ArtifactWrappers
#currently not being used
#=
# Read in f_max data and land sea mask
topmodel_dataset = ArtifactWrapper(
    @__DIR__,
    "processed_topographic_index 2.5 deg",
    ArtifactFile[ArtifactFile(
        url = "https://caltech.box.com/shared/static/dwa7g0uzhxd50a2z3thbx3a12n0r887s.nc",
        filename = "means_2.5_new.nc",
    ),],
)
infile_path = joinpath(get_data_folder(topmodel_dataset), "means_2.5_new.nc")

f_max = SpaceVaryingInput(
    infile_path,
    "fmax",
    surface_space;
    regridder_type
)
mask = SpaceVaryingInput(
    infile_path,
    "landsea_mask",
    surface_space;
    regridder_type
)

oceans_to_zero(field, mask) = mask > 0.5 ? field : eltype(field)(0)
f_over = FT(3.28) # 1/m
R_sb = FT(1.484e-4 / 1000) # m/s
runoff_model = ClimaLand.Soil.Runoff.TOPMODELRunoff{FT}(;
    f_over = f_over,
    f_max = f_max,
    R_sb = R_sb,
)
=#
# TODO: Change with @clima_artifact("soil_params_Gupta2020_2022", context)
soil_params_artifact_path = "/groups/esm/ClimaArtifacts/artifacts/soil_params_Gupta2020_2022"
extrapolation_bc =
    (Interpolations.Periodic(), Interpolations.Flat(), Interpolations.Flat())
soil_params_mask = SpaceVaryingInput(
        joinpath(
            soil_params_artifact_path,
            "vGalpha_map_gupta_etal2020_2.5x2.5x4.nc",
        ),
        "α",
        subsurface_space;
        regridder_type,
        regridder_kwargs = (; extrapolation_bc,),
        file_reader_kwargs = (; preprocess_func = (data) -> data > 0,),
    )
oceans_to_value(field, mask, value) = mask == 1.0 ? field : eltype(field)(value)    

vg_α = SpaceVaryingInput(
    joinpath(
        soil_params_artifact_path,
        "vGalpha_map_gupta_etal2020_2.5x2.5x4.nc",
    ),
    "α",
    subsurface_space;
    regridder_type,
    regridder_kwargs = (; extrapolation_bc,),
)
vg_n = SpaceVaryingInput(
    joinpath(soil_params_artifact_path, "vGn_map_gupta_etal2020_2.5x2.5x4.nc"),
    "n",
    subsurface_space;
    regridder_type,
    regridder_kwargs = (; extrapolation_bc,),
)
x  = parent(vg_α)
μ = mean(log10.(x[x .>0]))
vg_α .= oceans_to_value.(vg_α, soil_params_mask, 10.0^μ)

x  = parent(vg_n)
μ = mean(x[x .>0])
vg_n .= oceans_to_value.(vg_n, soil_params_mask, μ)

vg_fields_to_hcm_field(α::FT, n::FT) where {FT} =
    ClimaLand.Soil.vanGenuchten{FT}(; @NamedTuple{α::FT, n::FT}((α, n))...)
hydrology_cm = vg_fields_to_hcm_field.(vg_α, vg_n)

θ_r = SpaceVaryingInput(
    joinpath(
        soil_params_artifact_path,
        "residual_map_gupta_etal2020_2.5x2.5x4.nc",
    ),
    "θ_r",
    subsurface_space;
    regridder_type,
    regridder_kwargs = (; extrapolation_bc,),
)

ν = SpaceVaryingInput(
    joinpath(
        soil_params_artifact_path,
        "porosity_map_gupta_etal2020_2.5x2.5x4.nc",
    ),
    "ν",
    subsurface_space;
    regridder_type,
    regridder_kwargs = (; extrapolation_bc,),
)
K_sat = SpaceVaryingInput(
    joinpath(soil_params_artifact_path, "ksat_map_gupta_etal2020_2.5x2.5x4.nc"),
    "Ksat",
    subsurface_space;
    regridder_type,
    regridder_kwargs = (; extrapolation_bc,),
)

x  = parent(K_sat)
μ = mean(log10.(x[x .>0]))
K_sat .= oceans_to_value.(K_sat, soil_params_mask, 10.0^μ)

ν .= oceans_to_value.(ν, soil_params_mask, 1)

θ_r .= oceans_to_value.(θ_r, soil_params_mask, 0)


S_s = oceans_to_value.(ClimaCore.Fields.zeros(subsurface_space) .+ FT(1e-3), soil_params_mask, 1)
ν_ss_om = oceans_to_value.(ClimaCore.Fields.zeros(subsurface_space) .+ FT(0.1), soil_params_mask, 0)
ν_ss_quartz = oceans_to_value.(ClimaCore.Fields.zeros(subsurface_space) .+ FT(0.1),soil_params_mask, 0)
ν_ss_gravel = oceans_to_value.(ClimaCore.Fields.zeros(subsurface_space) .+ FT(0.1),soil_params_mask, 0)
PAR_albedo = ClimaCore.Fields.zeros(surface_space) .+ FT(0.2)
NIR_albedo = ClimaCore.Fields.zeros(surface_space) .+ FT(0.2)
soil_params = Soil.EnergyHydrologyParameters(
    FT;
    ν,
    ν_ss_om,
    ν_ss_quartz,
    ν_ss_gravel,
    hydrology_cm,
    K_sat,
    S_s,
    θ_r,
    PAR_albedo = PAR_albedo,
    NIR_albedo = NIR_albedo,
);


# TwoStreamModel parameters
Ω = FT(0.69)
ld = FT(0.5)
α_PAR_leaf = FT(0.1)
τ_PAR_leaf = FT(0.05)
α_NIR_leaf = FT(0.45)
τ_NIR_leaf = FT(0.25)

# Energy Balance model
ac_canopy = FT(2.5e4)

# Conductance Model
g1 = FT(141) # Wang et al: 141 sqrt(Pa) for Medlyn model; Natan used 300.

#Photosynthesis model
Vcmax25 = FT(9e-5) # from Yujie's paper 4.5e-5 , Natan used 9e-5

# Plant Hydraulics and general plant parameters
SAI = FT(0.0) # m2/m2
f_root_to_shoot = FT(3.5)
K_sat_plant = 5e-9 # m/s # seems much too small?
ψ63 = FT(-4 / 0.0098) # / MPa to m, Holtzman's original parameter value is -4 MPa
Weibull_param = FT(4) # unitless, Holtzman's original c param value
a = FT(0.05 * 0.0098) # Holtzman's original parameter for the bulk modulus of elasticity
conductivity_model =
    Canopy.PlantHydraulics.Weibull{FT}(K_sat_plant, ψ63, Weibull_param)
retention_model = Canopy.PlantHydraulics.LinearRetentionCurve{FT}(a)
plant_ν = FT(1.44e-4)
plant_S_s = FT(1e-2 * 0.0098) # m3/m3/MPa to m3/m3/m
rooting_depth = FT(0.5) # from Natan
n_stem = 0
n_leaf = 1
h_stem = 0.0
h_leaf = 1.0
zmax = 0.0
h_canopy = h_stem + h_leaf
compartment_midpoints =
    n_stem > 0 ? [h_stem / 2, h_stem + h_leaf / 2] : [h_leaf / 2]
compartment_surfaces = n_stem > 0 ? [zmax, h_stem, h_canopy] : [zmax, h_leaf]

z0_m = FT(0.13) * h_canopy
z0_b = FT(0.1) * z0_m


soilco2_ps = Soil.Biogeochemistry.SoilCO2ModelParameters(
    FT;
    ν = 1.0,# INCORRECT!
)
