export run_fluxnet

"""
    fluxnet_simulation(site_ID; kwargs)

Run ClimaLand at a fluxnet site.
"""
function run_fluxnet(
    site_ID;
    FT = Float64,
    context = ClimaComms.SingletonCommsContext(),
    setup = make_setup(site_ID),
    domain = make_domain(setup, FT),
    config = make_config(site_ID),
    params = make_parameters(site_ID),
    drivers = make_drivers(site_ID, setup, config, params, context),
    timestepper = make_timestepper(setup),
)
    #earth_param_set = create_lsm_parameters(FT)

    # Now we set up the model. For the soil model, we pick
    # a model type and model args:
    soil_domain = domain.land_domain
    soil_ps = Soil.EnergyHydrologyParameters(
        FT;
        ν = params.soil.ν,
        ν_ss_om = params.soil.ν_ss_om,
        ν_ss_quartz = params.soil.ν_ss_quartz,
        ν_ss_gravel = params.soil.ν_ss_gravel,
        hydrology_cm = params.soil.hydrology_cm,
        K_sat = params.soil.K_sat,
        S_s = params.soil.S_s,
        θ_r = params.soil.θ_r,
        z_0m = params.soil.z_0m,
        z_0b = params.soil.z_0b,
        emissivity = params.soil.emissivity,
        PAR_albedo = params.soil.PAR_albedo,
        NIR_albedo = params.soil.NIR_albedo,
    )

    soil_args = (domain = soil_domain, parameters = soil_ps)
    soil_model_type = Soil.EnergyHydrology{FT}

    # Soil microbes model
    soilco2_type = Soil.Biogeochemistry.SoilCO2Model{FT}

    soilco2_ps = SoilCO2ModelParameters(
        FT;
        ν = params.soil.ν,
        θ_a100 = params.hetero_resp.θ_a100,
        D_ref = params.hetero_resp.D_ref,
        b = params.hetero_resp.b,
        D_liq = params.hetero_resp.D_liq,
        # DAMM
        α_sx = params.hetero_resp.α_sx,
        Ea_sx = params.hetero_resp.Ea_sx,
        kM_sx = params.hetero_resp.kM_sx,
        kM_o2 = params.hetero_resp.kM_o2,
        O2_a = params.hetero_resp.O2_a,
        D_oa = params.hetero_resp.D_oa,
        p_sx = params.hetero_resp.p_sx,
        earth_param_set = earth_param_set,
    )

    # soil microbes args
    Csom = (z, t) -> eltype(z)(5.0)

    soilco2_top_bc = Soil.Biogeochemistry.AtmosCO2StateBC()
    soilco2_bot_bc = Soil.Biogeochemistry.SoilCO2FluxBC((p, t) -> 0.0) # no flux
    soilco2_sources = (MicrobeProduction{FT}(),)

    soilco2_boundary_conditions =
        (; top = soilco2_top_bc, bottom = CO2 = soilco2_bot_bc)

    soilco2_drivers = Soil.Biogeochemistry.SoilDrivers(
        Soil.Biogeochemistry.PrognosticMet{FT}(),
        Soil.Biogeochemistry.PrescribedSOC{FT}(Csom),
        drivers.atmos,
    )

    soilco2_args = (;
        boundary_conditions = soilco2_boundary_conditions,
        sources = soilco2_sources,
        domain = soil_domain,
        parameters = soilco2_ps,
        drivers = soilco2_drivers,
    )

    # Now we set up the canopy model, which we set up by component:
    # Component Types
    canopy_component_types = (;
        autotrophic_respiration = Canopy.AutotrophicRespirationModel{FT},
        radiative_transfer = Canopy.TwoStreamModel{FT},
        photosynthesis = Canopy.FarquharModel{FT},
        conductance = Canopy.MedlynConductanceModel{FT},
        hydraulics = Canopy.BigLeafHydraulicsModel{FT},
        energy = Canopy.BigLeafEnergyModel{FT},
    )
    # Individual Component arguments
    # Set up autotrophic respiration
    autotrophic_respiration_args = (;
        parameters = AutotrophicRespirationParameters{FT}(;
            ne = params.auto_resp.ne,
            ηsl = params.auto_resp.ηsl,
            σl = params.auto_resp.σl,
            μr = params.auto_resp.μr,
            μs = params.auto_resp.μs,
            f1 = params.auto_resp.f1,
            f2 = params.auto_resp.f2,
        )
    )
    # Set up radiative transfer
    radiative_transfer_args = (;
        parameters = TwoStreamParameters(
            FT;
            Ω = params.radiative_transfer.Ω,
            G_Function = params.radiative_transfer.G_Function,
            α_PAR_leaf = params.radiative_transfer.α_PAR_leaf,
            λ_γ_PAR = params.radiative_transfer.λ_γ_PAR,
            λ_γ_NIR = params.radiative_transfer.λ_γ_NIR,
            τ_PAR_leaf = params.radiative_transfer.τ_PAR_leaf,
            α_NIR_leaf = params.radiative_transfer.α_NIR_leaf,
            τ_NIR_leaf = params.radiative_transfer.τ_NIR_leaf,
            n_layers = params.radiative_transfer.n_layers,
            ϵ_canopy = params.radiative_transfer.ϵ_canopy,
        )
    )
    # Set up conductance
    conductance_args = (;
        parameters = MedlynConductanceParameters(
            FT;
            g1 = params.conductance.g1,
            Drel = params.conductance.Drel,
            g0 = params.conductance.g0,
        )
    )
    # Set up photosynthesis
    photosynthesis_args = (;
        parameters = FarquharParameters(
            FT,
            Canopy.C3();
            oi = params.photosynthesis.oi,
            ϕ = params.photosynthesis.ϕ,
            θj = params.photosynthesis.θj,
            f = params.photosynthesis.f,
            sc = params.photosynthesis.sc,
            pc = params.photosynthesis.pc,
            Vcmax25 = params.photosynthesis.Vcmax25,
            Γstar25 = params.photosynthesis.Γstar25,
            Kc25 = params.photosynthesis.Kc25,
            Ko25 = params.photosynthesis.Ko25,
            To = params.photosynthesis.To,
            ΔHkc = params.photosynthesis.ΔHkc,
            ΔHko = params.photosynthesis.ΔHko,
            ΔHVcmax = params.photosynthesis.ΔHVcmax,
            ΔHΓstar = params.photosynthesis.ΔHΓstar,
            ΔHJmax = params.photosynthesis.ΔHJmax,
            ΔHRd = params.photosynthesis.ΔHRd,
        )
    )
    # Set up plant hydraulics
    ai_parameterization = PrescribedSiteAreaIndex{FT}(
        drivers.LAIfunction,
        params.plant_hydraulics.SAI,
        drivers.RAI,
    )

    function root_distribution(
        z::T;
        rooting_depth = params.plant_hydraulics.rooting_depth,
    ) where {T}
        return T(1.0 / rooting_depth) * exp(z / T(rooting_depth)) # 1/m
    end

    plant_hydraulics_ps = PlantHydraulics.PlantHydraulicsParameters(;
        ai_parameterization = ai_parameterization,
        ν = drivers.plant_ν,
        S_s = params.plant_hydraulics.S_s,
        root_distribution = root_distribution,
        conductivity_model = params.plant_hydraulics.conductivity_model,
        retention_model = params.plant_hydraulics.retention_model,
    )
    plant_hydraulics_args = (parameters = plant_hydraulics_ps, h_stem, h_leaf)

    energy_args = (
        parameters = Canopy.BigLeafEnergyParameters{FT}(
            params.canopy_energy_balance.ac_canopy,
        ),
    )

    # Canopy component args
    canopy_component_args = (;
        autotrophic_respiration = autotrophic_respiration_args,
        radiative_transfer = radiative_transfer_args,
        photosynthesis = photosynthesis_args,
        conductance = conductance_args,
        hydraulics = plant_hydraulics_args,
        energy = energy_args,
    )

    z0_m = FT(0.13) * domain.h_canopy
    z0_b = FT(0.1) * z0_m

    # Other info needed
    shared_params = SharedCanopyParameters{FT, typeof(earth_param_set)}(
        z0_m,
        z0_b,
        earth_param_set,
    )

    canopy_model_args =
        (; parameters = shared_params, domain = domain.canopy_domain)

    # Integrated plant hydraulics and soil model
    land_input = (atmos = drivers.atmos, radiation = drivers.radiation)
    land = SoilCanopyModel{FT}(;
        soilco2_type = soilco2_type,
        soilco2_args = soilco2_args,
        land_args = land_input,
        soil_model_type = soil_model_type,
        soil_args = soil_args,
        canopy_component_types = canopy_component_types,
        canopy_component_args = canopy_component_args,
        canopy_model_args = canopy_model_args,
    )
    Y, p, cds = initialize(land)
    exp_tendency! = make_exp_tendency(land)

    #Initial conditions
    Y.soil.ϑ_l =
        drivers.drivers.SWC.status != absent ?
        drivers.drivers.SWC.values[1 + Int(round(setup.t0 / drivers.DATA_DT))] :
        params.soil.soil_ν / 2 # Get soil water content at t0
    # Both data and simulation are reference to 2005-01-01-00 (LOCAL)
    # or 2005-01-01-06 (UTC)
    Y.soil.θ_i = FT(0.0)
    T_0 =
        drivers.drivers.TS.status != absent ?
        drivers.drivers.TS.values[1 + Int(round(setup.t0 / drivers.DATA_DT))] :
        drivers.drivers.TA.values[1 + Int(round(setup.t0 / drivers.DATA_DT))] +
        40# Get soil temperature at t0
    ρc_s =
        volumetric_heat_capacity.(
            Y.soil.ϑ_l,
            Y.soil.θ_i,
            Ref(land.soil.parameters),
        )
    Y.soil.ρe_int =
        volumetric_internal_energy.(
            Y.soil.θ_i,
            ρc_s,
            T_0,
            Ref(land.soil.parameters),
        )
    Y.soilco2.C .= FT(0.000412) # set to atmospheric co2, mol co2 per mol air
    ψ_stem_0 = FT(-1e5 / 9800) # pressure in the leaf divided by rho_liquid*gravitational acceleration [m] 
    ψ_leaf_0 = FT(-2e5 / 9800)
    ψ_comps = setup.h_stem > 0 ? [ψ_stem_0, ψ_leaf_0] : ψ_leaf_0

    S_l_ini =
        inverse_water_retention_curve.(
            params.plant_hydraulics.retention_model,
            ψ_comps,
            drivers.plant_ν,
            params.plant_hydraulics.S_s,
        )

    for i in 1:(setup.h_stem > 0 + 1)
        Y.canopy.hydraulics.ϑ_l.:($i) .=
            augmented_liquid_fraction.(drivers.plant_ν, S_l_ini[i])
    end

    Y.canopy.energy.T =
        drivers.drivers.TA.values[1 + Int(round(setup.t0 / drivers.DATA_DT))] # Get atmos temperature at t0

    set_initial_cache! = make_set_initial_cache(land)
    set_initial_cache!(p, Y, setup.t0)

    # Simulation
    sv = (;
        t = Array{Float64}(undef, length(timestepper.saveat)),
        saveval = Array{NamedTuple}(undef, length(timestepper.saveat)),
    )
    saving_cb = ClimaLand.NonInterpSavingCallback(sv, timestepper.saveat)
    ## How often we want to update the drivers. Note that this uses the defined `t0` and `tf`
    ## defined in the simulatons file
    updateat = Array((setup.t0):(drivers.DATA_DT):(timestepper.tf))
    updatefunc = ClimaLand.make_update_drivers(drivers.atmos, drivers.radiation)
    driver_cb = ClimaLand.DriverUpdateCallback(updateat, updatefunc)
    cb = SciMLBase.CallbackSet(driver_cb, saving_cb)

    prob = SciMLBase.ODEProblem(
        CTS.ClimaODEFunction((T_exp!) = exp_tendency!),
        Y,
        (setup.t0, timestepper.tf),
        p,
    )
    sol = SciMLBase.solve(
        prob,
        timestepper.ode_algo;
        dt = setup.dt,
        callback = cb,
        adaptive = false,
        saveat = timestepper.saveat,
    )

    if isfile(
        joinpath(
            climaland_dir,
            "lib/ClimaLandSimulations/src/utilities/$site_ID/Artifacts.toml",
        ),
    )
        rm(
            joinpath(
                climaland_dir,
                "lib/ClimaLandSimulations/src/utilities/$site_ID/Artifacts.toml",
            ),
        )
    else
        nothing
    end

    return sv, sol, Y, p
end



