import SciMLBase
import ClimaTimeSteppers as CTS
using ClimaCore
using Plots
using Statistics
using Dates

using ClimaLand
using ClimaLand.Domains: Column
using ClimaLand.Soil
using ClimaLand.Soil.Biogeochemistry
using ClimaLand.Canopy
using ClimaLand.Canopy.PlantHydraulics
import ClimaLand
import ClimaLand.Parameters as LP
import ClimaParams

global climaland_dir = pkgdir(ClimaLand)
global site_ID = "US-MOz"

for float_type in (Float32, Float64)
    # Make these global so we can use them in other ozark files
    global FT = float_type
    global earth_param_set = LP.LandParameters(FT)

    # Create model, set initial conditions, and setup most simulation parameters
    include(
        joinpath(
            climaland_dir,
            "experiments/integrated/performance/conservation/ozark_conservation_setup.jl",
        ),
    )

    # Use smaller `tf` for Float32 simulation
    tf = (FT == Float64) ? t0 + 3600 * 24 * 10 : t0 + 2 * dt
    saveat = Array(t0:dt:tf)
    sv = (;
        t = Array{Float64}(undef, length(saveat)),
        saveval = Array{NamedTuple}(undef, length(saveat)),
    )
    saving_cb = ClimaLand.NonInterpSavingCallback(sv, saveat)

    updateat = deepcopy(saveat)
    drivers = ClimaLand.get_drivers(land)
    updatefunc = ClimaLand.make_update_drivers(drivers)
    driver_cb = ClimaLand.DriverUpdateCallback(updateat, updatefunc)
    prob = SciMLBase.ODEProblem(
        CTS.ClimaODEFunction(
            T_exp! = exp_tendency!,
            T_imp! = SciMLBase.ODEFunction(imp_tendency!; jac_kwargs...),
            dss! = ClimaLand.dss!,
        ),
        Y,
        (t0, tf),
        p,
    )
    cb = SciMLBase.CallbackSet(driver_cb, saving_cb)
    sol = SciMLBase.solve(
        prob,
        ode_algo;
        dt = dt,
        callback = cb,
        adaptive = false,
        saveat = saveat,
    )

    # Check that simulation still has correct float type
    @assert eltype(sol.u[end].soil) == FT
    @assert eltype(sol.u[end].soilco2) == FT
    @assert eltype(sol.u[end].canopy) == FT

    # Plotting for Float64 simulation
    if FT == Float64
        # Check that the driver variables stored in `p` are
        # updated correctly - just check one from radiation,
        # and one from atmos.
        cache_θs = [parent(sv.saveval[k].drivers.θs)[1] for k in 1:length(sv.t)]
        cache_Tair =
            [parent(sv.saveval[k].drivers.T)[1] for k in 1:length(sv.t)]
        @assert mean(
            abs.(radiation.θs.(sv.t, radiation.start_date) .- cache_θs),
        ) < eps(FT)
        T_mutable = Vector{FT}(undef, 1)
        atmos_T = map(sv.t) do time
            ClimaLand.evaluate!(T_mutable, atmos.T, time)
            return T_mutable[]
        end |> collect

        @assert mean(abs.(atmos_T .- cache_Tair)) < eps(FT)

        daily = sol.t[2:end] ./ 3600 ./ 24
        savedir = joinpath(
            climaland_dir,
            "experiments/integrated/performance/conservation",
        )

        # The aux state in `sv` is an index off from the solution..
        # Aux state at index 2 corresponds to the solution at index 1
        # This is a bug that we will need to fix at some point
        # Internally during the integration, however, aux and state
        # are updated consistently.

        ##  Soil water balance ##

        # Evaporation
        E = [
            parent(
                sv.saveval[k].soil.turbulent_fluxes.vapor_flux_liq .+
                sv.saveval[k].soil.turbulent_fluxes.vapor_flux_ice,
            )[1] for k in 2:length(sol.t)
        ]
        # Root sink term: a positive root extraction is a sink term for soil; add minus sign
        root_sink =
            [sum(-1 .* sv.saveval[k].root_extraction) for k in 2:length(sol.t)]
        # Free Drainage BC, F_bot = -K_bot
        soil_bottom_flux =
            [-parent(sv.saveval[k].soil.K)[1] for k in 2:length(sol.t)]
        # Precip is not stored in the aux state, evaluated using sol.t
        p_mutable = Vector{FT}(undef, 1)

        precip =
            map(sol.t[k] for k in 1:(length(sol.t) - 1)) do time
                ClimaLand.evaluate!(p_mutable, atmos.liquid_precip, time)
                return p_mutable[]
            end |> collect

        # Water balance equation

        # d[∫(ϑ_l+ θ_i ρ_i/ρ_l)dz] = [-(F_sfc - F_bot) + ∫Sdz]dt = -ΔF dt + ∫Sdz dt
        # N.B. in ClimaCore, sum(field) -> integral
        rhs_soil = -(precip .+ E .- soil_bottom_flux) .+ root_sink

        ρ_liq = LP.ρ_cloud_liq(earth_param_set)
        ρ_ice = LP.ρ_cloud_ice(earth_param_set)
        net_soil_water_storage = [
            sum(sol.u[k].soil.ϑ_l .+ sol.u[k].soil.θ_i .* ρ_ice ./ ρ_liq)[1]
            for k in 1:length(sol.t)
        ]
        lhs_soil = net_soil_water_storage[2:end] .- net_soil_water_storage[1]
        soil_mass_change_actual = lhs_soil
        soil_mass_change_exp = cumsum(rhs_soil) .* dt

        ## Canopy water balance ##

        # Bottom flux from roots to stem
        root_flux =
            [sum(sv.saveval[k].root_extraction) for k in 2:length(sol.t)]

        # Stem leaf flux
        stem_leaf_flux = [
            parent(sv.saveval[k].canopy.hydraulics.fa)[1] for
            k in 2:length(sol.t)
        ]
        # LAI (t)
        leaf_area_index = [
            parent(
                getproperty(sv.saveval[k].canopy.hydraulics.area_index, :leaf),
            )[1] for k in 2:length(sol.t)
        ]
        # Top boundary flux (transpiration)
        T = [
            parent(sv.saveval[k].canopy.conductance.transpiration)[1] for
            k in 2:length(sol.t)
        ]

        # Water balance equation
        # d[θ_leaf h_leaf+ θ_stem h_stem] = -[F_sl - Root Flux]/SAI - [T - F_sl]/LAI
        rhs_canopy = @. -T / leaf_area_index +
           root_flux / SAI +
           stem_leaf_flux * (1 / leaf_area_index - 1 / SAI)

        net_plant_water_storage = [
            sum(parent(sol.u[k].canopy.hydraulics.ϑ_l) .* [h_stem, h_leaf])
            for k in 1:length(sol.t)
        ]
        lhs_canopy =
            net_plant_water_storage[2:end] .- net_plant_water_storage[1]

        canopy_mass_change_actual = lhs_canopy
        canopy_mass_change_exp = cumsum(rhs_canopy) .* dt

        plt2 = Plots.plot(
            size = (1500, 400),
            ylabel = "Fractional Error",
            yaxis = :log,
            margin = 10Plots.mm,
            xlabel = "Day",
        )
        Plots.plot!(
            plt2,
            daily,
            eps(FT) .+
            abs.(
                (soil_mass_change_actual - soil_mass_change_exp) ./
                soil_mass_change_exp
            ),
            label = "Soil Water Balance",
        )
        Plots.plot!(
            plt2,
            daily,
            eps(FT) .+
            abs.(
                (canopy_mass_change_actual - canopy_mass_change_exp) ./
                canopy_mass_change_exp
            ),
            label = "Canopy Water Balance",
        )
        Plots.savefig(joinpath(savedir, "water_conservation.png"))



        ##  Soil energy balance ##
        # Energy of liquid water infiltrating soil is ignored in our model.

        # Turbulent fluxes
        LHF = [
            parent(sv.saveval[k].soil.turbulent_fluxes.lhf)[1] for
            k in 2:length(sol.t)
        ]
        SHF = [
            parent(sv.saveval[k].soil.turbulent_fluxes.shf)[1] for
            k in 2:length(sol.t)
        ]
        # Radiation is computed in LW and SW components
        # with positive numbers indicating the soil gaining energy.
        soil_Rn = [parent(sv.saveval[k].soil.R_n)[1] for k in 2:length(sol.t)]
        # Root sink term: a positive root extraction is a sink term for soil; add minus sign
        root_sink_energy = [
            sum(-1 .* sv.saveval[k].root_energy_extraction) for
            k in 2:length(sol.t)
        ]
        # Bottom energy BC
        soil_bottom_flux = FT(0)

        # Energy balance equation

        # d[∫Idz] = [-(F_sfc - F_bot) + ∫Sdz]dt = -ΔF dt + ∫Sdz dt
        # N.B. in ClimaCore, sum(field) -> integral
        rhs_soil_energy =
            -(LHF .+ SHF .- soil_Rn .- soil_bottom_flux) .+ root_sink_energy

        net_soil_energy_storage =
            [sum(sol.u[k].soil.ρe_int)[1] for k in 1:length(sol.t)]
        lhs_soil_energy =
            net_soil_energy_storage[2:end] .- net_soil_energy_storage[1]
        soil_energy_change_actual = lhs_soil_energy
        soil_energy_change_exp = cumsum(rhs_soil_energy) .* dt

        plt2 = Plots.plot(
            size = (1500, 400),
            ylabel = "Fractional Error",
            yaxis = :log,
            margin = 10Plots.mm,
            xlabel = "Day",
        )
        Plots.plot!(
            plt2,
            daily,
            eps(FT) .+
            abs.(
                (soil_energy_change_actual - soil_energy_change_exp) ./
                soil_energy_change_exp
            ),
            label = "Soil Energy Balance",
        )

        Plots.savefig(joinpath(savedir, "energy_conservation.png"))
    end
end
