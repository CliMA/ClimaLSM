include("leaderboard/leaderboard.jl")
"""
    make_figures(
        root_path,
        outdir,
        short_names;
        plot! = viz.heatmap2D_on_globe!,
    )

Generates one .pdf file called "figures.pdf" in the provided root_path,
this .pdf contains global maps for the provided short_names variables
contained in the provided outdir folder (see ClimaAnalysis documentation),
at the beginning, middle and end of their time period.
It also contains global seasonal cycle plots, as the monthly average of variables.
"""
function make_figures(
    root_path,
    outdir,
    short_names,
    units_labels,
    sim_var_units_labels,
    title_stubs;
    plot! = viz.heatmap2D_on_globe!,
)
    simdir = ClimaAnalysis.SimDir(outdir)

    # Set up for comparison to data (copied from leaderboard.jl)
    # use sim_var and obs_var together for the seasonal plot because they already have the same units :)
    sim_var_dict = get_sim_var_dict(outdir)
    obs_var_dict = get_obs_var_dict()
    # Set up dict for storing simulation and observational data after processing
    sim_obs_comparsion_dict = Dict()
    mask_dict = get_mask_dict()

    for short_name in [short_names[1]]
        var = get(simdir; short_name)
        title_stub = title_stubs[short_name]
        units_label = units_labels[short_name]
        sim_var_units_label = sim_var_units_labels[short_name]
        N = length(ClimaAnalysis.times(var))
        var_times = [ClimaAnalysis.times(var)[1]]#,
        #     ClimaAnalysis.times(var)[div(N, 2, RoundNearest)],
        #     ClimaAnalysis.times(var)[N],
        # ]

        ## SEASONAL CYCLE
        # data_sources.jl has observational data for "gpp", "lwu", and "et" only - maybe separate short_names loop for this
        # Simulation data
        sim_var = sim_var_dict[short_name]()
        kwarg_z = ClimaAnalysis.has_altitude(sim_var) ? Dict(:z => 1) : Dict() # if has altitude, take first layer
        sim_var_sliced = ClimaAnalysis.slice(sim_var; kwarg_z...)
        # var_global_average below is a vector of vector, one for each year of simulation, containing monthly global average of var.
        # i represent a year, from 1 to last year
        # for more details on the ClimaAnalysis functions, see ClimaAnalysis docs.

        # ~only compute seasonal cycle for last year so we skip spinup~
        # compute seasonal cycle for second to last year so we skip spinup AND have data for dec after off-by-one correction (shift_to_start_of_previous_month)
        i =
            Int(
                round(
                    last(ClimaAnalysis.times(sim_var_sliced) / (365 * 86400)),
                ),
            ) - 1# n years
        sim_var_global_average =
            ClimaAnalysis.average_lon(
                ClimaAnalysis.weighted_average_lat(
                    ClimaAnalysis.apply_oceanmask(
                        ClimaAnalysis.window(
                            sim_var_sliced,
                            "time",
                            left = (i - 1) * 366 * 86400 + 30 * 86400, # 1 year left of year i, in seconds.
                            right = i * 366 * 86400, # 1 year right of year i, in seconds
                        ),
                    ),
                ),
            ).data #for i in range(
        #     1,
        #     round(
        #         last(ClimaAnalysis.times(var_sliced) / (365 * 86400)),
        #     ), # n years
        # )

        fig_seasonal_cycle = CairoMakie.Figure(size = (600, 400))
        ax = Axis(
            fig_seasonal_cycle[1, 1],
            # xlabel = "Month of the year",
            ylabel = "$sim_var_units_label",
            title = CairoMakie.rich(title_stub, fontsize = 18),
            xticks = (
                1:1:12,
                [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dev",
                ],
            ),
        )
        # [
        # plot model output
        # TODO apply shift_to_start_of_previous_month
        CairoMakie.lines!(
            ax,
            sim_var_global_average,
            color = :blue,#RGBf(0.5, 0.5, 0.5),
            linewidth = 3,
            # linestyle = (i == 1 ? :dash : :solid), # dashed line for the 1st year
        ) #for i in 1:length(var_global_average)
        # ]
        # The three next lines here are computing the average for each month of var, January to December. It accounts for cases if the last simulated year is incomplete. In that case, the last vector of var_global_average would be shorter than 12, so in order to compute the average, it needs to be padded with NaNs until it reaches the length of 12.
        # We also want the average for each month to be computed even if there is NaNs in the last vector, for example January may have 3 data points, but December only two, but we still want the average of all months.
        # max_len = maximum(length.(var_global_average)) # 12 months
        # padded_var = [
        #     vcat(v, fill(NaN, max_len - length(v))) for
        #     v in var_global_average
        # ] # fill with NaN up to 12, if a vector is shorter
        # monthly_averages_var = [
        #     mean(filter(!isnan, collect([v[i] for v in padded_var])))
        #     for i in 1:max_len
        # ] # compute average of each month, ignoring potential NaNs
        # CairoMakie.lines!(
        #     ax,
        #     monthly_averages_var,
        #     color = :black,
        #     linewidth = 3,
        # )
        # CairoMakie.xlims!(ax, 1, 12)



        # Add comparison to observational data (copied from leaderboard.jl)


        # Observational data
        obs_var = obs_var_dict[short_name](sim_var.attributes["start_date"])

        obs_var_sliced = ClimaAnalysis.slice(obs_var; kwarg_z...)
        # var_global_average below is a vector of vector, one for each year of simulation, containing monthly global average of var.
        # i represent a year, from 1 to last year
        # for more details on the ClimaAnalysis functions, see ClimaAnalysis docs.

        obs_var_global_average =
            ClimaAnalysis.average_lon(
                ClimaAnalysis.weighted_average_lat(
                    ClimaAnalysis.apply_oceanmask(
                        ClimaAnalysis.window(
                            obs_var_sliced,
                            "time",
                            left = (i - 1) * 366 * 86400 + 30 * 86400, # 1 year left of year i, in seconds.
                            right = i * 366 * 86400, # 1 year right of year i, in seconds
                        ),
                    ),
                ),
            ).data

        CairoMakie.scatter!(ax, obs_var_global_average, color = :orange)

        CairoMakie.save(
            joinpath(root_path, "$(short_name)_global_monthly.pdf"),
            fig_seasonal_cycle,
        )

        ## GLOBAL HEATMAP
        #     for t in var_times
        #         title = CairoMakie.rich(title_stub, fontsize = 18) # title of the figure
        #         fig = CairoMakie.Figure(size = (600, 400))
        #         kwargs = ClimaAnalysis.has_altitude(var) ? Dict(:z => 1) : Dict()
        #         viz.heatmap2D_on_globe!(
        #             fig,
        #             ClimaAnalysis.slice(var, time = t; kwargs...),
        #             units_label = units_labels[short_name],
        #             mask = viz.oceanmask(),
        #             more_kwargs = Dict(
        #                 :mask => ClimaAnalysis.Utils.kwargs(color = :white),
        #                 :plot => ClimaAnalysis.Utils.kwargs(rasterize = true),
        #                 :axis => ClimaAnalysis.Utils.kwargs(
        #                     title = title,
        #                     xticklabelsvisible = false, # don't show lat labels
        #                     yticklabelsvisible = false, # don't show lon labels
        #                     xgridvisible = false, # don't show lat grid
        #                     ygridvisible = false, # don't show lon grid
        #                 ),
        #                 # :cb => ClimaAnalysis.Utils.kwargs(
        #                 #     rightspinevisible = true,
        #                 # ),
        #             ),
        #         )
        #         CairoMakie.save(joinpath(root_path, "$(short_name)_$t.pdf"), fig)
        #     end
    end
    # figures = readdir(root_path, join = true)
    # pdfunite() do unite
    #     run(Cmd([unite, figures..., joinpath(root_path, "figures.pdf")]))
    # end
    return nothing
end
