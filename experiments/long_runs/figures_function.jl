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
    short_names;
    plot! = viz.heatmap2D_on_globe!,
)
    simdir = ClimaAnalysis.SimDir(outdir)
    mktempdir(root_path) do tmpdir
        for short_name in short_names
            var = get(simdir; short_name)
            N = length(ClimaAnalysis.times(var))
            if N > 1 # multiple times
                var_times = [
                    ClimaAnalysis.times(var)[1],
                    ClimaAnalysis.times(var)[div(N, 2, RoundNearest)],
                    ClimaAnalysis.times(var)[N],
                ]
            else # only one time
                var_times = ClimaAnalysis.times(var)[1]
            end
            kwarg_z = ClimaAnalysis.has_altitude(var) ? Dict(:z => 1) : Dict() # if has altitude, take first layer
            var_sliced = ClimaAnalysis.slice(var; kwarg_z...)
            # var_global_average below is a vector of vector, one for each year of simulation, containing monthly global average of var.
            # i represent a year, from 1 to last year
            # for more details on the ClimaAnalysis functions, see ClimaAnalysis docs.
            var_global_average = [
                ClimaAnalysis.average_lon(
                    ClimaAnalysis.weighted_average_lat(
                        ClimaAnalysis.apply_oceanmask(
                            ClimaAnalysis.window(
                                var_sliced,
                                "time",
                                left = (i - 1) * 366 * 86400 + 30 * 86400, # 1 year left of year i, in seconds.
                                right = i * 366 * 86400, # 1 year right of year i, in seconds
                            ),
                        ),
                    ),
                ).data for i in range(
                    1,
                    round(
                        last(ClimaAnalysis.times(var_sliced) / (365 * 86400)),
                    ), # n years
                )
            ]
            fig_seasonal_cycle = CairoMakie.Figure(size = (600, 400))
            ax = Axis(
                fig_seasonal_cycle[1, 1],
                xlabel = "Month of the year",
                ylabel = "$short_name [$(ClimaAnalysis.units(var))]",
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
            [
                CairoMakie.lines!(
                    ax,
                    var_global_average[i],
                    color = RGBf(0.5, 0.5, 0.5),
                    linewidth = 1,
                    linestyle = (i == 1 ? :dash : :solid), # dashed line for the 1st year
                ) for i in 1:length(var_global_average)
            ]
            # The three next lines here are computing the average for each month of var, January to December. It accounts for cases if the last simulated year is incomplete. In that case, the last vector of var_global_average would be shorter than 12, so in order to compute the average, it needs to be padded with NaNs until it reaches the length of 12.
            # We also want the average for each month to be computed even if there is NaNs in the last vector, for example January may have 3 data points, but December only two, but we still want the average of all months.
            max_len = maximum(length.(var_global_average)) # 12 months
            padded_var = [
                vcat(v, fill(NaN, max_len - length(v))) for
                v in var_global_average
            ] # fill with NaN up to 12, if a vector is shorter
            monthly_averages_var = [
                mean(filter(!isnan, collect([v[i] for v in padded_var])))
                for i in 1:max_len
            ] # compute average of each month, ignoring potential NaNs
            CairoMakie.lines!(
                ax,
                monthly_averages_var,
                color = :black,
                linewidth = 3,
            )
            CairoMakie.xlims!(ax, 1, 12)
            CairoMakie.save(
                joinpath(tmpdir, "$(short_name)_global_monthly.pdf"),
                fig_seasonal_cycle,
            )
            for t in var_times
                fig = CairoMakie.Figure(size = (600, 400))
                kwargs =
                    ClimaAnalysis.has_altitude(var) ? Dict(:z => 1) : Dict()
                plot!(
                    fig,
                    ClimaAnalysis.slice(var, time = t; kwargs...),
                    mask = viz.oceanmask(),
                    more_kwargs = Dict(
                        :mask => ClimaAnalysis.Utils.kwargs(color = :white),
                        :plot => ClimaAnalysis.Utils.kwargs(rasterize = true),
                    ),
                )
                CairoMakie.save(joinpath(tmpdir, "$(short_name)_$t.pdf"), fig)
            end
        end
        figures = readdir(tmpdir, join = true)
        pdfunite() do unite
            run(Cmd([unite, figures..., joinpath(root_path, "figures.pdf")]))
        end
    end
    return nothing
end
