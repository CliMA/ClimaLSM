# https://github.com/jheinen/GR.jl/issues/278#issuecomment-587090846
ENV["GKSwstype"] = "nul"
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Distributed
@everywhere using Documenter
@everywhere using Literate
using ClimaLand
include("pages_helper.jl")
include("list_tutorials.jl")

@everywhere const clima_dir = dirname(dirname(pathof(ClimaLand)));
@everywhere source_dir = joinpath(@__DIR__, "src")
@everywhere GENERATED_DIR = joinpath(source_dir, "generated") # generated files directory
rm(GENERATED_DIR, force = true, recursive = true)
mkpath(GENERATED_DIR)
@everywhere function generate_tutorial(tutorials_dir, tutorial)
    rpath = relpath(dirname(tutorial), tutorials_dir)
    rpath = rpath == "." ? "" : rpath
    gen_dir = joinpath(GENERATED_DIR, rpath)
    mkpath(gen_dir)

    cd(gen_dir) do
        # change the Edit on GitHub link:
        path = relpath(clima_dir, pwd())
        content = """
                # ```@meta
                    # EditURL = "https://github.com/CliMA/ClimaLand.jl/$(path)"
                    # ```
                """
        mdpre(str) = content * str
        input = abspath(tutorial)
        Literate.markdown(
            input;
            execute = true,
            documenter = false,
            preprocess = mdpre,
        )
    end
end
tutorials_jl = flatten_to_array_of_strings(get_second(tutorials))
println("Building literate tutorials...")
tutorials_dir = joinpath(@__DIR__, "tutorials")
tutorials_jl = map(x -> joinpath(tutorials_dir, x), tutorials_jl)
pmap(t -> generate_tutorial(tutorials_dir, t), tutorials_jl)

# update list of rendered markdown tutorial output for mkdocs
ext_jl2md(x) = joinpath(basename(GENERATED_DIR), replace(x, ".jl" => ".md"))
tutorials = transform_second(x -> ext_jl2md(x), tutorials)
include("list_of_apis.jl")
include("list_standalone_models.jl")
include("list_diagnostics.jl")
pages = Any[
    "Home" => "index.md",
    "Getting Started" => "getting_started.md",
    "Repository structure" => "folderstructure.md",
    "Tutorials" => tutorials,
    "Standalone models" => standalone_models,
    "Diagnostics" => diagnostics,
    "Calibration" => "calibration.md",
    "Leaderboard" => "leaderboard/leaderboard.md",
    "Restarts" => "restarts.md",
    "Misc. utilities" => "shared_utilities.md",
    "APIs" => apis,
    "Contribution guide" => "Contributing.md",
]

mathengine = MathJax(
    Dict(
        :TeX => Dict(
            :equationNumbers => Dict(:autoNumber => "AMS"),
            :Macros => Dict(),
        ),
    ),
)

format = Documenter.HTML(
    prettyurls = !isempty(get(ENV, "CI", "")),
    collapselevel = 1,
    mathengine = mathengine,
    ansicolor = true,
)

makedocs(
    sitename = "ClimaLand.jl",
    authors = "Clima Land Model Team",
    format = format,
    pages = pages,
    checkdocs = :exports,
    doctest = true,
    warnonly = true,
    clean = true,
    modules = [ClimaLand],
)

deploydocs(
    repo = "github.com/CliMA/ClimaLand.jl.git",
    target = "build",
    push_preview = true,
    devbranch = "main",
    forcepush = true,
)
