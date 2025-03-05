import ClimaLand as CL
using PrettyTables

# Print all available diagnostics to an ASCII table

CL.Diagnostics.define_diagnostics!(nothing)
short_names = []
long_names = []
units = []
comments = []
standard_names = []
for d in values(CL.Diagnostics.ALL_DIAGNOSTICS)
    push!(short_names, d.short_name)
    push!(long_names, d.long_name)
    push!(units, d.units)
    push!(comments, d.comments)
    push!(standard_names, d.standard_name)
end
i = sortperm(short_names) # indices of short_names sorted alphabetically
data = hcat(
    short_names[i],
    long_names[i],
    units[i],
    comments[i],
    standard_names[i],
)
pretty_table(
    data;
    autowrap = true,
    linebreaks = true,
    columns_width = [10, 15, 8, 32, 15],  # Width = 80
    body_hlines = collect(1:size(data)[1]),
    header = ["Short name", "Long name", "Units", "Comments", "Standard name"],
    alignment = :l,
)
