export make_setup

"""
    make_setup()

setup site specific domain and time stepping:
- dz_bottom, the bottom of the soil domain (m)
- dz_top, the top of the soil domain (m)
- h_stem, the height of each stem (m)
- h_leaf, the height of each leaf (m)
- t0, the start time of the simulation (s)
- dt, the time step (s)
- n, the number of time step between saving outputs
"""
function make_setup(; dz_bottom, dz_top, h_stem, h_leaf, t0, dt, n)
    return (
        dz_bottom = dz_bottom,
        dz_top = dz_top,
        h_stem = h_stem,
        h_leaf = h_leaf,
        t0 = t0,
        dt = dt,
        n = n,
    )
end

function make_setup(site_ID)
    default_args = Dict(
        "US-MOz" => ozark_default_args(),
        "US-Ha1" => harvard_default_args(),
        "US-NR1" => niwotridge_default_args(),
        "US-Var" => vairaranch_default_args(),
    )

    if haskey(default_args, site_ID)
        make_setup(; default_args[site_ID]...)
    else
        println("This site_ID isn't supported.")
        return
    end
end
