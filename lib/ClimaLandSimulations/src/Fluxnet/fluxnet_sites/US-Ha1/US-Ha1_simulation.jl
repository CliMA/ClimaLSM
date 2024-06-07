export harvard_default_args

"""
	function harvard_default_args(;
		dz_bottom = FT(1.5),
		dz_top = FT(0.025),
		h_leaf = FT(12), # m
		h_stem = FT(14), # m
		t0 = Float64(120 * 3600 * 24),
		dt = Float64(40),
		n = 45,
		)
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

Named Tuple containing default simulation parameter for the Harvard site. 
"""
function harvard_default_args(;
    dz_bottom = FT(1.5),
    dz_top = FT(0.025),
    h_leaf = FT(12), # m
    h_stem = FT(14), # m
    t0 = Float64(120 * 3600 * 24),
    dt = Float64(40),
    n = 45,
)
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
