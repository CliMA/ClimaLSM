export niwotridge_default_args

"""
	function niwotridge_default_args(;
		dz_bottom = FT(1.25),
		dz_top = FT(0.05),
		h_leaf = FT(6.5), # m,
		h_stem = FT(7.5), # m,
		t0 = Float64(120 * 3600 * 24), # start mid year to avoid snow,
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

Named Tuple containing default simulation parameter for the Niwot Ridge site. 
"""
function niwotridge_default_args(;
    dz_bottom = FT(1.25),
    dz_top = FT(0.05),
    h_leaf = FT(6.5), # m,
    h_stem = FT(7.5), # m,
    t0 = Float64(120 * 3600 * 24), # start mid year to avoid snow,
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
