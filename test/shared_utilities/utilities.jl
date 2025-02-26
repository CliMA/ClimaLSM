using Test
import ClimaComms
ClimaComms.@import_required_backends
using ClimaCore: Spaces, Geometry, Fields
using ClimaLand
using ClimaLand: Domains, condition, SavingAffect, saving_initialize

## Callback tests
mutable struct Integrator{FT}
    t::Any
    p::Array{FT}
end

function upd_integrator(integrator::Integrator{FT}, dt) where {FT}
    integrator.t += dt
    integrator.p .+= FT(1)
end

FT = Float32
@testset "callback cond, affect! test, FT = $FT" begin
    t0 = Float64(0)
    dt = Float64(10)
    tf = Float64(100)
    t_range = collect(t0:dt:tf)

    # specify when to save in the callback (test different dts)
    saveats = (
        collect(t0:dt:tf),
        collect(t0:(2 * dt):tf),
        collect(t0:(tf - t0):tf),
        collect(t0:(0.5 * dt):tf),
    )
    for saveat in saveats
        # note: delete non-multiples of dt in saveat before calling callback
        deleteat!(saveat, findall(x -> (x - t0) % dt != 0, saveat))

        saved_values = (;
            t = Array{Float64}(undef, length(saveat)),
            saveval = Array{Array{FT}}(undef, length(saveat)),
        )

        # set up components of callback
        cond = condition(saveat)
        saveiter = 0
        affect! = SavingAffect(saved_values, saveat, saveiter)

        p_init = copy(t_range)
        integrator = Integrator{FT}(t0, p_init)

        # use this to manually save `integrator` at various `t`
        integ_saved = (;
            t = Array{Float64}(undef, length(saveat)),
            p = Array{Array{FT}}(undef, length(saveat)),
        )

        # simulate callback behavior
        saveiter = 0
        for t in t_range
            if cond(0, t, 0)
                affect!(integrator)

                # save the current state of `integrator`
                saveiter += 1
                integ_saved.t[saveiter] = integrator.t
                integ_saved.p[saveiter] = copy(integrator.p)
            end
            upd_integrator(integrator, dt)
        end

        @test affect!.saved_values.t == integ_saved.t
        @test affect!.saved_values.saveval == integ_saved.p
    end
end

@testset "callback saving_initialize test, FT = $FT" begin
    t0 = Float64(0)
    dt = Float64(10)
    tf = Float64(100)

    # we're only testing the save at t0 so one saveat range is sufficient
    saveat = collect(t0:dt:tf)
    saved_values = (;
        t = Array{Float64}(undef, length(saveat)),
        saveval = Array{Array{FT}}(undef, length(saveat)),
    )

    saveiter = 0
    affect! = SavingAffect(saved_values, saveat, saveiter)
    cb = (; affect! = affect!)

    p_init = collect(t0:dt:tf)
    integrator = Integrator{FT}(t0, p_init)

    # case 1: t not in saveat
    t1 = FT(-1)
    @test saving_initialize(cb, 0, t1, integrator) == false
    @test cb.affect!.saved_values.t != integrator.t
    @test cb.affect!.saved_values.saveval != integrator.p[1]

    # case 2: t in saveat
    t2 = t0
    saving_initialize(cb, 0, t2, integrator)
    @test cb.affect!.saved_values.t[1] == integrator.t[1]
    @test cb.affect!.saved_values.saveval[1] == integrator.p
end


## DSS 0D/1D tests
@testset "dss! - 0D/1D spaces, FT = $FT" begin
    # Test for Spaces.PointSpace and Spaces.FiniteDifferenceSpace
    domain1 = ClimaLand.Domains.Point(; z_sfc = FT(0))
    domain2 =
        ClimaLand.Domains.Column(; zlim = FT.((-1.0, 0.0)), nelements = (5))
    domains = (domain1, domain2, domain2)
    spaces =
        (domain1.space.surface, domain2.space.subsurface, domain2.space.surface)
    for (domain, space) in zip(domains, spaces)
        field = Fields.coordinate_field(space)
        subfield1 = similar(field)
        parent(subfield1) .= parent(copy(field)) .+ FT(1)

        Y = Fields.FieldVector(
            field = field,
            subfields = (subfield1 = subfield1),
        )
        Y_copy = copy(Y)
        p = (;)
        ClimaLand.dss!(Y, p, FT(0))
        @test p == (;)
        @test ClimaLand.add_dss_buffer_to_aux(p, domain) == p
        # On a 1D space, we expect dss! to do nothing
        @test Y.field == Y_copy.field
        @test Y.subfields == Y_copy.subfields
    end
end

@testset "dss! - 2D spaces, FT = $FT" begin
    # Test for Spaces.SpectralElementSpace2D
    domain1 = ClimaLand.Domains.Plane(;
        xlim = FT.((0.0, 1.0)),
        ylim = FT.((0.0, 1.0)),
        nelements = (2, 2),
        periodic = (true, true),
        npolynomial = 1,
    )
    domain2 = ClimaLand.Domains.SphericalSurface(;
        radius = FT(2),
        nelements = 10,
        npolynomial = 3,
    )

    domains = (domain1, domain2)
    for domain in domains
        space = domain.space.surface
        field = Fields.zeros(space)
        subfield1 = Fields.zeros(space)
        subfield2 = Fields.zeros(space)

        # make fields non-constant in order to check the
        # impact of the dss step
        ArrayType = ClimaComms.array_type(ClimaComms.device())
        local_size = size(parent(field))

        # We prepare a new Array or a CuArray (depending on the ClimaComms device) and swap
        # in-place the parent of our fields with the new data
        sin_i = ArrayType(
            reshape([sin(i) for i in eachindex(parent(field))], local_size),
        )
        cos_i = ArrayType(
            reshape([cos(i) for i in eachindex(parent(field))], local_size),
        )
        sin_cos_i = sin_i .* cos_i

        parent(field) .= sin_i
        parent(subfield1) .= cos_i
        parent(subfield2) .= sin_cos_i

        Y = Fields.FieldVector(
            field = field,
            subfields = (subfield1 = subfield1, subfield2 = subfield2),
        )
        Y_copy = copy(Y)
        p = (;)
        p = ClimaLand.add_dss_buffer_to_aux(p, domain)
        @test typeof(p.dss_buffer_2d) ==
              typeof(Spaces.create_dss_buffer(Fields.zeros(space)))
        ClimaLand.dss!(Y, p, FT(0))

        # On a 2D space, we expect dss! to change Y
        # but we changed dss! to do nothing, so replace != with  ==
        @test Y.field == Y_copy.field
        @test Y.subfields.subfield1 == Y_copy.subfields.subfield1
        @test Y.subfields.subfield2 == Y_copy.subfields.subfield2
    end
end

@testset "dss! - 3D spaces, FT = $FT" begin
    domain1 = ClimaLand.Domains.HybridBox(;
        xlim = FT.((0.0, 1.0)),
        ylim = FT.((0.0, 1.0)),
        zlim = FT.((0.0, 1.0)),
        nelements = (2, 2, 2),
        periodic = (true, true),
        npolynomial = 1,
    )
    domain2 = ClimaLand.Domains.SphericalShell(;
        radius = FT(2),
        depth = FT(1.0),
        nelements = (10, 5),
        npolynomial = 3,
    )

    domains = (domain1, domain2)
    for domain in domains
        space = domain.space.subsurface
        field = Fields.zeros(space)
        subfield1 = Fields.zeros(space)
        subfield2 = Fields.zeros(space)

        # make fields non-constant in order to check
        # the impact of the dss step
        ArrayType = ClimaComms.array_type(ClimaComms.device())
        local_size = size(parent(field))

        sin_i = ArrayType(
            reshape([sin(i) for i in eachindex(parent(field))], local_size),
        )
        cos_i = ArrayType(
            reshape([cos(i) for i in eachindex(parent(field))], local_size),
        )
        sin_cos_i = sin_i .* cos_i

        parent(field) .= sin_i
        parent(subfield1) .= cos_i
        parent(subfield2) .= sin_cos_i

        Y = Fields.FieldVector(
            field = field,
            subfields = (subfield1 = subfield1, subfield2 = subfield2),
        )
        Y_copy = copy(Y)
        p = (;)
        p = ClimaLand.add_dss_buffer_to_aux(p, domain)
        @test typeof(p.dss_buffer_3d) ==
              typeof(Spaces.create_dss_buffer(Fields.zeros(space)))
        @test typeof(p.dss_buffer_2d) == typeof(
            Spaces.create_dss_buffer(Fields.zeros(domain.space.surface)),
        )
        ClimaLand.dss!(Y, p, FT(0))

        # On a 3D space, we expect dss! to change Y
        # but we changed dss! to do nothing, so replace != with  ==
        @test Y.field == Y_copy.field
        @test Y.subfields.subfield1 == Y_copy.subfields.subfield1
        @test Y.subfields.subfield2 == Y_copy.subfields.subfield2
    end
end

@testset "count_nans_state, FT = $FT" begin
    # Test on a 3D spherical domain
    domain = ClimaLand.Domains.SphericalShell(;
        radius = FT(2),
        depth = FT(1.0),
        nelements = (10, 5),
        npolynomial = 3,
    )

    # Construct some fields
    space = domain.space.subsurface
    var1 = Fields.zeros(space)
    var2 = Fields.zeros(space)
    var3 = Fields.zeros(space)
    fieldvec = Fields.FieldVector(var2 = var2, var3 = var3)

    # Construct a FieldVector containing the fields and a nested FieldVector
    Y = Fields.FieldVector(var1 = var1, fieldvec = fieldvec)

    # Count and log the number of NaNs in the state (test verbose and non-verbose cases)
    @test_logs (:info, "Checking NaNs in var1") (:info, "No NaNs found") (
        :info,
        "Checking NaNs in fieldvec",
    ) (:info, "Checking NaNs in var2") (:info, "No NaNs found") (
        :info,
        "Checking NaNs in var3",
    ) (:info, "No NaNs found") ClimaLand.count_nans_state(Y, verbose = true)

    @test_logs (:info, "Checking NaNs in var1") (
        :info,
        "Checking NaNs in fieldvec",
    ) (:info, "Checking NaNs in var2") (:info, "Checking NaNs in var3") ClimaLand.count_nans_state(
        Y,
    )

    # Add some NaNs to the fields.
    # Note that we only check if NaNs are present in the surface field, since
    # any NaN in the column will propagate to the surface within several timesteps.
    # Note: this code uses `parent` and scalar indexing,
    # which shouldn't be replicated outside of tests
    ClimaComms.allowscalar(ClimaComms.device()) do
        parent(var1)[:, 1, 1, 1, 1] .= NaN
        parent(var2)[:, 1, 1, 1, 1] .= NaN
        parent(var2)[:, 2, 1, 1, 1] .= NaN
    end

    # Count and log the number of NaNs in the state (test verbose and non-verbose cases)
    @test_logs (:info, "Checking NaNs in var1") (:warn, "1 NaNs found") (
        :info,
        "Checking NaNs in fieldvec",
    ) (:info, "Checking NaNs in var2") (:warn, "2 NaNs found") (
        :info,
        "Checking NaNs in var3",
    ) (:info, "No NaNs found") ClimaLand.count_nans_state(Y, verbose = true)

    @test_logs (:info, "Checking NaNs in var1") (:warn, "1 NaNs found") (
        :info,
        "Checking NaNs in fieldvec",
    ) (:info, "Checking NaNs in var2") (:warn, "2 NaNs found") (
        :info,
        "Checking NaNs in var3",
    ) ClimaLand.count_nans_state(Y)

    # Test with a mask
    sfc_space = domain.space.surface

    mask_zeros = Fields.zeros(sfc_space)
    @test_logs (:info, "Checking NaNs in var1") (:info, "No NaNs found") (
        :info,
        "Checking NaNs in fieldvec",
    ) (:info, "Checking NaNs in var2") (:info, "No NaNs found") (
        :info,
        "Checking NaNs in var3",
    ) (:info, "No NaNs found") ClimaLand.count_nans_state(
        Y,
        mask = mask_zeros,
        verbose = true,
    )
    @test_logs (:info, "Checking NaNs in var1") (
        :info,
        "Checking NaNs in fieldvec",
    ) (:info, "Checking NaNs in var2") (:info, "Checking NaNs in var3") ClimaLand.count_nans_state(
        Y,
        mask = mask_zeros,
    )

    mask_ones = Fields.ones(sfc_space)
    @test_logs (:info, "Checking NaNs in var1") (:warn, "1 NaNs found") (
        :info,
        "Checking NaNs in fieldvec",
    ) (:info, "Checking NaNs in var2") (:warn, "2 NaNs found") (
        :info,
        "Checking NaNs in var3",
    ) (:info, "No NaNs found") ClimaLand.count_nans_state(
        Y,
        mask = mask_ones,
        verbose = true,
    )

    @test_logs (:info, "Checking NaNs in var1") (:warn, "1 NaNs found") (
        :info,
        "Checking NaNs in fieldvec",
    ) (:info, "Checking NaNs in var2") (:warn, "2 NaNs found") (
        :info,
        "Checking NaNs in var3",
    ) ClimaLand.count_nans_state(Y, mask = mask_ones)
end
