import ClimaComms
ClimaComms.@import_required_backends
using ClimaCore
using Test
using ClimaLand
using ClimaLand: Domains
using ClimaLand.Domains:
    Column,
    HybridBox,
    Plane,
    Point,
    SphericalShell,
    SphericalSurface,
    coordinates,
    obtain_surface_space,
    obtain_face_space,
    obtain_surface_domain,
    get_Δz,
    top_face_to_surface

FT = Float32
@testset "Clima Core Domains, FT = $FT" begin
    zmin = FT(-1.0)
    zmax = FT(0.0)
    xlim = FT.((0.0, 10.0))
    ylim = FT.((0.0, 1.0))
    zlim = FT.((zmin, zmax))
    nelements = (1, 1, 10)
    radius = FT(100)
    depth = FT(30)
    n_elements_sphere = (6, 20)
    npoly_sphere = 3
    # Spherical Shell
    shell = SphericalShell(;
        radius = radius,
        depth = depth,
        nelements = n_elements_sphere,
        npolynomial = npoly_sphere,
    )
    @test shell.fields.depth == depth
    @test shell.fields.z ==
          ClimaCore.Fields.coordinate_field(shell.space.subsurface).z
    face_space = obtain_face_space(shell.space.subsurface)
    z_face = ClimaCore.Fields.coordinate_field(face_space).z
    @test shell.fields.z_sfc == top_face_to_surface(z_face, shell.space.surface)
    Δz_top, Δz_bottom, Δz = get_Δz(shell.fields.z)
    @test shell.fields.Δz_top == Δz_top
    @test shell.fields.Δz_bottom == Δz_bottom
    @test shell.radius == radius
    @test shell.depth == depth
    @test shell.nelements == n_elements_sphere
    @test shell.npolynomial == npoly_sphere
    shell_coords = coordinates(shell).subsurface
    @test eltype(shell_coords) == ClimaCore.Geometry.LatLongZPoint{FT}
    @test typeof(shell_coords) <: ClimaCore.Fields.Field
    @test typeof(ClimaCore.Spaces.horizontal_space(shell.space.subsurface)) <:
          ClimaCore.Spaces.SpectralElementSpace2D
    @test typeof(shell.space.subsurface) <:
          ClimaCore.Spaces.CenterExtrudedFiniteDifferenceSpace
    @test typeof(shell.space.surface) <: ClimaCore.Spaces.SpectralElementSpace2D
    @test obtain_surface_space(shell.space.subsurface) == shell.space.surface
    @test obtain_surface_domain(shell) ==
          SphericalSurface{FT, typeof((; surface = shell.space.surface))}(
        radius,
        n_elements_sphere[1],
        npoly_sphere,
        (; surface = shell.space.surface),
    )
    shell_stretch = SphericalShell(;
        radius = radius,
        depth = FT(1.0),
        dz_tuple = FT.((0.3, 0.03)),
        nelements = (6, 10),
        npolynomial = 3,
    )
    shell_coords_stretch = coordinates(shell_stretch).subsurface
    dz =
        Array(parent(shell_coords_stretch.z))[:, 1, 4, 1, 216][2:end] .-
        Array(parent(shell_coords_stretch.z))[:, 1, 4, 1, 216][1:(end - 1)]
    @test abs(dz[1] - 0.3) < 1e-1
    @test abs(dz[end] - 0.03) < 1e-2

    # Spherical Surface
    shell_surface = SphericalSurface(;
        radius = radius,
        nelements = n_elements_sphere[1],
        npolynomial = npoly_sphere,
    )
    @test shell_surface.radius == radius
    @test shell_surface.nelements == n_elements_sphere[1]
    @test shell_surface.npolynomial == npoly_sphere
    shell_surface_coords = coordinates(shell_surface).surface
    @test eltype(shell_surface_coords) == ClimaCore.Geometry.LatLongPoint{FT}
    @test typeof(shell_surface_coords) <: ClimaCore.Fields.Field
    @test typeof(shell_surface.space.surface) <:
          ClimaCore.Spaces.SpectralElementSpace2D

    @test ClimaComms.context(shell_surface) == ClimaComms.context()
    @test ClimaComms.device(shell_surface) == ClimaComms.device()

    # HybridBox
    box = HybridBox(;
        xlim = xlim,
        ylim = ylim,
        zlim = zlim,
        nelements = nelements,
        npolynomial = 0,
    )
    @test box.fields.depth == zlim[2] - zlim[1]
    @test box.fields.z ==
          ClimaCore.Fields.coordinate_field(box.space.subsurface).z
    face_space = obtain_face_space(box.space.subsurface)
    z_face = ClimaCore.Fields.coordinate_field(face_space).z
    @test box.fields.z_sfc == top_face_to_surface(z_face, box.space.surface)
    Δz_top, Δz_bottom, Δz = get_Δz(box.fields.z)
    @test box.fields.Δz_top == Δz_top
    @test box.fields.Δz_bottom == Δz_bottom
    box_coords = coordinates(box).subsurface
    @test eltype(box_coords) == ClimaCore.Geometry.XYZPoint{FT}
    @test typeof(box_coords) <: ClimaCore.Fields.Field
    @test box.xlim == FT.(xlim)
    @test box.ylim == FT.(ylim)
    @test box.zlim == FT.(zlim)
    @test box.nelements == nelements
    @test box.npolynomial == 0
    @test box.periodic == (true, true)
    @test typeof(ClimaCore.Spaces.horizontal_space(box.space.subsurface)) <:
          ClimaCore.Spaces.SpectralElementSpace2D
    @test typeof(box.space.subsurface) <:
          ClimaCore.Spaces.CenterExtrudedFiniteDifferenceSpace
    @test typeof(box.space.surface) <: ClimaCore.Spaces.SpectralElementSpace2D
    @test obtain_surface_space(box.space.subsurface) == box.space.surface
    @test obtain_surface_domain(box) ==
          Plane{FT, typeof((; surface = box.space.surface))}(
        xlim,
        ylim,
        nothing,
        nelements[1:2],
        (true, true),
        0,
        (; surface = box.space.surface),
    )

    stretch_box = HybridBox(;
        xlim = xlim,
        ylim = ylim,
        zlim = zlim,
        dz_tuple = FT.((0.3, 0.03)),
        nelements = nelements,
        npolynomial = 0,
    )
    box_coords_stretch = coordinates(stretch_box).subsurface
    dz =
        Array(parent(box_coords_stretch.z))[:][2:end] .-
        Array(parent(box_coords_stretch.z))[:][1:(end - 1)]
    @test abs(dz[1] - 0.3) < 1e-1
    @test abs(dz[end] - 0.03) < 1e-2

    # Plane
    xy_plane = Plane(;
        xlim = xlim,
        ylim = ylim,
        nelements = nelements[1:2],
        periodic = (true, true),
        npolynomial = 0,
    )
    plane_coords = coordinates(xy_plane).surface
    @test eltype(plane_coords) == ClimaCore.Geometry.XYPoint{FT}
    @test typeof(plane_coords) <: ClimaCore.Fields.Field
    @test xy_plane.xlim == FT.(xlim)
    @test xy_plane.ylim == FT.(ylim)
    @test xy_plane.nelements == nelements[1:2]
    @test xy_plane.npolynomial == 0
    @test xy_plane.periodic == (true, true)
    @test typeof(xy_plane.space.surface) <:
          ClimaCore.Spaces.SpectralElementSpace2D

    # Plane latlong
    dxlim = (FT(50_000), FT(80_000))
    dylim = (FT(30_000), FT(40_000))
    longlat = (FT(-118.14452), FT(34.14778))
    radius_earth = FT(6.378e6)
    xlim_longlat = (
        longlat[2] - dylim[1] / FT(2π * radius_earth) * 360,
        longlat[2] + dylim[2] / FT(2π * radius_earth) * 360,
    )
    ylim_longlat = (
        longlat[1] - dxlim[1] / FT(2π * radius_earth) * 360,
        longlat[1] + dxlim[2] / FT(2π * radius_earth) * 360,
    )

    longlat_plane = Plane(;
        xlim = dxlim,
        ylim = dylim,
        longlat,
        nelements = nelements[1:2],
        npolynomial = 0,
    )
    plane_coords = coordinates(longlat_plane).surface
    @test eltype(plane_coords) == ClimaCore.Geometry.LatLongPoint{FT}
    @test typeof(plane_coords) <: ClimaCore.Fields.Field
    @test longlat_plane.xlim == FT.(xlim_longlat)
    @test longlat_plane.ylim == FT.(ylim_longlat)
    @test longlat_plane.nelements == nelements[1:2]
    @test longlat_plane.npolynomial == 0
    @test longlat_plane.periodic == (false, false)
    @test typeof(longlat_plane.space.surface) <:
          ClimaCore.Spaces.SpectralElementSpace2D

    # Box latlong
    longlat_box = HybridBox(;
        xlim = dxlim,
        ylim = dylim,
        zlim = zlim,
        longlat,
        nelements = nelements,
        npolynomial = 0,
    )
    @test longlat_box.fields.depth == zlim[2] - zlim[1]
    @test longlat_box.fields.z ==
          ClimaCore.Fields.coordinate_field(longlat_box.space.subsurface).z
    face_space = obtain_face_space(longlat_box.space.subsurface)
    z_face = ClimaCore.Fields.coordinate_field(face_space).z
    @test longlat_box.fields.z_sfc ==
          top_face_to_surface(z_face, longlat_box.space.surface)
    Δz_top, Δz_bottom = get_Δz(longlat_box.fields.z)
    @test longlat_box.fields.Δz_top == Δz_top
    @test longlat_box.fields.Δz_bottom == Δz_bottom
    longlat_box_coords = coordinates(longlat_box).subsurface
    @test eltype(longlat_box_coords) == ClimaCore.Geometry.LatLongZPoint{FT}
    @test typeof(longlat_box_coords) <: ClimaCore.Fields.Field
    @test longlat_box.xlim == FT.(xlim_longlat)
    @test longlat_box.ylim == FT.(ylim_longlat)
    @test longlat_box.zlim == FT.(zlim)
    @test longlat_box.nelements == nelements
    @test longlat_box.npolynomial == 0
    @test longlat_box.periodic == (false, false)
    @test typeof(
        ClimaCore.Spaces.horizontal_space(longlat_box.space.subsurface),
    ) <: ClimaCore.Spaces.SpectralElementSpace2D
    @test typeof(longlat_box.space.subsurface) <:
          ClimaCore.Spaces.CenterExtrudedFiniteDifferenceSpace
    @test typeof(longlat_box.space.surface) <:
          ClimaCore.Spaces.SpectralElementSpace2D
    @test obtain_surface_space(longlat_box.space.subsurface) ==
          longlat_box.space.surface
    @test obtain_surface_domain(longlat_box) ==
          Plane{FT, typeof((; surface = longlat_box.space.surface))}(
        xlim_longlat,
        ylim_longlat,
        longlat,
        nelements[1:2],
        (false, false),
        0,
        (; surface = longlat_box.space.surface),
    )

    # Column

    z_column = Column(; zlim = zlim, nelements = nelements[3])
    @test z_column.fields.z ==
          ClimaCore.Fields.coordinate_field(z_column.space.subsurface).z
    @test z_column.fields.depth == zlim[2] - zlim[1]
    face_space = obtain_face_space(z_column.space.subsurface)
    z_face = ClimaCore.Fields.coordinate_field(face_space).z
    @test z_column.fields.z_sfc ==
          top_face_to_surface(z_face, z_column.space.surface)
    Δz_top, Δz_bottom, Δz = get_Δz(z_column.fields.z)
    z = ClimaCore.Fields.coordinate_field(z_column.space.subsurface).z
    @test z_column.fields.z == z
    @test z_column.fields.Δz_top == Δz_top
    @test z_column.fields.Δz_bottom == Δz_bottom
    column_coords = coordinates(z_column).subsurface
    @test z_column.zlim == FT.(zlim)
    @test z_column.nelements[1] == nelements[3]
    @test eltype(column_coords) == ClimaCore.Geometry.ZPoint{FT}
    @test typeof(column_coords) <: ClimaCore.Fields.Field
    @test typeof(z_column.space.subsurface) <:
          ClimaCore.Spaces.CenterFiniteDifferenceSpace
    @test typeof(z_column.space.surface) <: ClimaCore.Spaces.PointSpace
    @test any(
        Array(parent(column_coords))[:][2:end] .-
        Array(parent(column_coords))[:][1:(end - 1)] .≈
        (zmax - zmin) / nelements[3],
    )
    @test obtain_surface_space(z_column.space.subsurface) ==
          z_column.space.surface
    @test obtain_surface_domain(z_column) ==
          Point{FT, typeof((; surface = z_column.space.surface))}(
        zlim[2],
        (; surface = z_column.space.surface),
    )
    z_column_stretch =
        Column(; zlim = zlim, nelements = 10, dz_tuple = FT.((0.3, 0.03)))
    column_coords = coordinates(z_column_stretch).subsurface
    @test z_column_stretch.zlim == FT.(zlim)
    dz =
        Array(parent(column_coords))[:][2:end] .-
        Array(parent(column_coords))[:][1:(end - 1)]
    @test abs(dz[1] - 0.3) < 1e-1
    @test abs(dz[end] - 0.03) < 1e-2
end

@testset "Point Domain, FT = $FT" begin
    zmin = FT(1.0)
    point = Point(; z_sfc = zmin)
    @test point.z_sfc == zmin
    point_space = point.space.surface
    @test point_space isa ClimaCore.Spaces.PointSpace
    coords = coordinates(point).surface
    @test coords isa ClimaCore.Fields.Field
    @test eltype(coords) == ClimaCore.Geometry.ZPoint{FT}
    @test Array(parent(ClimaCore.Fields.field_values(coords)))[] ==
          ClimaCore.Geometry.ZPoint(zmin).z[]
end
