using ClimaCore.MatrixFields
import ClimaCore.MatrixFields: @name, ⋅
using ClimaCore: Spaces
import LinearAlgebra
import LinearAlgebra: I

export make_jacobian,
    make_compute_jacobian, set_dfluxBCdY!, ImplicitEquationJacobian


"""
   make_jacobian(model::AbstractModel)

Creates and returns a function which updates the auxiliary
variables `p` in place and then updates the entries of the
Jacobian matrix `W` for the `model` in place.

The default is that no updates are required, no implicit tendency is
present, and hence the timestepping is entirely explicit.

Note that the returned function `jacobian!` should be
used as `Wfact!` in `ClimaTimeSteppers.jl` and `SciMLBase.jl`.
"""
function make_jacobian(model::AbstractModel)
    update_aux! = make_update_aux(model)
    update_boundary_fluxes! = make_update_boundary_fluxes(model)
    compute_jacobian! = make_compute_jacobian(model)
    function jacobian!(W, Y, p, dtγ, t)
        update_aux!(p, Y, t)
        update_boundary_fluxes!(p, Y, t)
        compute_jacobian!(W, Y, p, dtγ, t)
    end
    return jacobian!
end

"""
    make_compute_jacobian(model::AbstractModel)

Creates and returns a function which computes the entries
of the Jacobian matrix `W` in place.

If the implicit tendency function is given by
`T!(dY, Y, p, t) = make_implicit_tendency(model)`, the Jacobian
should be given by `W_{i,j}! = ∂T!_i/∂Y_j`, where `Y_j` is the
`j-th` state variable
and `T!_i` is the implicit tendency of the `i-th` state variable.

The default is that no updates are required, but this function
must be extended for models that use implicit timestepping.
"""
function make_compute_jacobian(model::AbstractModel)
    function compute_jacobian!(W, Y, p, dtγ, t) end
    return compute_jacobian!
end

"""
    set_dfluxBCdY!(::AbstractModel,
                  ::AbstractBC,
                  ::AbstractBoundary,
                  _...)::Union{ClimaCore.Fields.FieldVector, Nothing}

A function stub which returns the derivative of the implicit tendency
term of the `model` arising from the boundary condition,
with respect to the state Y.
"""
function set_dfluxBCdY!(
    ::AbstractModel,
    ::AbstractBC,
    ::AbstractBoundary,
    _...,
)::Union{ClimaCore.Fields.FieldVector, Nothing} end

"""
    ImplicitEquationJacobian{M, S}

A struct containing the necessary information for constructing a block
Jacobian matrix used for implicit timestepping.

`matrix` is a block matrix containing one block on the diagonal for each
    variable in the model.
`solver` is a diagonal solver because our matrix is block diagonal.

Note that the diagonal, upper diagonal, and lower diagonal entry values
are stored in this struct and updated in place.
"""
struct ImplicitEquationJacobian{M, S}
    "Jacobian matrix stored as a MatrixFields.FieldMatrix"
    matrix::M
    "Solver to use for solving the matrix system"
    solver::S
end

"""
    ImplicitEquationJacobian(Y::ClimaCore.Fields.FieldVector)

Outer constructor for the ImplicitEquationJacobian Jacobian
matrix struct.

For variables that will be stepped implicitly, the Jacobian matrix
is a tridiagonal matrix. For variables that will be stepped explicitly,
the Jacobian matrix is a negative identity matrix.

To run a model with one or more prognostic variables stepped implicitly,
the Jacobian matrix must be constructed and passed to the solver.
All implicitly-stepped variables of the model should be added to the
`implicit_names` tuple, and any explicitly-stepped variables should be added
to the `explicit_names` tuple.
"""
function ImplicitEquationJacobian(Y::ClimaCore.Fields.FieldVector)
    FT = eltype(Y)
    # Only add jacobian blocks for fields that are in Y for this model
    is_in_Y(var) = MatrixFields.has_field(Y, var)

    # Define the implicit and explicit variables of any model we use
    implicit_vars =
        (@name(soil.ϑ_l), @name(soil.ρe_int), @name(canopy.energy.T))
    explicit_vars = (
        @name(soilco2.C),
        @name(soil.θ_i),
        @name(canopy.hydraulics.ϑ_l),
        @name(snow.S),
        @name(snow.Sl),
        @name(snow.U)
    )

    # Filter out the variables that are not in this model's state, `Y`
    available_implicit_vars =
        MatrixFields.unrolled_filter(is_in_Y, implicit_vars)
    available_explicit_vars =
        MatrixFields.unrolled_filter(is_in_Y, explicit_vars)

    get_jac_type(
        space::Union{
            Spaces.FiniteDifferenceSpace,
            Spaces.ExtrudedFiniteDifferenceSpace,
        },
        FT,
    ) = MatrixFields.TridiagonalMatrixRow{FT}
    get_jac_type(
        space::Union{Spaces.PointSpace, Spaces.SpectralElementSpace2D},
        FT,
    ) = MatrixFields.DiagonalMatrixRow{FT}

    get_j_field(space, FT) = zeros(get_jac_type(space, FT), space)

    implicit_blocks = MatrixFields.unrolled_map(
        var ->
            (var, var) =>
                get_j_field(axes(MatrixFields.get_field(Y, var)), FT),
        available_implicit_vars,
    )

    # We include some terms ∂T_x/∂y where x ≠ y
    # These are the off-diagonal terms in the Jacobian matrix
    # Here, we take the convention that each pair has order (T_x, y) to produce ∂T_x/∂y as above
    off_diagonal_pairs = ((@name(soil.ρe_int), @name(soil.ϑ_l)),)
    available_off_diagonal_pairs = MatrixFields.unrolled_filter(
        pair -> all(is_in_Y.(pair)),
        off_diagonal_pairs,
    )
    implicit_off_diagonals = MatrixFields.unrolled_map(
        pair ->
            (pair[1], pair[2]) =>
                get_j_field(axes(MatrixFields.get_field(Y, pair[1])), FT),
        available_off_diagonal_pairs,
    )
    # For explicitly-stepped variables, use the negative identity matrix
    # Note: We have to use FT(-1) * I instead of -I because inv(-1) == -1.0,
    # which means that multiplying inv(-1) by a Float32 will yield a Float64.
    explicit_blocks = MatrixFields.unrolled_map(
        var -> (var, var) => FT(-1) * I,
        available_explicit_vars,
    )

    matrix = MatrixFields.FieldMatrix(
        implicit_blocks...,
        implicit_off_diagonals...,
        explicit_blocks...,
    )

    # Choose algorithm based on whether off-diagonal blocks are present
    if is_in_Y(@name(soil.ρe_int)) && is_in_Y(@name(soil.ϑ_l))
        # Set up lower triangular solver for block Jacobian with off-diagonal blocks
        # Specify which variable to compute ∂T_x/∂x for first
        alg = MatrixFields.BlockLowerTriangularSolve(@name(soil.ϑ_l))
    else
        # Set up block diagonal solver for block Jacobian with no off-diagonal blocks
        alg = MatrixFields.BlockDiagonalSolve()
    end
    solver = MatrixFields.FieldMatrixSolver(alg, matrix, Y)

    return ImplicitEquationJacobian(matrix, solver)
end

Base.similar(w::ImplicitEquationJacobian) = w

function LinearAlgebra.ldiv!(
    x::Fields.FieldVector,
    A::ImplicitEquationJacobian,
    b::Fields.FieldVector,
)
    MatrixFields.field_matrix_solve!(A.solver, x, A.matrix, b)
end
