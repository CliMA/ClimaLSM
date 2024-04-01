# This script tests the output of the ClimaLand TwoStream implementation
# against the T. Quaife pySellersTwoStream implementation by providing
# the same setups to each model and checking that the outputs are equal.
# The output of the python module for a variety of inputs is given in the
# linked file which then gets read and compared to the Clima version, ensuring
# the FAPAR of each model is within 1/2 of a percentage point.

using Test
using ClimaLand
using ClimaLand.Canopy
using DelimitedFiles
using ArtifactWrappers

# Get the test data
# data_file = ArtifactWrapper(
#     @__DIR__,
#     "PySellersTwoStream Data",
#     ArtifactFile[ArtifactFile(
#         url = "https://caltech.box.com/shared/static/e7angzdnw18tmf8gctse5flsrkjsrlhx.csv",
#         filename = "2_str_test_data2.csv",
#     ),],
# )
# datapath = get_data_folder(data_file)
# data = joinpath(datapath, "2_str_test_data2.csv")
data = "/Users/espeer/Desktop/ClimaLand/2_str_test_data.csv"
test_set = readdlm(data, ',')

# Floating point precision to use
for FT in (Float32, Float64)
    @testset "Two-Stream Model Correctness, FT = $FT" begin
        # Read the conditions for each setup parameter from the test file
        column_names = test_set[1, :]
        θs = acos.(FT.(test_set[2:end, column_names .== "mu"]))
        LAI = FT.(test_set[2:end, column_names .== "LAI"])
        χl = FT.(test_set[2:end, column_names .== "chi"])
        α_PAR_leaf = FT.(test_set[2:end, column_names .== "rho"])
        τ = FT.(test_set[2:end, column_names .== "tau"])
        a_soil = FT.(test_set[2:end, column_names .== "a_soil"])
        n_layers = UInt64.(test_set[2:end, column_names .== "n_layers"])
        PropDif = FT.(test_set[2:end, column_names .== "prop_diffuse"])

        # Read the result for each setup from the Python output
        py_FAPAR = FT.(test_set[2:end, column_names .== "FAPAR"])

        # Python code does not use clumping index, and λ_γ does not impact FAPAR
        Ω = FT(1)
        λ_γ_PAR = FT(5e-7)
        λ_γ_NIR = FT(1.65e-6)

        # Test over all rows in the stored output from the Python module
        for i in 2:(size(test_set, 1) - 1)

            # Set the parameters based on the setup read from the file
            RT_params = TwoStreamParameters{FT}(;
                χl = χl[i],
                α_PAR_leaf = α_PAR_leaf[i],
                τ_PAR_leaf = τ[i],
                Ω = Ω,
                λ_γ_PAR = λ_γ_PAR,
                λ_γ_NIR = λ_γ_NIR,
                n_layers = n_layers[i],
            )

            # Initialize the TwoStream model
            RT = TwoStreamModel(RT_params)

            # Compute the predicted FAPAR using the ClimaLand TwoStream implementation
            K = extinction_coeff(χl[i], θs[i])
            output = plant_absorbed_pfd(
                RT,
                FT(1),
                RT_params.α_PAR_leaf,
                RT_params.τ_PAR_leaf,
                LAI[i],
                K,
                θs[i],
                a_soil[i],
                PropDif[i],
            )
            FAPAR = output.abs
            # Check that the predictions are app. equivalent to the Python model
            @test isapprox(py_FAPAR[i], FAPAR, atol = 0.01)
        end
    end
end
