module L1DRAC


# Exports
export sim_params, init_Gaussian, dynamics, test_simple
#

## Include main files
include("Ensemble_libraries.jl")
include("Ensemble_custom_structs.jl")
include("Ensemble_functions.jl")
# include("Ensemble_simulation_parameters.jl")
# include("Ensemble_dynamics.jl")

function test_simple()
    println("revise really works")
end

end # Module end
