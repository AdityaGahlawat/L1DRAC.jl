module L1DRAC


# Exports
# export test_simple
# #

# ## Include main files
# include("Ensemble_libraries.jl")
# include("Ensemble_custom_structs.jl")
# include("Ensemble_functions.jl")
# include("Ensemble_simulation_parameters.jl")
# include("Ensemble_dynamics.jl")

function test_simple()
    println("revise ACTUALLY works")
end

function without_export()
    println("This function is not exported, so it won't be available outside the module.")
end

end
