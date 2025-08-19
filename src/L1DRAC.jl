module L1DRAC


# Exports

# #

## Include main files
include("Libraries.jl")

function test_simple()
    println("revise ACTUALLY works")
end

function without_export()
    println("This function is not exported, so it won't be available outside the module.")
end

end
