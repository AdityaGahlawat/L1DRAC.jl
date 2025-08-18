using Pkg; Pkg.activate("L1DRAC.jl/") # Relative path to the julia initiation folder -- e.g. L1DRAC.jl is in ~/Desktop/GitRepos and Julia is tarted from the same
using Revise, L1DRAC # Revise.jl before L1DRAC during dev

L1DRAC.test_simple() # Test function to check if Revise is working

L1DRAC.without_export() # Test function to check if Revise is working without export