using Pkg; Pkg.activate(".julia/dev/L1DRAC") # only during dev

using Distributed, CUDA; addprocs(length(devices()));
using Revise 
using L1DRAC # Revise.jl before L1DRAC during dev


test_simple()

# @time GPU_solve_test(sim_parameters)