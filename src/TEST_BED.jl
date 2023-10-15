using Distributed, CUDA; addprocs(length(devices()));
using Revise, Pkg
Pkg.activate(".julia/dev/L1DRAC")
using L1DRAC # My package



@time GPU_solve_test(sim_parameters)