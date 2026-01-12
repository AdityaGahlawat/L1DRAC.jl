using L1DRAC
using Test
using CUDA
using LinearAlgebra
using Distributions
using StaticArrays

@testset "L1DRAC.jl" begin
    include("test_run_simulations_cpu.jl")

    if CUDA.functional()
        include("test_run_simulations_gpu.jl")
    else
        @info "Skipping GPU tests - no CUDA device available"
    end
end
