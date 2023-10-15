__precompile__

module SDE_GPU_main

using Distributed

if myid() == 1
    @info "======== LOADING: SDE_GPU_main module ==========="
end

## Export list
export sim_parameters, dynamics_tuple, name_dynamics, name_dynamics_string, GPU_solve_test

## Loading files


include("Ensemble_libraries.jl")
include("Ensemble_custom_structs.jl")
include("Ensemble_functions.jl")
include("Ensemble_simulation_parameters.jl")
include("Ensemble_dynamics.jl")

## Main function -- TEST

function GPU_solve_test(s::sim_params)
    prob_GPU = construct_SDE_prob(dynamics_tuple[1], s)
    d = construct_total_params(s, dynamics_tuple[1])
    init_matrix = construct_init_matrix(d, dynamics_tuple[1])
    M = Array{Future, 1}(undef, length(devices()))
    for (i, gpu) in zip(1:length(devices()), devices())
        println("Working on GPU-", gpu, " at Worker #",  workers()[i])
        
        M[i] = @spawnat workers()[i] begin
            CUDA.allowscalar(false)
            CUDA.device!(gpu)
            CUDA.@sync ts, us = DiffEqGPU.vectorized_solve(cu(prob_func(prob_GPU, d, init_matrix[i])), prob_GPU, GPUEM(); dt = d.Δt, saveat = d.tₛ, debug = false)
            Array(us)
        end
    end#---------------------------------
    # Fetching Solutions
    #---------------------------------    
    sol = Array{Matrix{SVector{2, Float32}}}(undef, 3)
    for i ∈ 1:length(devices())
        println("Fetching solution from Worker #", workers()[i])
        sol[i] = fetch(M[i]);
    end
end
## Main function -- TEST END

## Initializing

    # Getting the name of dynamics
        name_dynamics = collect(keys(dynamics_tuple));
    # Constructing lightweight parameter set for a quick initialization
        sim_parameters_compile = sim_params(sim_parameters.Nₓ, 10, 0.5f0, (0.0f0, 1.0f0), 0.5f0)
    # Everything in ___init__() will be executed (compiled) at module load time 

function __init__()
    if myid() == 1
        @info ">>> Initializing functions"
    end
    # for i = 1:length(dynamics_tuple)
    #     if myid() == 1
    #         println("Setting up SDE prob for: ", name_dynamics[i])
    #     end    
    #     precompile_prob = construct_SDE_prob(dynamics_tuple[i], sim_parameters_compile);
    # end

    GPU_solve_test(sim_parameters_compile)
    

    if myid() == 1
        @info "======== LOADING: SDE_GPU_main module: COMPLETE ==========="
        println(" ")
    end
end
#

end # module end