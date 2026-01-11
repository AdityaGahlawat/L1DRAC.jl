module L1DRAC

using LinearAlgebra
using DifferentialEquations
using DiffEqGPU
using CUDA
using StaticArrays
using UnPack
using ProgressLogging
using TerminalLoggers
using Logging: global_logger
using Distributions
using OptimalTransport
using JuMP
using Ipopt
using JuMP: MOI

include("types.jl")
include("auxiliary.jl")
include("nominal_system.jl")
include("true_system.jl")
include("L1_system.jl")

# Types for computation Backends
export CPU, GPU

# GPU setup functions
export get_backend, get_numGPUs, cleanup_environment

# Types for solutions, needed for multiple dispatch plot functions
export RODESolution
export EnsembleSolution

# Custom functions
export sim_params
export sys_dims
export nominal_vector_fields
export uncertain_vector_fields
export init_dist
export nom_sys
export true_sys
export system_simulation
export concat_state
export drac_params
export predictor_test
export assumption_constants
export validate
export Delta_star_Linear_Nominal

function __init__()
    global_logger(TerminalLogger())
end

end
