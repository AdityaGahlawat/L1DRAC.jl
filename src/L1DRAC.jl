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

# Types for computation Backends
export CPU, GPU

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

# Intermediate constants exports
export intermediate_constants
export IntermediateConstants
export ReferenceProcessConstants, TrueProcessConstants
export DeltaHatRef, DeltaCircRef, DeltaCircledCircRef, DeltaOdotRef, DeltaOtimesRef, DeltaCircledAstRef
export DeltaHatTrue, DeltaCircledCircTrue, DeltaOdotTrue, DeltaOtimesTrue, DeltaCircledAstTrue
export frakp, frakp_prime, frakp_double_prime, Lip_f

# Bound functions exports
export alpha_zero
export Gamma_r, Gamma_a, Gamma_r_inf, Gamma_a_inf
export Theta_r, Theta_a
export optimal_bounds, bounds_sweep
export rho_r_condition, rho_a_condition, rho_r_condition_inf, rho_a_condition_inf
export bandwidth_condition_r, bandwidth_condition_a
export Gamma_r_breakdown, Gamma_a_breakdown, summarize_intermediate_constants


include("types.jl")
include("intermediateconstants.jl")
include("momentfunctions.jl")
include("boundfunctions.jl")
include("simfunctionsCPU.jl")
include("L1simfunctionsCPU.jl")
include("simfunctionsGPU.jl")
include("L1simfunctionsGPU.jl")

function __init__()
    global_logger(TerminalLogger())
end

end
