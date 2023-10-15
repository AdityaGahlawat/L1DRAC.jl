if myid() == 1
    println(">>> Loading custom structs")
end
# Simulation paramters
mutable struct sim_params
    Nₓ::Int64 # State-space dim
    N_traj::Int64 # of sample paths to be simulated
    Δt::Float32 # fixed time step for solvers
    tspan::Tuple{Float32, Float32}
    Δtₛ::Float32 # fixed time step for saving data
end

# User input of initial condition distribution
struct init_Gaussian
    μ₀::Union{Vector{Float32}, Float32} # Union is required for scalar systems
    Σ₀::Union{Matrix{Float32}, Float32}
end

# struct for the final prepared sim params
mutable struct prepared_params
    Nₓ::Int64 # State-space dim
    N_traj::Int64 # of sample paths to be simulated
    Δt::Float32 # fixed time step for solvers
    tspan::Tuple{Float32, Float32}
    Δtₛ::Float32 # fixed time step for saving data
    u0::Any # Automatically constructed initial condition using ::init_Gaussian to define the problem
    tₛ::StepRangeLen{Float32, Float64, Float64, Int64} # Vector of time instances for saveat generated using Δtₛ 
    N_traj_GPU::Int64 # Number of trajectories per GPU
end

# struct for dynamics
export dynamics
struct dynamics{drift_type, diffusion_type} # Abstract type
    drift::drift_type
    diffusion::diffusion_type
    init_distribution::init_Gaussian
end