using Pkg; Pkg.activate(".julia/dev/L1DRAC") # only during dev
using Distributed, CUDA; addprocs(length(devices()));
using StaticArrays 


@time using Revise, L1DRAC # Revise.jl before L1DRAC during dev


# Simulation parameters
begin
    local Nₓ = 2
    local N_traj = 1e6
    local Δt = 0.001f0
    local tspan= (0.0f0, 30.0f0)
    local Δtₛ = 0.1f0
    sim_parameters = sim_params(Nₓ, N_traj, Δt, tspan, Δtₛ) 
end

# Dynamics
begin
 ## Van der Pol
    # initial condition distribution
    local μ₀ = [1.0f0; 1.0f0]
    local Σ₀ = [1.0f0 0.0f0; 0.0f0 1.0f0]
    local VdP_ν₀ = init_Gaussian(μ₀, Σ₀)
    
    # dynamics functions
    local function VdP_f(X, p, t)
        local μ = 2
        dX1 = X[2]
        dX2 = (μ * X[2]) * (1 - X[1]^2) - X[1]
        return SVector{2}(dX1, dX2)
    end
    local function VdP_p(X, p, t)
        dX11 = 0.1*(sin(X[1]^2) + cos(X[2]^2)) 
        dX12 = 0.1
        dX21 = 0.2
        dX22 = 0.1*sin(X[1]*X[2]^2)*cos(X[1]^2*X[2])
        return @SMatrix [dX11 dX12; dX21 dX22]
    end
    local VdP = dynamics(VdP_f, VdP_p, VdP_ν₀)  
    dynamics_tuple = (VanderPol = VdP, );
end
