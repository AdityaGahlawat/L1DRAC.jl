if myid() == 1
    println(">>> Loading simulation parameters: USER DEFINED")
    @info "---> simulation parameters can be edited after SDE_GPU_main module is loaded"
    @warn "---> Do NOT change Nₓ as it will destroy benefits of precomilation"
end


begin
    local Nₓ = 2
    local N_traj = 1e6
    local Δt = 0.001f0
    local tspan= (0.0f0, 30.0f0)
    local Δtₛ = 0.1f0
    sim_parameters = sim_params(Nₓ, N_traj, Δt, tspan, Δtₛ) 
end