# FUNCTIONS

if myid() ≠ 1
    println("I'm loading functions")
end

# Function: constructing the final parameter set
function construct_total_params(p::sim_params, dyns::dynamics)
    if myid() == 1
        println("---> Constructing final parameter set")
    end
    # Reconstructing and distributing
    begin
        Nₓ = p.Nₓ 
        N_traj = p.N_traj
        Δt = p.Δt
        tspan = p.tspan
        Δtₛ = p.Δtₛ
        Nₓ > 1 ? ν₀ = MvNormal(dyns.init_distribution.μ₀, dyns.init_distribution.Σ₀) : ν₀ = Normal(dyns.init_distribution.μ₀, dyns.init_distribution.Σ₀)
        
        u0_reg = rand(ν₀)
        # Unfortunately, have to do the following because @SVector range is evaluated at global scope, so "u0 = @SVector [u0_reg[i] for i ∈ 1:Nₓ]" throws up an error
        if Nₓ == 1
            u0 = @SVector [u0_reg]
        elseif Nₓ == 2
            u0 = @SVector [u0_reg[i] for i ∈ 1:2]
        elseif Nₓ == 3
            u0 = @SVector [u0_reg[i] for i ∈ 1:3]
        else
            @error "construct_total_params(): Could not construct initial condition, the case for the value of Nₓ needs to be added as elseif Nₓ = val"
        end
            
        tₛ = p.tspan[1]:p.Δtₛ:p.tspan[2]
        N_traj_GPU = convert(Int64, ceil(p.N_traj/length(devices())))  
    end
    d = prepared_params(Nₓ, N_traj, Δt, tspan, Δtₛ, u0, tₛ, N_traj_GPU)
    return d
end

# Function: constructing initial conditions from the initial condition distribution
function construct_init_matrix(d::prepared_params, dyns::dynamics)
    if myid() == 1
        println("---> Sampling initial conditions")
    end
    init_matrix = Array{Matrix{Float32}, 1}(undef, length(devices()))
    
    d.Nₓ > 1 ? init_dist = MvNormal(dyns.init_distribution.μ₀, dyns.init_distribution.Σ₀) : init_dist = Normal(dyns.init_distribution.μ₀, dyns.init_distribution.Σ₀)

    
    for i ∈ 1:length(devices())
        init_matrix[i] = rand(init_dist, d.N_traj_GPU)
    end
    return init_matrix
end

# Function: prob_func
function prob_func(prob::SDEProblem, d::prepared_params, init_matrix::Array)
    if myid() == 1
        println("---> Constructing prob_func to supply sampled initial conditions to the solver")
    end
    N_t_GPU = d.N_traj_GPU
    prob_inits = map(1:N_t_GPU) do i
        remake(prob, u0 = (@SVector [init_matrix[1,i]; init_matrix[2,i]]));
    end
    return prob_inits
end

# Function: constructing SDE problem
function construct_SDE_prob(dyns::dynamics, p::sim_params)
    d::prepared_params = construct_total_params(p::sim_params, dyns::dynamics) 

    # Determining dims of diffusion matrix
    diffusion_dims = size(dyns.diffusion(zeros(d.Nₓ), nothing, 0))
    if myid() == 1
        println("---> Constructing SDE problem")
    end

    if p.Nₓ == 1
        SDE_prob = SDEProblem(dyns.drift, dyns.diffusion, d.u0, d.tspan, noise_rate_prototype = noise_rate_prototype = (@SMatrix [0])) # Noise_rate_protype takes as its argument a matrix of zeros of the same dimension as the diffusion matrix
    else
        SDE_prob = SDEProblem(dyns.drift, dyns.diffusion, d.u0, d.tspan, noise_rate_prototype = noise_rate_prototype = (@SMatrix zeros(diffusion_dims[1], diffusion_dims[2]))) # Noise_rate_protype takes as its argument a matrix of zeros of the same dimension as the diffusion matrix
    end

    return SDE_prob
end

# Function: run a small instance to compile the system dynamics
function __init__DracGpuSims(dyns::dynamics, p::sim_params)
    
    if myid() == 1
        @info "Compiling Dynamics"
    end

    d::prepared_params = construct_total_params(p::sim_params, dyns::dynamics) 

    # Determining dims of diffusion matrix
    diffusion_dims = size(dyns.diffusion(zeros(d.Nₓ), nothing, 0))
    

    if p.Nₓ == 1
        SDE_prob = SDEProblem(dyns.drift, dyns.diffusion, d.u0, d.tspan, noise_rate_prototype = noise_rate_prototype = (@SMatrix [0])) # Noise_rate_protype takes as its argument a matrix of zeros of the same dimension as the diffusion matrix
    else
        SDE_prob = SDEProblem(dyns.drift, dyns.diffusion, d.u0, d.tspan, noise_rate_prototype = noise_rate_prototype = (@SMatrix zeros(diffusion_dims[1], diffusion_dims[2]))) # Noise_rate_protype takes as its argument a matrix of zeros of the same dimension as the diffusion matrix
    end

    return SDE_prob
end

