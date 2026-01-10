####################################################################################
# Nominal System Simulation
####################################################################################

##### HELPER FUNCTIONS (CPU) #####

function _nominal_drift!(dX, X, (nominal_system, ), t)
    @unpack n = getfield(nominal_system, :sys_dims)
    @unpack f, dynamics_params = getfield(nominal_system, :nom_vec_fields)
    dX[1:n] = f(t, X, dynamics_params)[1:n]
end

function _nominal_diffusion!(dX, X, (nominal_system, ), t)
    @unpack n, d = getfield(nominal_system, :sys_dims)
    @unpack p, dynamics_params = getfield(nominal_system, :nom_vec_fields)
    for i in 1:n
        for j in 1:d
            dX[i,j] = p(t, X, dynamics_params)[i,j]
        end
    end
end

##### METHODS #####

# Method 1: CPU (single trajectory + ensemble)
function system_simulation(simulation_parameters::SimParams, nominal_system::NominalSystem, ::CPU; kwargs...)
    prog_steps = 1000
    @unpack tspan, Δₜ, Ntraj, Δ_saveat = simulation_parameters
    @unpack n, d = getfield(nominal_system, :sys_dims)
    @unpack nominal_ξ₀ = getfield(nominal_system, :init_dists)
    nom_init = rand(nominal_ξ₀)

    nominal_problem = SDEProblem(_nominal_drift!, _nominal_diffusion!, nom_init, tspan,
                                 noise_rate_prototype = zeros(n, d), (nominal_system,))

    if haskey(kwargs, :simtype) && kwargs[:simtype] == :ensemble
        solver = EM()
        ensemble_alg = EnsembleThreads()
        function nominal_prob_func(prob, i, repeat)
            remake(prob, u0 = rand(nominal_ξ₀))
        end
        ensemble_nominal_problem = EnsembleProblem(nominal_problem, prob_func = nominal_prob_func)

        @info "Running Ensemble Simulation of Nominal System (CPU)"
        nominal_sol = solve(ensemble_nominal_problem, solver, ensemble_alg, dt=Δₜ,
                           trajectories = Ntraj, progress = true, progress_steps = prog_steps,
                           saveat = Δ_saveat)
    else
        @info "Running Single Trajectory Simulation of Nominal System"
        nominal_sol = solve(nominal_problem, EM(), dt=Δₜ, progress = true, progress_steps = prog_steps, saveat = Δ_saveat)
    end
    @info "Done"
    return nominal_sol
end

# Method 2: GPU - dispatches to inner private methods for single/multi GPU
function system_simulation(simulation_parameters::SimParams, nominal_system::NominalSystem, gpu::GPU; kwargs...)
    @unpack n, d = getfield(nominal_system, :sys_dims)
    if gpu.numGPUs == 1
        _system_simulation_nominal_gpu(simulation_parameters, nominal_system, Val(n), Val(d))
    else
        _system_simulation_nominal_gpu(simulation_parameters, nominal_system, Val(n), Val(d), gpu.numGPUs)
    end
end

##### INNER PRIVATE METHODS (GPU) #####

# Inner Private Method 1: Single GPU (Val pattern for compile-time dimensions)
function _system_simulation_nominal_gpu(simulation_parameters, nominal_system::NominalSystem, ::Val{n_gpu}, ::Val{d_gpu}) where {n_gpu, d_gpu}

    prog_steps = 1000
    @unpack tspan, Δₜ, Ntraj, Δ_saveat = simulation_parameters
    @unpack nominal_ξ₀ = getfield(nominal_system, :init_dists)
    @unpack f, p, dynamics_params = getfield(nominal_system, :nom_vec_fields)

    drift_gpu(X, dynamics_params, t) = f(t, X, dynamics_params)
    diffusion_gpu(X, dynamics_params, t) = p(t, X, dynamics_params)

    u0 = SVector{n_gpu}(Float32.(rand(nominal_ξ₀)))

    nominal_problem = SDEProblem(drift_gpu, diffusion_gpu, u0, Float32.(tspan), dynamics_params,
                                 noise_rate_prototype = SMatrix{n_gpu, d_gpu}(zeros(Float32, n_gpu, d_gpu)))

    function nominal_prob_func(prob, i, repeat)
        remake(prob, u0 = SVector{n_gpu}(Float32.(rand(nominal_ξ₀))))
    end
    ensemble_nominal_problem = EnsembleProblem(nominal_problem, prob_func = nominal_prob_func)

    @info "Running Ensemble Simulation of Nominal System (GPU)"
    @CUDA.time nominal_sol = solve(ensemble_nominal_problem, GPUEM(), DiffEqGPU.EnsembleGPUKernel(CUDA.CUDABackend()),
                       dt=Float32(Δₜ), trajectories=Ntraj, progress=true, progress_steps=prog_steps,
                       saveat=Float32(Δ_saveat), adaptive=false)
    @info "Done"
    return nominal_sol
end

# Inner Private Method 2: Multi-GPU (@async approach)
# NOTE: Returns vector of EnsembleSolutions for now. Step 5 will combine into single EnsembleSolution.
function _system_simulation_nominal_gpu(simulation_parameters, nominal_system::NominalSystem,
                                        ::Val{n_gpu}, ::Val{d_gpu}, numGPUs::Int) where {n_gpu, d_gpu}
    @unpack Ntraj = simulation_parameters

    ## Weighted trajectory distribution: GPU 0 gets half the load of other GPUs
    #
    # Let X = trajectories per "full" GPU (GPUs 1, 2, ...)
    # GPU 0 does X/2 (reduced capacity due to main process memory overhead)
    #
    # Equation for N GPUs:
    #   X/2 + X + X + ... + X = Ntraj
    #   X/2 + (N-1)*X         = Ntraj
    #   X * (0.5 + N - 1)     = Ntraj
    #   X = Ntraj / (N - 0.5)
    #
    # Example: 3 GPUs, 100 trajectories
    #   X = 100 / 2.5 = 40
    #   GPU 0: 20, GPU 1: 40, GPU 2: 40
    #
    other_traj = ceil(Int, Ntraj / (numGPUs - 0.5))  # X
    gpu0_traj = other_traj ÷ 2                        # X/2

    solutions = Vector{Any}(undef, numGPUs)

    # Assigning each GPU to a batch of trajectories
    @sync begin
        for gpu_id in 0:(numGPUs-1)
            @async begin
                CUDA.device!(gpu_id)
                gpu_traj = (gpu_id == 0) ? gpu0_traj : other_traj

                @info "GPU $gpu_id: solving $gpu_traj trajectories"
                gpu_params = sim_params(simulation_parameters.tspan,
                                        simulation_parameters.Δₜ,
                                        gpu_traj,
                                        simulation_parameters.Δ_saveat)
                solutions[gpu_id + 1] = _system_simulation_nominal_gpu(gpu_params, nominal_system,
                                                                       Val(n_gpu), Val(d_gpu))
            end
        end
    end

    # TODO Step 5: Combine into single EnsembleSolution
    @info "Returning vector of $(numGPUs) EnsembleSolutions (combine in Step 5)"
    return solutions
end

#= OLD Distributed.jl Multi-GPU Method (commented out - see v1.1.0-GPU-parallel-PMAP)
function _system_simulation_nominal_gpu(simulation_parameters, nominal_system::NominalSystem, ::Val{n_gpu}, ::Val{d_gpu}, numGPUs::Int) where {n_gpu, d_gpu}

    prog_steps = 1000
    @unpack tspan, Δₜ, Ntraj, Δ_saveat = simulation_parameters
    @unpack nominal_ξ₀ = getfield(nominal_system, :init_dists)
    @unpack f, p, dynamics_params = getfield(nominal_system, :nom_vec_fields)

    drift_gpu(X, dynamics_params, t) = f(t, X, dynamics_params)
    diffusion_gpu(X, dynamics_params, t) = p(t, X, dynamics_params)

    u0 = SVector{n_gpu}(Float32.(rand(nominal_ξ₀)))

    nominal_problem = SDEProblem(drift_gpu, diffusion_gpu, u0, Float32.(tspan), dynamics_params,
                                 noise_rate_prototype = SMatrix{n_gpu, d_gpu}(zeros(Float32, n_gpu, d_gpu)))

    function nominal_prob_func(prob, i, repeat)
        remake(prob, u0 = SVector{n_gpu}(Float32.(rand(nominal_ξ₀))))
    end
    ensemble_nominal_problem = EnsembleProblem(nominal_problem, prob_func = nominal_prob_func)

    batch_size = cld(Ntraj, numGPUs)
    @info "Running Ensemble Simulation of Nominal System on $numGPUs GPUs with batch size of $batch_size per GPU"
    nominal_sol = solve(ensemble_nominal_problem, GPUEM(), DiffEqGPU.EnsembleGPUKernel(CUDA.CUDABackend()),
                       dt=Float32(Δₜ), trajectories=Ntraj, batch_size=batch_size,
                       progress=true, progress_steps=prog_steps,
                       saveat=Float32(Δ_saveat), adaptive=false)
    @info "Done"
    return nominal_sol
end
=#
