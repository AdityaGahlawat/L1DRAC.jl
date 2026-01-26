####################################################################################
# True System Simulation (without L1-DRAC control)
####################################################################################

##### HELPER FUNCTIONS (CPU) #####

function _uncertain_drift!(dX, X, (true_system, ), t)
    @unpack n = getfield(true_system, :sys_dims)
    @unpack Λμ = getfield(true_system, :unc_vec_fields)
    @unpack dynamics_params = getfield(true_system, :nom_vec_fields)
    dX[1:n] = Λμ(t, X, dynamics_params)[1:n]
end

function _true_drift!(dX, X, (true_system, ), t)
    @unpack n = getfield(true_system, :sys_dims)
    @unpack f, dynamics_params = getfield(true_system, :nom_vec_fields)
    @unpack Λμ = getfield(true_system, :unc_vec_fields)
    Fμ(t, X) = f(t, X, dynamics_params) + Λμ(t, X, dynamics_params)
    dX[1:n] = Fμ(t,X)[1:n]
end

function _uncertain_diffusion!(dX, X, (true_system, ), t)
    @unpack n, d = getfield(true_system, :sys_dims)
    @unpack Λσ = getfield(true_system, :unc_vec_fields)
    @unpack dynamics_params = getfield(true_system, :nom_vec_fields)
    for i in 1:n
        for j in 1:d
            dX[i,j] = Λσ(t, X, dynamics_params)[i,j]
        end
    end
end

function _true_diffusion!(dX, X, (true_system, ), t)
    @unpack n, d = getfield(true_system, :sys_dims)
    @unpack p, dynamics_params = getfield(true_system, :nom_vec_fields)
    @unpack Λσ = getfield(true_system, :unc_vec_fields)
    Fσ(t, X) = p(t, X, dynamics_params) + Λσ(t, X, dynamics_params)
    for i in 1:n
        for j in 1:d
            dX[i,j] = Fσ(t, X)[i,j]
        end
    end
end

##### METHODS #####

# Method 1: CPU (single trajectory + ensemble)
function system_simulation(simulation_parameters::SimParams, true_system::TrueSystem, ::CPU; kwargs...)
    prog_steps = 1000
    @unpack tspan, Δₜ, Ntraj, Δ_saveat = simulation_parameters
    @unpack n, d = getfield(true_system, :sys_dims)
    @unpack true_ξ₀ = getfield(true_system, :init_dists)
    true_init = rand(true_ξ₀)

    true_problem = SDEProblem(_true_drift!, _true_diffusion!, true_init, tspan,
                              noise_rate_prototype = zeros(n, d), (true_system,))

    if haskey(kwargs, :simtype) && kwargs[:simtype] == :ensemble
        solver = EM()
        ensemble_alg = EnsembleThreads()
        function true_prob_func(prob, i, repeat)
            remake(prob, u0 = rand(true_ξ₀))
        end
        ensemble_true_problem = EnsembleProblem(true_problem, prob_func = true_prob_func)

        @info "Running Ensemble Simulation of True System (CPU)"
        true_sol = solve(ensemble_true_problem, solver, ensemble_alg, dt=Δₜ,
                        trajectories = Ntraj, progress = true, progress_steps = prog_steps,
                        saveat = Δ_saveat)
    else
        @info "Running Single Trajectory Simulation of True System"
        true_sol = solve(true_problem, EM(), dt=Δₜ, progress = true, progress_steps = prog_steps, saveat = Δ_saveat)
    end
    @info "Done"
    return [true_sol]
end

# Method 2: GPU - dispatches to inner private methods for single/multi GPU
function system_simulation(simulation_parameters::SimParams, true_system::TrueSystem, gpu::GPU; kwargs...)
    @unpack n, d = getfield(true_system, :sys_dims)
    if gpu.numGPUs == 1
        _system_simulation_true_gpu(simulation_parameters, true_system, Val(n), Val(d))
    else
        _system_simulation_true_gpu(simulation_parameters, true_system, Val(n), Val(d), gpu.numGPUs)
    end
end

##### INNER PRIVATE METHODS (GPU) #####

# Helper function: Core GPU solve kernel (called by both single-GPU and multi-GPU methods)
# Returns raw EnsembleSolution (not wrapped in array)
function _true_gpu_solve_kernel(tspan, Δₜ, Ntraj, Δ_saveat, true_ξ₀, f, p, Λμ, Λσ, dynamics_params,
                                ::Val{n_gpu}, ::Val{d_gpu}) where {n_gpu, d_gpu}
    prog_steps = 1000

    drift_gpu(X, dynamics_params, t) = f(t, X, dynamics_params) + Λμ(t, X, dynamics_params)
    diffusion_gpu(X, dynamics_params, t) = p(t, X, dynamics_params) + Λσ(t, X, dynamics_params)

    u0 = SVector{n_gpu}(Float32.(rand(true_ξ₀)))

    true_problem = SDEProblem(drift_gpu, diffusion_gpu, u0, Float32.(tspan), dynamics_params,
                              noise_rate_prototype = SMatrix{n_gpu, d_gpu}(zeros(Float32, n_gpu, d_gpu)))

    function true_prob_func(prob, i, repeat)
        remake(prob, u0 = SVector{n_gpu}(Float32.(rand(true_ξ₀))))
    end
    ensemble_true_problem = EnsembleProblem(true_problem, prob_func = true_prob_func)

    true_sol = solve(ensemble_true_problem, GPUEM(), DiffEqGPU.EnsembleGPUKernel(CUDA.CUDABackend()),
                     dt=Float32(Δₜ), trajectories=Ntraj, progress=true, progress_steps=prog_steps,
                     saveat=Float32(Δ_saveat), adaptive=false)

    return true_sol
end

# Inner Private Method 1: Single GPU - calls _true_gpu_solve_kernel
function _system_simulation_true_gpu(simulation_parameters, true_system::TrueSystem, ::Val{n_gpu}, ::Val{d_gpu}) where {n_gpu, d_gpu}
    @unpack tspan, Δₜ, Ntraj, Δ_saveat = simulation_parameters
    @unpack true_ξ₀ = getfield(true_system, :init_dists)
    @unpack f, p, dynamics_params = getfield(true_system, :nom_vec_fields)
    @unpack Λμ, Λσ = getfield(true_system, :unc_vec_fields)

    @info "Running Ensemble Simulation of True System (GPU)"
    @CUDA.time true_sol = _true_gpu_solve_kernel(tspan, Δₜ, Ntraj, Δ_saveat, true_ξ₀, f, p, Λμ, Λσ, dynamics_params,
                                                  Val(n_gpu), Val(d_gpu))
    @info "Done"
    return [true_sol]
end

# Inner Private Method 2: Multi-GPU - calls _true_gpu_solve_kernel in each @async
# Returns Vector{EnsembleSolution} (one per GPU)
function _system_simulation_true_gpu(simulation_parameters, true_system::TrueSystem,
                                     ::Val{n_gpu}, ::Val{d_gpu}, numGPUs::Int) where {n_gpu, d_gpu}
    @unpack tspan, Δₜ, Ntraj, Δ_saveat = simulation_parameters
    @unpack true_ξ₀ = getfield(true_system, :init_dists)
    @unpack f, p, dynamics_params = getfield(true_system, :nom_vec_fields)
    @unpack Λμ, Λσ = getfield(true_system, :unc_vec_fields)

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

    @sync begin
        for gpu_id in 0:(numGPUs-1)
            @async begin
                CUDA.device!(gpu_id)
                gpu_traj = (gpu_id == 0) ? gpu0_traj : other_traj
                @info "GPU $gpu_id: solving $gpu_traj trajectories"

                # Calls helper with DIFFERENT function name - clear call graph
                gpu_sol = _true_gpu_solve_kernel(tspan, Δₜ, gpu_traj, Δ_saveat, true_ξ₀, f, p, Λμ, Λσ, dynamics_params,
                                                 Val(n_gpu), Val(d_gpu))
                solutions[gpu_id + 1] = gpu_sol
            end
        end
    end

    @info "Multi-GPU complete" num_solutions=numGPUs trajectories_per_gpu=length.(solutions)
    return solutions
end
