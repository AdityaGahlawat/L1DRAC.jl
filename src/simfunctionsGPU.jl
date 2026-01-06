####################################################################################
# GPU Simulation Functions (EnsembleGPUKernel)
# - METHOD 1: Nominal System (ensemble only)
# - METHOD 2: True System (ensemble only)
#
# Note: Drift/diffusion functions are defined locally inside each method to avoid
# passing structs with functions as parameters (GPU requires isbits parameters).
####################################################################################

##### MAIN SIMULATION FUNCTION (Multiple Dispatch) #####
# METHOD 1: simulation of nominal system (GPU)
function system_simulation(simulation_parameters::SimParams, nominal_system::NominalSystem, ::GPU)
    prog_steps = 1000
    @unpack tspan, Δₜ, Ntraj, Δ_saveat = simulation_parameters
    @unpack n, d = getfield(nominal_system, :sys_dims)
    @unpack nominal_ξ₀ = getfield(nominal_system, :init_dists)
    @unpack f, p = getfield(nominal_system, :nom_vec_fields)
    nom_init = rand(nominal_ξ₀)

    # Local drift/diffusion that capture f, p (no struct parameter needed)
    drift_gpu(X, _, t) = SVector{n}(f(t, X)[1:n]...)
    diffusion_gpu(X, _, t) = SMatrix{n,d}(p(t, X))

    nominal_problem = SDEProblem(drift_gpu, diffusion_gpu,
                                 SVector{n}(Float32.(nom_init)...), Float32.(tspan))

    function nominal_prob_func(prob, i, repeat)
        remake(prob, u0 = SVector{n}(Float32.(rand(nominal_ξ₀))...))
    end
    ensemble_nominal_problem = EnsembleProblem(nominal_problem, prob_func = nominal_prob_func)

    @info "Running Ensemble Simulation of Nominal System (GPU)"
    nominal_sol = solve(ensemble_nominal_problem, GPUEM(), DiffEqGPU.EnsembleGPUKernel(CUDA.CUDABackend()),
                       dt=Float32(Δₜ), trajectories=Ntraj, progress=true, progress_steps=prog_steps,
                       saveat=Float32(Δ_saveat), adaptive=false)
    @info "Done"
    return nominal_sol
end
# METHOD 2: simulation of true system (GPU)
function system_simulation(simulation_parameters::SimParams, true_system::TrueSystem, ::GPU)
	prog_steps = 1000
	@unpack tspan, Δₜ, Ntraj, Δ_saveat = simulation_parameters
	@unpack n, d = getfield(true_system, :sys_dims)
	@unpack true_ξ₀ = getfield(true_system, :init_dists)
	true_init = rand(true_ξ₀)

    # Solve the problem
    if haskey(kwargs, :simtype) && kwargs[:simtype] == :ensemble
        backend = get(kwargs, :backend, :cpu)

        if backend == :cpu
            # CPU: in-place functions, EM solver, EnsembleThreads
            true_problem = SDEProblem(_true_drift!, _true_diffusion!, true_init, tspan,
                                      noise_rate_prototype = zeros(n, d), (true_system,))
            solver = EM()
            ensemble_alg = EnsembleThreads()
            function true_prob_func_cpu(prob, i, repeat)
                remake(prob, u0 = rand(true_ξ₀))
            end
            ensemble_true_problem = EnsembleProblem(true_problem, prob_func = true_prob_func_cpu)

        elseif backend == :gpu
            # GPU: out-of-place functions, GPUEM solver, EnsembleGPUKernel, Float32
            true_problem = SDEProblem(_true_drift_oop, _true_diffusion_oop,
                                      SVector{n}(Float32.(true_init)...), Float32.(tspan), (true_system,))
            solver = GPUEM()
            ensemble_alg = DiffEqGPU.EnsembleGPUKernel(CUDA.CUDABackend())
            function true_prob_func_gpu(prob, i, repeat)
                remake(prob, u0 = SVector{n}(Float32.(rand(true_ξ₀))...))
            end
            ensemble_true_problem = EnsembleProblem(true_problem, prob_func = true_prob_func_gpu)

        else
            error("Unknown backend: $backend. Supported: :cpu, :gpu")
        end

        @info "Running Ensemble Simulation of True System" backend=backend
        true_sol = solve(ensemble_true_problem, solver, ensemble_alg, dt=Float32(Δₜ),
                        trajectories = Ntraj, progress = true, progress_steps = prog_steps,
                        saveat = Float32(Δ_saveat))
    else
        # Single trajectory (CPU only)
        true_problem = SDEProblem(_true_drift!, _true_diffusion!, true_init, tspan,
                                  noise_rate_prototype = zeros(n, d), (true_system,))
        @info "Running Single Trajectory Simulation of True System"
	    true_sol = solve(true_problem, EM(), dt=Δₜ, progress = true, progress_steps = prog_steps, saveat = Δ_saveat)
    end
	@info "Done"
	return true_sol
end

# METHOD 3: simulation of L1 DRAC system is in src/L1functions.jl
