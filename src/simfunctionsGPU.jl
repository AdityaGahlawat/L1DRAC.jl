####################################################################################
# GPU Simulation Functions (EnsembleGPUKernel)
# - METHOD 1: Nominal System (ensemble only)
# - METHOD 2: True System (ensemble only)
#
# Note: Drift/diffusion functions are defined locally inside each method to avoid
# passing structs with functions as parameters (GPU requires isbits parameters).
####################################################################################

#=
GPU Compilation Requirements:
- StaticArrays (SVector, SMatrix) need dimensions known at compile-time
- SVector{n} where n is a runtime Int causes "dynamic function invocation" error
- SVector{N} where N is a type parameter (via `where {N}`) works

Solution: Val Pattern (standard Julia idiom for runtime → compile-time bridging)
- Outer function: handles multiple dispatch, extracts runtime dimensions, converts to Val
- Inner function: receives Val{N}, Val{D}, extracts compile-time N, D via `where` clause

Additionally:
- dynamics_params (NamedTuple with matrices) must be passed through SDEProblem's p argument
- User's f(t, x, params) receives dynamics_params explicitly, not via closure (closures break GPU)
=#

##### MAIN SIMULATION FUNCTION (Multiple Dispatch) #####

# METHOD 1: simulation of nominal system (GPU)
# Outer function: dispatches on NominalSystem, bridges runtime dims to compile-time
function system_simulation(simulation_parameters::SimParams, nominal_system::NominalSystem, ::GPU)
    # converts n, m, d to FIXED n_gpu, d_gpu via Val pattern for GPU compatibility
    @unpack n, d = getfield(nominal_system, :sys_dims)
    _system_simulation(simulation_parameters, nominal_system, Val(n), Val(d))
end

# Inner function: dispatches on system type, n_gpu/d_gpu are compile-time constants via `where` clause
function _system_simulation(simulation_parameters, nominal_system::NominalSystem, ::Val{n_gpu}, ::Val{d_gpu}) where {n_gpu, d_gpu}

    # (n_gpu, m_gpu, d_gpu) = (n, m, d) as compile-time constants

    prog_steps = 1000
    @unpack tspan, Δₜ, Ntraj, Δ_saveat = simulation_parameters
    @unpack nominal_ξ₀ = getfield(nominal_system, :init_dists)
    @unpack f, p, dynamics_params = getfield(nominal_system, :nom_vec_fields)

    # Wrappers: dynamics_params flows through SDEProblem's p argument to user's f, p functions
    drift_gpu(X, dynamics_params, t) = f(t, X, dynamics_params)
    diffusion_gpu(X, dynamics_params, t) = p(t, X, dynamics_params)

    # Initial condition with compile-time dimension n_gpu
    u0 = SVector{n_gpu}(Float32.(rand(nominal_ξ₀)))

    # SDEProblem: dynamics_params passed as p argument, flows to drift_gpu/diffusion_gpu
    nominal_problem = SDEProblem(drift_gpu, diffusion_gpu, u0, Float32.(tspan), dynamics_params,
                                 noise_rate_prototype = SMatrix{n_gpu, d_gpu}(zeros(Float32, n_gpu, d_gpu)))

    # prob_func for ensemble: uses compile-time n_gpu for SVector
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

# METHOD 2: simulation of true system (GPU)
# Outer function: dispatches on TrueSystem, bridges runtime dims to compile-time
function system_simulation(simulation_parameters::SimParams, true_system::TrueSystem, ::GPU)
    # converts n, d to FIXED n_gpu, d_gpu via Val pattern for GPU compatibility
    @unpack n, d = getfield(true_system, :sys_dims)
    _system_simulation(simulation_parameters, true_system, Val(n), Val(d))
end

# Inner function: dispatches on system type, n_gpu/d_gpu are compile-time constants via `where` clause
function _system_simulation(simulation_parameters, true_system::TrueSystem, ::Val{n_gpu}, ::Val{d_gpu}) where {n_gpu, d_gpu}

    # (n_gpu, d_gpu) = (n, d) as compile-time constants

    prog_steps = 1000
    @unpack tspan, Δₜ, Ntraj, Δ_saveat = simulation_parameters
    @unpack true_ξ₀ = getfield(true_system, :init_dists)
    @unpack f, p, dynamics_params = getfield(true_system, :nom_vec_fields)
    @unpack Λμ, Λσ = getfield(true_system, :unc_vec_fields)

    # Wrappers: true system = nominal + uncertainty
    drift_gpu(X, dynamics_params, t) = f(t, X, dynamics_params) + Λμ(t, X, dynamics_params)
    diffusion_gpu(X, dynamics_params, t) = p(t, X, dynamics_params) + Λσ(t, X, dynamics_params)

    # Initial condition with compile-time dimension n_gpu
    u0 = SVector{n_gpu}(Float32.(rand(true_ξ₀)))

    # SDEProblem: dynamics_params passed as p argument, flows to drift_gpu/diffusion_gpu
    true_problem = SDEProblem(drift_gpu, diffusion_gpu, u0, Float32.(tspan), dynamics_params,
                              noise_rate_prototype = SMatrix{n_gpu, d_gpu}(zeros(Float32, n_gpu, d_gpu)))

    # prob_func for ensemble: uses compile-time n_gpu for SVector
    function true_prob_func(prob, i, repeat)
        remake(prob, u0 = SVector{n_gpu}(Float32.(rand(true_ξ₀))))
    end
    ensemble_true_problem = EnsembleProblem(true_problem, prob_func = true_prob_func)

    @info "Running Ensemble Simulation of True System (GPU)"
    @CUDA.time true_sol = solve(ensemble_true_problem, GPUEM(), DiffEqGPU.EnsembleGPUKernel(CUDA.CUDABackend()),
                     dt=Float32(Δₜ), trajectories=Ntraj, progress=true, progress_steps=prog_steps,
                     saveat=Float32(Δ_saveat), adaptive=false)
    @info "Done"
    return true_sol
end

# METHOD 3: simulation of L1 DRAC system is in src/L1functions.jl
