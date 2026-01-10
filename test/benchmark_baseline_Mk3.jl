# Benchmark timing for L1DRAC ensemble simulations (Mk3 - @async multi-GPU)

using CUDA, L1DRAC
using LinearAlgebra, Distributions, ControlSystemsBase
using StaticArrays, UnPack, StochasticDiffEq, DiffEqGPU

include(joinpath(@__DIR__, "Setup_DoubleIntegrator1D.jl"))

# === BENCHMARK FUNCTION (nominal_system only for now) ===
function benchmark_nominal(numGPUs::Int; Ntraj = 10, warmup = true)
    setup = setup_double_integrator()
    @unpack tspan, Δₜ, Δ_saveat = setup.simulation_parameters

    backend = get_backend(numGPUs)

    if warmup
        
        warmup_params = sim_params(tspan, Δₜ, 2, Δ_saveat)
        for gpu_id in 0:(numGPUs - 1)
            CUDA.device!(gpu_id)
            @warn "Warming up GPU $gpu_id..."
            system_simulation(warmup_params, setup.nominal_system, GPU(1); simtype = :ensemble);
            # TODO: Add true_system and L1_system warmup here
        end
        # @info "Warmup complete"
    end

    params = sim_params(tspan, Δₜ, Ntraj, Δ_saveat)
    @warn "Timing nominal_system" backend=backend Ntraj=Ntraj
    t = @elapsed sol = system_simulation(params, setup.nominal_system, backend; simtype = :ensemble)

    # Handle both single GPU (EnsembleSolution) and multi-GPU (Vector of EnsembleSolutions)
    # TODO: Remove this after Step 5 when multi-GPU returns single EnsembleSolution
    if sol isa Vector
        total_traj = sum(length.(sol))
        @info "Result (multi-GPU placeholder)" time=t num_solutions=length(sol) trajectories_per_gpu=length.(sol) total_trajectories=total_traj
        result = (time = t, num_solutions = length(sol), trajectories = total_traj)
    else
        @info "Result" time=t trajectories=length(sol)
        result = (time = t, trajectories = length(sol))
    end

    GC.gc()
    CUDA.reclaim()
    return result
end

# === USAGE ===
# benchmark_nominal(1; Ntraj = 1000)   # Single GPU
# benchmark_nominal(2; Ntraj = 1000)   # 2 GPUs (after Step 2)
# benchmark_nominal(3; Ntraj = 1000)   # 3 GPUs (after Step 2)
# cleanup_gpu_environment()
