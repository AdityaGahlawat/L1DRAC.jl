# Benchmark timing for L1DRAC ensemble simulations (Mk2 - restructured for multi-GPU)

using Distributed, CUDA, L1DRAC

# === CALL GPU SETUP (edit max_GPUs value as needed) ===
numGPUs = setup_gpu_workers(; max_GPUs = 3)

# === PACKAGES (workers now exist, loaded on all processes) ===
@everywhere begin
    using L1DRAC
    using LinearAlgebra
    using Distributions
    using ControlSystemsBase
    using Dates
    using DataFrames
    using CSV
    using CUDA
    using StaticArrays
    using UnPack
    using StochasticDiffEq
    using DiffEqGPU
end

@everywhere include(joinpath($(@__DIR__), "Setup_DoubleIntegrator1D.jl"))

# === ASSIGN GPUs TO WORKERS (only needed for multi-GPU) ===
if numGPUs > 1
    assign_gpus_to_workers()
end

# === BENCHMARK FUNCTION ===
function computation_benchmark_setup(numGPUs::Int; Ntraj = 10, warmup = true)

    setup = setup_double_integrator()
    @unpack tspan, Δₜ, Δ_saveat = setup.simulation_parameters

    # Backend determined by numGPUs
    backend = get_backend(numGPUs)

    # For logging only
    device_type = numGPUs == 0 ? "threads" : "gpu"
    n_devices = numGPUs == 0 ? Threads.nthreads() : numGPUs

    # Warmup run (JIT compilation)
    if warmup
        @info "Warmup run (Ntraj=2)..."
        warmup_params = sim_params(tspan, Δₜ, max(numGPUs*2,2), Δ_saveat)
        system_simulation(warmup_params, setup.nominal_system, backend; simtype = :ensemble)
        system_simulation(warmup_params, setup.true_system, backend; simtype = :ensemble)
        system_simulation(warmup_params, setup.true_system, setup.L1params, backend; simtype = :ensemble)
        @info "Warmup complete"
    end

    # Create simulation parameters
    params = sim_params(tspan, Δₜ, Ntraj, Δ_saveat)

    # Timed runs
    @warn "==============================\nTiming runs" backend=backend device_type=device_type n_devices=n_devices Ntraj=Ntraj
    t1 = @elapsed system_simulation(params, setup.nominal_system, backend; simtype = :ensemble)
    t2 = @elapsed system_simulation(params, setup.true_system, backend; simtype = :ensemble)
    t3 = @elapsed system_simulation(params, setup.true_system, setup.L1params, backend; simtype = :ensemble)

    # Results
    @warn "==============================\nResults" Nominal=t1 True=t2 L1DRAC=t3 Total=t1+t2+t3

    # # Log to CSV
    # logfile = "test/benchmark_log.csv"
    # row = DataFrame(
    #     timestamp = now(),
    #     backend = string(backend),
    #     device_type = device_type,
    #     n_devices = n_devices,
    #     ntraj = Ntraj,
    #     nominal = round(t1, digits=3),
    #     true_sys = round(t2, digits=3),
    #     l1drac = round(t3, digits=3),
    #     total = round(t1+t2+t3, digits=3)
    # )
    # CSV.write(logfile, row, append=isfile(logfile))
    # @info "Results logged to $logfile"

    return (nominal=t1, true_sys=t2, l1drac=t3, total=t1+t2+t3)
end

# Run and cleanup
# results = computation_benchmark_setup(numGPUs; Ntraj = 1000)
# cleanup_gpu_environment()
