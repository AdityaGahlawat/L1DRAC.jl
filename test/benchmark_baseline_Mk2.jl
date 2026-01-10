# Benchmark timing for L1DRAC ensemble simulations (Mk2 - restructured for multi-GPU)

using Distributed, CUDA

# === CLEANUP FUNCTION ===
function cleanup_gpu_environment()
    @info "Removing Stale Workers and Shedding Memory"
    if nprocs() > 1
        rmprocs(workers())
    end
    GC.gc()
    CUDA.reclaim()
end

# === GPU SETUP FUNCTION ===
function setup_gpu_workers(; max_GPUs = 1)

    cleanup_gpu_environment()

    @assert max_GPUs isa Integer && max_GPUs >= 0 "max_GPUs must be a non-negative integer"

    available_GPUs = length(CUDA.devices())
        
    numGPUs = min(max_GPUs, available_GPUs)

     # Setup Distributed workers for multi-GPU

    if numGPUs > 1 && nprocs() == 1
        
        addprocs(numGPUs)
        @everywhere @eval using CUDA

        for (i, w) in enumerate(workers()[1:numGPUs])
            remotecall_wait(w) do
                CUDA.device!(i - 1)
            end
        end
        @info "Multi-GPU mode" Detected = available_GPUs Assigned = numGPUs Devices=join(["CuDevice($i): $(CUDA.name(CUDA.CuDevice(i)))" for i in 0:(numGPUs-1)], ", ")
    elseif numGPUs == 1
        @info "Single GPU mode" Detected = available_GPUs Assigned = numGPUs Device="CuDevice(0): $(CUDA.name(CUDA.CuDevice(0)))"
    else
        @info "CPU only mode" Detected = available_GPUs Assigned = numGPUs Threads=Threads.nthreads()
    end

    return numGPUs
end

# === CALL GPU SETUP (edit max_GPUs value as needed) ===
numGPUs = setup_gpu_workers(; max_GPUs = 2)

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

# === BENCHMARK FUNCTION ===
function computation_benchmark_setup(numGPUs::Int; Ntraj = 10, warmup = true)

    setup = setup_double_integrator()
    @unpack tspan, Δₜ, Δ_saveat = setup.simulation_parameters

    # Backend determined by numGPUs
    if numGPUs == 0
        backend = CPU()
        device_type = "threads"
        n_devices = Threads.nthreads()
    else
        backend = GPU(numGPUs)
        device_type = "gpu"
        n_devices = numGPUs
    end

    # Warmup run (JIT compilation)
    if warmup
        @info "Warmup run (Ntraj=2)..."
        warmup_params = sim_params(tspan, Δₜ, max(numGPUs*2,0), Δ_saveat)
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
    cleanup_gpu_environment()
end


