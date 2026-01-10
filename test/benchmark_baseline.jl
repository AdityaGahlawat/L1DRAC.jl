# Benchmark timing for L1DRAC ensemble simulations

using L1DRAC
using LinearAlgebra
using Distributions
using ControlSystemsBase
using Dates
using DataFrames
using CSV
using CUDA
using Distributed
using StaticArrays
using UnPack
using StochasticDiffEq
using DiffEqGPU

# include("Setup_DoubleIntegrator1D.jl")

function computation_benchmark_setup(;max_GPUs = 0, Ntraj = 10)

    # max_GPUs: how many GPUs to use (0 = CPU only, default = 1)
    # User can define max_GPUs before running this script to override
    # No wrong answer: code will use min(max_GPUs, available) anyway
    # GPU Configuration
    if !@isdefined(max_GPUs)
        max_GPUs = 1
    end


    # Validate: must be non-negative integer
    @assert max_GPUs isa Integer && max_GPUs >= 0 "max_GPUs must be a non-negative integer (0, 1, 2, ...)"

    # Detect how many GPUs the machine has
    available_GPUs = length(CUDA.devices())

    # Use the smaller of: what user wants vs what machine has
    # (no wrong answer - asking for 100 GPUs when you have 3 just gives you 3)
    numGPUs = min(max_GPUs, available_GPUs)


    # === DISTRIBUTED SETUP FOR MULTI-GPU ===
    # numGPUs > 1: spawn worker processes, assign each to a GPU
    # numGPUs == 1: single GPU, no Distributed needed
    # numGPUs == 0: CPU only, no GPU packages needed
    if numGPUs > 1
        if nprocs() == 1
            addprocs(numGPUs)
        end
        @everywhere @eval using CUDA
        for (i, w) in enumerate(workers()[1:numGPUs])
            remotecall_wait(w) do
                CUDA.device!(i - 1)
            end
        end
        @everywhere @eval using DiffEqGPU, StochasticDiffEq, StaticArrays, L1DRAC
        @info "Multi-GPU mode" devices=join(["CuDevice($i): $(CUDA.name(CUDA.CuDevice(i)))" for i in 0:(numGPUs-1)], ", ")
    elseif numGPUs == 1
        @info "Single GPU mode" device="CuDevice(0): $(CUDA.name(CUDA.CuDevice(0)))"
    else
        @info "CPU only mode" threads=Threads.nthreads()
    end
    @everywhere @eval include(joinpath($(@__DIR__), "Setup_DoubleIntegrator1D.jl"))

    

    function run_benchmark(; Ntraj=10, warmup=true)
        # Get system setup
        
        setup = setup_double_integrator()
        @unpack tspan, Δₜ, Δ_saveat = setup.simulation_parameters

        # Backend determined by numGPUs (set at top of file)
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
            warmup_params = sim_params(tspan, Δₜ, 2, Δ_saveat)
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

    # Run benchmark (uncomment to execute)
    run_benchmark(Ntraj=Ntraj)

    @info "Removing Workers and Shedding Memory"
    if nprocs() > 1 # Workers exist
        rmprocs(workers())
    end
    GC.gc()
    CUDA.reclaim()

end