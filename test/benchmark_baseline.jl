# Benchmark timing for L1DRAC ensemble simulations



max_GPUs = 0


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
    using Distributed
    if nprocs() == 1
        addprocs(numGPUs)
    end
    @everywhere using CUDA
    for (i, w) in enumerate(workers()[1:numGPUs])
        remotecall_wait(w) do
            CUDA.device!(i - 1)
        end
    end
    @everywhere using DiffEqGPU, StochasticDiffEq, StaticArrays
    @info "Multi-GPU mode" devices=join(["CuDevice($i): $(CUDA.name(CUDA.CuDevice(i)))" for i in 0:(numGPUs-1)], ", ")
elseif numGPUs == 1
    using DiffEqGPU, StaticArrays
    @info "Single GPU mode" device="CuDevice(0): $(CUDA.name(CUDA.CuDevice(0)))"
else
    @info "CPU only mode" threads=Threads.nthreads()
end

include("Setup_DoubleIntegrator1D.jl")

function run_benchmark(; Ntraj=10, warmup=true)
    # Get system setup
    setup = setup_double_integrator()

    # Backend determined by numGPUs (set at top of file)
    if numGPUs == 0
        backend = :cpu
        device_type = "threads"
        n_devices = Threads.nthreads()
    else
        backend = :gpu
        device_type = "gpu"
        n_devices = numGPUs
    end

    # Warmup run (JIT compilation)
    if warmup
        @info "Warmup run (Ntraj=2)..."
        warmup_params = sim_params(setup.tspan, setup.Δₜ, 2, setup.Δ_saveat)
        system_simulation(warmup_params, setup.nominal_system; simtype=:ensemble, backend=backend)
        system_simulation(warmup_params, setup.true_system; simtype=:ensemble, backend=backend)
        system_simulation(warmup_params, setup.true_system, setup.L1params; simtype=:ensemble, backend=backend)
        @info "Warmup complete"
    end

    # Create simulation parameters
    params = sim_params(setup.tspan, setup.Δₜ, Ntraj, setup.Δ_saveat)

    # Timed runs
    @warn "==============================\nTiming runs" backend=backend device_type=device_type n_devices=n_devices Ntraj=Ntraj
    t1 = @elapsed system_simulation(params, setup.nominal_system; simtype=:ensemble, backend=backend);
    t2 = @elapsed system_simulation(params, setup.true_system; simtype=:ensemble, backend=backend);
    t3 = @elapsed system_simulation(params, setup.true_system, setup.L1params; simtype=:ensemble, backend=backend);

    # Results
    @warn "==============================\nResults" Nominal=t1 True=t2 L1DRAC=t3 Total=t1+t2+t3

    # Log to CSV
    logfile = "test/benchmark_log.csv"
    row = DataFrame(
        timestamp = now(),
        backend = string(backend),
        device_type = device_type,
        n_devices = n_devices,
        ntraj = Ntraj,
        nominal = round(t1, digits=3),
        true_sys = round(t2, digits=3),
        l1drac = round(t3, digits=3),
        total = round(t1+t2+t3, digits=3)
    )
    CSV.write(logfile, row, append=isfile(logfile))
    @info "Results logged to $logfile"

    return (nominal=t1, true_sys=t2, l1drac=t3, total=t1+t2+t3)
end

# Run benchmark (uncomment to execute)
run_benchmark(Ntraj=10)

@info "Removing Workers"
if nprocs() > 1 # Workers exist
    rmprocs(workers())
end