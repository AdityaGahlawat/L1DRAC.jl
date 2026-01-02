# Benchmark timing for L1DRAC ensemble simulations

using Dates, DataFrames, CSV

include("Setup_DoubleIntegrator1D.jl")

function run_benchmark(; backend=:cpu, Ntraj=100)
    # Get system setup
    setup = setup_double_integrator()

    # Device info
    device_type = "threads"
    n_devices = Threads.nthreads()

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

# Run benchmark
run_benchmark(backend=:cpu, Ntraj=100)
