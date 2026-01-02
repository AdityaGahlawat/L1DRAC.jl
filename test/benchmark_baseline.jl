# Benchmark timing for L1DRAC ensemble simulations

using Dates, DataFrames, CSV

include("Setup_DoubleIntegrator1D.jl")

# Backend configuration
backend = :cpu
device_type = "threads"
n_devices = Threads.nthreads()

# Warmup (trigger JIT compilation)
@warn "==============================\nWarmup..."
Ntraj_warmup = 2
warmup_params = sim_params(tspan, Δₜ, Ntraj_warmup, Δ_saveat)
system_simulation(warmup_params, nominal_system; simtype=:ensemble, backend=backend);
system_simulation(warmup_params, true_system; simtype=:ensemble, backend=backend);
system_simulation(warmup_params, true_system, L1params; simtype=:ensemble, backend=backend);
@warn "Warmup complete."

# Timed runs
# Change Ntraj in Setup_DoubleIntegrator1D.jl to adjust number of trajectories
@warn "==============================\nTiming runs" backend=backend device_type=device_type n_devices=n_devices Ntraj=Ntraj
t1 = @elapsed system_simulation(simulation_parameters, nominal_system; simtype=:ensemble, backend=backend);
t2 = @elapsed system_simulation(simulation_parameters, true_system; simtype=:ensemble, backend=backend);
t3 = @elapsed system_simulation(simulation_parameters, true_system, L1params; simtype=:ensemble, backend=backend);

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
