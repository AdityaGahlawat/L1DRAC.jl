# Benchmark timing for L1DRAC ensemble simulations (Mk3 - @async multi-GPU)

using CUDA, L1DRAC, Plots, Plots.Measures
using LinearAlgebra, Distributions, ControlSystemsBase
using StaticArrays, UnPack, StochasticDiffEq, DiffEqGPU
using Dates, DataFrames, CSV

include(joinpath(@__DIR__, "Setup_DoubleIntegrator1D.jl"))

# === BENCHMARK FUNCTION (nominal + true_system + L1_system) ===
function benchmark_systems(max_GPUs::Int; Ntraj = 10, warmup = true, logging = true)
    numGPUs = get_numGPUs(max_GPUs)
    setup = setup_double_integrator(; Ntraj = Ntraj)
    @unpack tspan, Δₜ, Δ_saveat = setup.simulation_parameters

    backend = get_backend(numGPUs)

    # For logging only
    device_type = numGPUs == 0 ? "threads" : "gpu"
    n_devices = numGPUs == 0 ? Threads.nthreads() : numGPUs

    # Warmup run (JIT compilation) - each GPU warmed separately for @async pattern
    if warmup
        warmup_params = sim_params(tspan, Δₜ, 2, Δ_saveat)
        if numGPUs == 0
            @info "Warmup run (CPU)..."
            system_simulation(warmup_params, setup.nominal_system, CPU(); simtype = :ensemble);
            system_simulation(warmup_params, setup.true_system, CPU(); simtype = :ensemble);
            system_simulation(warmup_params, setup.true_system, setup.L1params, CPU(); simtype = :ensemble);
        else
            for gpu_id in 0:(numGPUs - 1)
                CUDA.device!(gpu_id)
                @info "Warmup run (GPU $gpu_id)..."
                system_simulation(warmup_params, setup.nominal_system, GPU(1); simtype = :ensemble);
                system_simulation(warmup_params, setup.true_system, GPU(1); simtype = :ensemble);
                system_simulation(warmup_params, setup.true_system, setup.L1params, GPU(1); simtype = :ensemble);
            end
        end
        @info "Warmup complete"
    end

    # Timed runs
    @warn "==============================\nTiming runs" backend=backend device_type=device_type n_devices=n_devices Ntraj=Ntraj
    t1 = @elapsed solA = system_simulation(setup.simulation_parameters, setup.nominal_system, backend; simtype = :ensemble)
    t2 = @elapsed solB = system_simulation(setup.simulation_parameters, setup.true_system, backend; simtype = :ensemble)
    t3 = @elapsed solC = system_simulation(setup.simulation_parameters, setup.true_system, setup.L1params, backend; simtype = :ensemble)

    # Results
    @warn "==============================\nResults" Nominal=t1 True=t2 L1DRAC=t3 Total=t1+t2+t3

    # Log to CSV
    if logging 
        logfile = joinpath(@__DIR__, "benchmark_log.csv")
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
    end
    
    cleanup_environment(backend)
    return (nominal=t1, true_sys=t2, l1drac=t3, total=t1+t2+t3), (solA, solB, solC)
    
end

# === PLOTTING ===
function plot_benchmarks()
    logfile = joinpath(@__DIR__, "benchmark_log.csv")
    df = CSV.read(logfile, DataFrame)

    # Create device class label (e.g., "1 Thread", "24 Threads", "1 GPU", "2 GPUs")
    function format_label(dtype, n)
        if dtype == "threads"
            n == 1 ? "1 Thread" : "$n Threads"
        else
            n == 1 ? "1 GPU" : "$n GPUs"
        end
    end
    df.device_class = format_label.(df.device_type, df.n_devices)

    # Get unique classes for coloring
    classes = unique(df.device_class)

    p = plot(
        title = "Computation Benchmarking",
        xlabel = "# Trajectories",
        ylabel = "Wall-clock Time (s)",
        legend = :topleft,
        yscale = :log10,
        xscale = :log10,
        size = (800, 300),
        left_margin = 5mm,
        right_margin = 5mm,
        top_margin = 2mm,
        bottom_margin = 5mm
    )

    colors = palette(:default)
    for (i, class) in enumerate(classes)
        subset = df[df.device_class .== class, :]
        sorted_subset = sort(subset, :ntraj)
        plot!(p, sorted_subset.ntraj, sorted_subset.total,
              label = "",
              linewidth = 1.5, linealpha = 0.7, color = colors[i])
        scatter!(p, sorted_subset.ntraj, sorted_subset.total,
                 label = class,
                 markersize = 6, markerstrokewidth = 1,
                 markeralpha = 1, markerstrokealpha = 1.0, color = colors[i])
    end

    savefig(p, joinpath(@__DIR__, "benchmark_results.png"))
    return p
end



# === USAGE ===
# results = benchmark_systems(1; Ntraj = 1000)   # Single GPU
# results = benchmark_systems(3; Ntraj = 1000)   # 3 GPUs
# (cleanup happens internally)
# plot_benchmarks()  # Generate scatter plot
