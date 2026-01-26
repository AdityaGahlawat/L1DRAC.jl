# GPU Sanity Check
# Simple test: CPU vs GPU with configurable max_GPUs (default=1)

# === CONFIGURATION ===
max_GPUs = 1  # Default: single GPU (optimal for most workloads)

# === SETUP ===
using CUDA
numGPUs_available = length(CUDA.devices())
numGPUs = max_GPUs == 0 ? 0 : min(max_GPUs, numGPUs_available)

println("="^60)
println("   GPU Sanity Check")
println("   Available: $numGPUs_available GPU(s)")
println("   Using: $numGPUs GPU(s)")
println("="^60)

# Multi-GPU setup (only if numGPUs > 1)
if numGPUs > 1
    using Distributed
    if nprocs() == 1
        addprocs(numGPUs)
        println("   Added $numGPUs workers for multi-GPU")
    end
    @everywhere using CUDA
    for (i, w) in enumerate(workers()[1:numGPUs])
        remotecall_wait(w) do
            CUDA.device!(i - 1)
        end
    end
    @everywhere using DiffEqGPU, StochasticDiffEq, StaticArrays
    @everywhere CUDA.allowscalar(false)
    @everywhere f_GPU(u, p, t) = -u
    @everywhere g_GPU(u, p, t) = @SVector [Float32(0.1)]
else
    # Single GPU or CPU - no Distributed needed
    using DiffEqGPU, StochasticDiffEq, StaticArrays
    CUDA.allowscalar(false)
    f_GPU(u, p, t) = -u
    g_GPU(u, p, t) = @SVector [Float32(0.1)]
end

using TerminalLoggers, Logging
global_logger(TerminalLogger())

println("   Setup complete")
println("="^60)

# === TEST ===
function run_sanity_check()
    u0 = @SVector [Float32(1.0)]
    tspan = (0.0f0, 5.0f0)
    dt = 0.0001f0

    prob = SDEProblem(f_GPU, g_GPU, u0, tspan)
    ensemble_prob = EnsembleProblem(prob)

    trajectory_counts = [100, 1_000, 10_000, 50_000, 100_000]

    println("\n" * "-"^60)
    println("   CPU vs GPU ($numGPUs GPU(s))")
    println("-"^60)

    for Ntraj in trajectory_counts
        # CPU
        t_cpu = @elapsed solve(ensemble_prob, EM(), EnsembleThreads(),
                               dt=dt, trajectories=Ntraj, saveat=0.1f0, adaptive=false)

        # GPU (with batch_size for multi-GPU support)
        batch_size = numGPUs > 1 ? cld(Ntraj, numGPUs) : Ntraj
        t_gpu = @elapsed solve(ensemble_prob, GPUEM(), EnsembleGPUKernel(CUDA.CUDABackend()),
                               dt=dt, trajectories=Ntraj, batch_size=batch_size,
                               saveat=0.1f0, adaptive=false)

        speedup = t_cpu / t_gpu
        println("  Ntraj=$Ntraj: CPU=$(round(t_cpu, digits=2))s, GPU=$(round(t_gpu, digits=2))s ($(round(speedup, digits=1))x)")
    end

    println("-"^60)
    println("   SANITY CHECK COMPLETE")
    println("="^60)
end

run_sanity_check()
