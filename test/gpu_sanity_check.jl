# GPU Sanity Check - Find GPU speedup crossover point

using CUDA
using DiffEqGPU
using StochasticDiffEq
using StaticArrays
using TerminalLoggers
using Logging: global_logger
global_logger(TerminalLogger())

CUDA.allowscalar(false)

# Module scope, OUT-OF-PLACE functions (required for EnsembleGPUKernel)
f_GPU(u, p, t) = -u
g_GPU(u, p, t) = @SVector [Float32(0.1)]

# Config struct for parameters (no global variable pollution)
struct GPUSanityConfig
    trajectory_counts::Vector{Int}
    dt::Float32
    tspan::Tuple{Float32, Float32}
end

function run_gpu_sanity(config::GPUSanityConfig)
    u0 = @SVector [Float32(1.0)]
    prob = SDEProblem(f_GPU, g_GPU, u0, config.tspan)
    ensemble_prob = EnsembleProblem(prob)

    println("="^60)
    println("   FINDING GPU SPEEDUP CROSSOVER POINT")
    println("   Using EnsembleGPUKernel + GPUEM")
    println("="^60)
    println()

    for Ntraj in config.trajectory_counts
        println("-"^60)
        println(">>> TESTING Ntraj = $Ntraj <<<")
        println("-"^60)

        # GPU with EnsembleGPUKernel + GPUEM
        println("  Running GPU (EnsembleGPUKernel + GPUEM)...")
        t_gpu = @elapsed solve(ensemble_prob, GPUEM(), EnsembleGPUKernel(CUDA.CUDABackend()),
                               dt=config.dt, trajectories=Ntraj, saveat=0.1f0, adaptive=false)
        println("  GPU done: $(round(t_gpu, digits=2))s")

        # CPU with EnsembleThreads + EM (with progress bar via TerminalLoggers)
        println("  Running CPU (EnsembleThreads + EM)...")
        t_cpu = @elapsed solve(ensemble_prob, EM(), EnsembleThreads(),
                               dt=config.dt, trajectories=Ntraj, saveat=0.1f0, adaptive=false)
        println("  CPU done: $(round(t_cpu, digits=2))s")

        speedup = t_cpu / t_gpu

        if speedup > 1
            println("  SPEEDUP: $(round(speedup, digits=2))x (GPU is faster)")
        else
            println("  NO SPEEDUP: $(round(speedup, digits=2))x (CPU is faster)")
        end
        println()
    end

    println("="^60)
    println("   TEST COMPLETE")
    println("="^60)
end

# Run test
config = GPUSanityConfig(
    [100, 500, 1000, 5000, 10_000, 15_000, 20_000, 25_000, 30_000, 35_000],
    0.0001f0,
    (0.0f0, 5.0f0)
)
run_gpu_sanity(config)
