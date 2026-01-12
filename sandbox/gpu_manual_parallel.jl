# GPU Manual Parallel Test
# Theory: Manual @async dispatch should beat DiffEqGPU's pmap

# === SETUP (must be at top level) ===
using Distributed

using CUDA
const NUMGPUS = length(CUDA.devices())
println("Found $NUMGPUS GPU(s)")

if NUMGPUS < 2
    error("Need at least 2 GPUs for this test")
end

# Add workers if needed
if nprocs() == 1
    addprocs(NUMGPUS)
    println("Added $NUMGPUS workers")
end

# Assign each worker to a GPU
@everywhere using CUDA
for (i, w) in enumerate(workers())
    remotecall_wait(w) do
        CUDA.device!(i - 1)
    end
end
println("Workers assigned to GPUs")

# Load packages on all workers
@everywhere using DiffEqGPU, StochasticDiffEq, StaticArrays
@everywhere CUDA.allowscalar(false)

# Define problem on all workers
@everywhere f_GPU(u, p, t) = -u
@everywhere g_GPU(u, p, t) = @SVector [Float32(0.1)]

# Pre-create problem on each worker (avoids serialization)
@everywhere begin
    const LOCAL_u0 = @SVector [Float32(1.0)]
    const LOCAL_tspan = (0.0f0, 5.0f0)
    const LOCAL_prob = SDEProblem(f_GPU, g_GPU, LOCAL_u0, LOCAL_tspan)
    const LOCAL_ensemble = EnsembleProblem(LOCAL_prob)
end

println("Setup complete\n")

# === TEST FUNCTIONS ===

# Method 1: DiffEqGPU's built-in (uses pmap internally)
function solve_diffeqgpu(ensemble_prob, Ntraj, ngpus; dt, saveat)
    batch_size = cld(Ntraj, ngpus)
    solve(ensemble_prob, GPUEM(), EnsembleGPUKernel(CUDA.CUDABackend()),
          dt=dt, trajectories=Ntraj, batch_size=batch_size,
          saveat=saveat, adaptive=false)
end

# Method 2: Manual @async dispatch (sends ensemble_prob to workers)
function solve_manual_parallel(ensemble_prob, Ntraj, ngpus; dt, saveat)
    batch_size = cld(Ntraj, ngpus)
    results = Vector{Any}(undef, ngpus)

    @sync begin
        for (i, w) in enumerate(workers()[1:ngpus])
            @async begin
                # Last batch might be smaller
                traj_count = (i == ngpus) ? (Ntraj - (ngpus-1)*batch_size) : batch_size

                results[i] = remotecall_fetch(w) do
                    solve(ensemble_prob, GPUEM(), EnsembleGPUKernel(CUDA.CUDABackend()),
                          dt=dt, trajectories=traj_count,
                          saveat=saveat, adaptive=false)
                end
            end
        end
    end

    return results
end

# Method 3: Local problem (no serialization - uses pre-created problem on worker)
function solve_local_problem(Ntraj, ngpus; dt, saveat)
    batch_size = cld(Ntraj, ngpus)
    results = Vector{Any}(undef, ngpus)

    @sync begin
        for (i, w) in enumerate(workers()[1:ngpus])
            @async begin
                traj_count = (i == ngpus) ? (Ntraj - (ngpus-1)*batch_size) : batch_size

                # Only send integers, use LOCAL_ensemble on worker
                results[i] = remotecall_fetch(w, traj_count, dt, saveat) do tc, d, s
                    solve(LOCAL_ensemble, GPUEM(), EnsembleGPUKernel(CUDA.CUDABackend()),
                          dt=d, trajectories=tc, saveat=s, adaptive=false)
                end
            end
        end
    end

    return results
end

# === WARMUP ===
println("="^50)
println("WARMUP (JIT compilation on each worker)")
println("="^50)

u0 = @SVector [Float32(1.0)]
tspan = (0.0f0, 1.0f0)
prob = SDEProblem(f_GPU, g_GPU, u0, tspan)
ensemble_prob = EnsembleProblem(prob)

# Warmup each worker individually
for (i, w) in enumerate(workers())
    print("  Warming up worker $w (GPU $(i-1))... ")
    t = @elapsed remotecall_fetch(w) do
        solve(ensemble_prob, GPUEM(), EnsembleGPUKernel(CUDA.CUDABackend()),
              dt=0.001f0, trajectories=10, saveat=0.1f0, adaptive=false)
    end
    println("$(round(t, digits=2))s")
end

# Warmup main process too
print("  Warming up main process... ")
t = @elapsed solve(ensemble_prob, GPUEM(), EnsembleGPUKernel(CUDA.CUDABackend()),
                   dt=0.001f0, trajectories=100, saveat=0.1f0, adaptive=false)
println("$(round(t, digits=2))s")

println("\nWarmup complete\n")

# === BENCHMARK ===
println("="^50)
println("BENCHMARK: Single vs Multi-GPU approaches")
println("="^50)

# Test parameters
dt = 0.0001f0
saveat = 0.1f0
tspan = (0.0f0, 5.0f0)
prob = SDEProblem(f_GPU, g_GPU, u0, tspan)
ensemble_prob = EnsembleProblem(prob)

trajectory_counts = [100, 1_000, 10_000, 50_000, 100_000]

println("\nUsing $NUMGPUS GPUs\n")

for Ntraj in trajectory_counts
    println("-"^50)
    println("Ntraj = $Ntraj")

    # Single GPU baseline
    t_single = @elapsed solve(ensemble_prob, GPUEM(), EnsembleGPUKernel(CUDA.CUDABackend()),
                              dt=dt, trajectories=Ntraj, saveat=saveat, adaptive=false)

    # Method 1: DiffEqGPU pmap
    t_pmap = @elapsed solve_diffeqgpu(ensemble_prob, Ntraj, NUMGPUS; dt=dt, saveat=saveat)

    # Method 2: Manual parallel (sends ensemble_prob)
    t_manual = @elapsed solve_manual_parallel(ensemble_prob, Ntraj, NUMGPUS; dt=dt, saveat=saveat)

    # Method 3: Local problem (no serialization)
    t_local = @elapsed solve_local_problem(Ntraj, NUMGPUS; dt=dt, saveat=saveat)

    println("  Single GPU:      $(round(t_single, digits=3))s (baseline)")
    println("  DiffEqGPU pmap:  $(round(t_pmap, digits=3))s ($(round(t_single/t_pmap, digits=2))x)")
    println("  Manual @async:   $(round(t_manual, digits=3))s ($(round(t_single/t_manual, digits=2))x)")
    println("  Local problem:   $(round(t_local, digits=3))s ($(round(t_single/t_local, digits=2))x)")

    # Find best multi-GPU method
    best_multi = min(t_pmap, t_manual, t_local)
    if best_multi < t_single
        println("  >>> Multi-GPU wins! $(round(t_single/best_multi, digits=2))x speedup")
    else
        println("  >>> Single GPU still fastest")
    end
end

println("\n" * "="^50)
println("TEST COMPLETE")
println("="^50)
