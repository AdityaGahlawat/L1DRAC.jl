# GPU Profiling - Deep dive into multi-GPU overhead
# Uses BenchmarkTools + CUDA profiling for Ntraj = 10,000

# === SETUP ===
using Distributed

using CUDA
const NUMGPUS = length(CUDA.devices())
println("Found $NUMGPUS GPU(s)")

if NUMGPUS < 2
    error("Need at least 2 GPUs for this test")
end

if nprocs() == 1
    addprocs(NUMGPUS)
    println("Added $NUMGPUS workers")
end

@everywhere using CUDA
for (i, w) in enumerate(workers())
    remotecall_wait(w) do
        CUDA.device!(i - 1)
    end
end

@everywhere using DiffEqGPU, StochasticDiffEq, StaticArrays, BenchmarkTools
@everywhere CUDA.allowscalar(false)

@everywhere f_GPU(u, p, t) = -u
@everywhere g_GPU(u, p, t) = @SVector [Float32(0.1)]

# Pre-create problem on each worker
@everywhere begin
    const LOCAL_u0 = @SVector [Float32(1.0)]
    const LOCAL_tspan = (0.0f0, 5.0f0)
    const LOCAL_prob = SDEProblem(f_GPU, g_GPU, LOCAL_u0, LOCAL_tspan)
    const LOCAL_ensemble = EnsembleProblem(LOCAL_prob)
end

using BenchmarkTools
using Statistics  # for mean, std
using Dates       # for timestamp

println("Setup complete\n")

# === PARAMETERS ===
const Ntraj = 10_000
const dt = 0.0001f0
const saveat = 0.1f0

u0 = @SVector [Float32(1.0)]
tspan = (0.0f0, 5.0f0)
prob = SDEProblem(f_GPU, g_GPU, u0, tspan)
ensemble_prob = EnsembleProblem(prob)

# === WARMUP ===
println("="^60)
println("WARMUP")
println("="^60)

# Warmup all workers
for (i, w) in enumerate(workers())
    remotecall_fetch(w) do
        solve(LOCAL_ensemble, GPUEM(), EnsembleGPUKernel(CUDA.CUDABackend()),
              dt=0.001f0, trajectories=100, saveat=0.1f0, adaptive=false)
    end
end
# Warmup main
solve(ensemble_prob, GPUEM(), EnsembleGPUKernel(CUDA.CUDABackend()),
      dt=0.001f0, trajectories=100, saveat=0.1f0, adaptive=false)

println("Warmup complete\n")

# === BENCHMARKS ===
const NSAMPLES = 100  # Solid statistical confidence (σ/10 standard error)
println("="^60)
println("BENCHMARKS (Ntraj = $Ntraj, $NSAMPLES samples each)")
println("="^60)

# --- Single GPU ---
@info "[1/4] Benchmarking Single GPU (baseline)..."
function bench_single()
    CUDA.@sync solve(ensemble_prob, GPUEM(), EnsembleGPUKernel(CUDA.CUDABackend()),
                     dt=dt, trajectories=Ntraj, saveat=saveat, adaptive=false)
end
b_single = @benchmark $bench_single() samples=NSAMPLES
display(b_single)
@info "[1/4] Single GPU done ✓"

# --- DiffEqGPU pmap ---
@info "[2/4] Benchmarking DiffEqGPU pmap ($NUMGPUS GPUs)..."
batch_size_pmap = cld(Ntraj, NUMGPUS)
function bench_pmap()
    CUDA.@sync solve(ensemble_prob, GPUEM(), EnsembleGPUKernel(CUDA.CUDABackend()),
                     dt=dt, trajectories=Ntraj, batch_size=batch_size_pmap,
                     saveat=saveat, adaptive=false)
end
b_pmap = @benchmark $bench_pmap() samples=NSAMPLES
display(b_pmap)
@info "[2/4] DiffEqGPU pmap done ✓"

# --- Manual @async (sends problem) ---
@info "[3/4] Benchmarking Manual @async (sends problem)..."
function solve_manual()
    batch_size = cld(Ntraj, NUMGPUS)
    results = Vector{Any}(undef, NUMGPUS)
    @sync begin
        for (i, w) in enumerate(workers()[1:NUMGPUS])
            @async begin
                traj_count = (i == NUMGPUS) ? (Ntraj - (NUMGPUS-1)*batch_size) : batch_size
                results[i] = remotecall_fetch(w) do
                    solve(ensemble_prob, GPUEM(), EnsembleGPUKernel(CUDA.CUDABackend()),
                          dt=dt, trajectories=traj_count, saveat=saveat, adaptive=false)
                end
            end
        end
    end
    return results
end
b_manual = @benchmark $solve_manual() samples=NSAMPLES
display(b_manual)
@info "[3/4] Manual @async done ✓"

# --- Local problem (no serialization) ---
@info "[4/4] Benchmarking Local problem (no serialization)..."
function solve_local()
    batch_size = cld(Ntraj, NUMGPUS)
    results = Vector{Any}(undef, NUMGPUS)
    @sync begin
        for (i, w) in enumerate(workers()[1:NUMGPUS])
            @async begin
                traj_count = (i == NUMGPUS) ? (Ntraj - (NUMGPUS-1)*batch_size) : batch_size
                results[i] = remotecall_fetch(w, traj_count, dt, saveat) do tc, d, s
                    solve(LOCAL_ensemble, GPUEM(), EnsembleGPUKernel(CUDA.CUDABackend()),
                          dt=d, trajectories=tc, saveat=s, adaptive=false)
                end
            end
        end
    end
    return results
end
b_local = @benchmark $solve_local() samples=NSAMPLES
display(b_local)
@info "[4/4] Local problem done ✓"

# === MEMORY ANALYSIS ===
println("="^60)
println("MEMORY ANALYSIS")
println("="^60)

@info "Measuring allocations..."
println("\nSingle GPU allocations:")
@time solve(ensemble_prob, GPUEM(), EnsembleGPUKernel(CUDA.CUDABackend()),
            dt=dt, trajectories=Ntraj, saveat=saveat, adaptive=false);

println("\nManual @async (sends problem) allocations:")
@time solve_manual();

println("\nLocal problem (no serialization) allocations:")
@time solve_local();
@info "Memory analysis done ✓"

# === CUDA PROFILE ===
println("\n" * "="^60)
println("CUDA PROFILE (Single GPU)")
println("="^60)
@info "Running CUDA profiler..."
CUDA.@profile trace=true begin
    solve(ensemble_prob, GPUEM(), EnsembleGPUKernel(CUDA.CUDABackend()),
          dt=dt, trajectories=Ntraj, saveat=saveat, adaptive=false)
end
@info "CUDA profile done ✓"

println("\n" * "="^60)
println("SUMMARY")
println("="^60)
println("Single GPU median:     $(round(median(b_single.times)/1e6, digits=2)) ms")
println("DiffEqGPU pmap median: $(round(median(b_pmap.times)/1e6, digits=2)) ms")
println("Manual @async median:  $(round(median(b_manual.times)/1e6, digits=2)) ms")
println("Local problem median:  $(round(median(b_local.times)/1e6, digits=2)) ms")

# === SAVE RESULTS ===
results_file = "test/gpu_profile_results.txt"
open(results_file, "w") do f
    println(f, "GPU Profile Results")
    println(f, "=" ^ 60)
    println(f, "Date: $(Dates.now())")
    println(f, "Ntraj: $Ntraj")
    println(f, "Samples: $NSAMPLES")
    println(f, "GPUs: $NUMGPUS")
    println(f, "")
    println(f, "BENCHMARK RESULTS (times in ms)")
    println(f, "-" ^ 60)
    println(f, "")
    println(f, "Single GPU:")
    println(f, "  min:    $(round(minimum(b_single.times)/1e6, digits=3)) ms")
    println(f, "  median: $(round(median(b_single.times)/1e6, digits=3)) ms")
    println(f, "  mean:   $(round(mean(b_single.times)/1e6, digits=3)) ms")
    println(f, "  max:    $(round(maximum(b_single.times)/1e6, digits=3)) ms")
    println(f, "  std:    $(round(std(b_single.times)/1e6, digits=3)) ms")
    println(f, "")
    println(f, "DiffEqGPU pmap ($NUMGPUS GPUs):")
    println(f, "  min:    $(round(minimum(b_pmap.times)/1e6, digits=3)) ms")
    println(f, "  median: $(round(median(b_pmap.times)/1e6, digits=3)) ms")
    println(f, "  mean:   $(round(mean(b_pmap.times)/1e6, digits=3)) ms")
    println(f, "  max:    $(round(maximum(b_pmap.times)/1e6, digits=3)) ms")
    println(f, "  std:    $(round(std(b_pmap.times)/1e6, digits=3)) ms")
    println(f, "")
    println(f, "Manual @async ($NUMGPUS GPUs):")
    println(f, "  min:    $(round(minimum(b_manual.times)/1e6, digits=3)) ms")
    println(f, "  median: $(round(median(b_manual.times)/1e6, digits=3)) ms")
    println(f, "  mean:   $(round(mean(b_manual.times)/1e6, digits=3)) ms")
    println(f, "  max:    $(round(maximum(b_manual.times)/1e6, digits=3)) ms")
    println(f, "  std:    $(round(std(b_manual.times)/1e6, digits=3)) ms")
    println(f, "")
    println(f, "Local problem ($NUMGPUS GPUs):")
    println(f, "  min:    $(round(minimum(b_local.times)/1e6, digits=3)) ms")
    println(f, "  median: $(round(median(b_local.times)/1e6, digits=3)) ms")
    println(f, "  mean:   $(round(mean(b_local.times)/1e6, digits=3)) ms")
    println(f, "  max:    $(round(maximum(b_local.times)/1e6, digits=3)) ms")
    println(f, "  std:    $(round(std(b_local.times)/1e6, digits=3)) ms")
    println(f, "")
    println(f, "-" ^ 60)
    println(f, "SPEEDUP vs Single GPU (using median)")
    println(f, "-" ^ 60)
    t_single = median(b_single.times)
    println(f, "DiffEqGPU pmap: $(round(t_single/median(b_pmap.times), digits=2))x")
    println(f, "Manual @async:  $(round(t_single/median(b_manual.times), digits=2))x")
    println(f, "Local problem:  $(round(t_single/median(b_local.times), digits=2))x")
end
@info "Results saved to $results_file"

println("\n" * "="^60)
println("PROFILE COMPLETE")
println("="^60)
