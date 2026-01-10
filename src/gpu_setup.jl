####################################################################################
# GPU Setup and Cleanup Functions
#
# Usage:
#   using Distributed, CUDA, L1DRAC
#   numGPUs = setup_gpu_workers(; max_GPUs = 2)
#   @everywhere using CUDA, L1DRAC, ...  # User loads packages on workers
#   assign_gpus_to_workers()              # Assign GPUs to workers
#   # ... run simulations ...
#   cleanup_gpu_environment()
####################################################################################

#=
GPU Compilation Requirements:
- StaticArrays (SVector, SMatrix) need dimensions known at compile-time
- SVector{n} where n is a runtime Int causes "dynamic function invocation" error
- SVector{N} where N is a type parameter (via `where {N}`) works

Solution: Val Pattern (standard Julia idiom for runtime → compile-time bridging)
- Outer function: handles multiple dispatch, extracts runtime dimensions, converts to Val
- Inner function: receives Val{N}, Val{M}, Val{D}, extracts compile-time N, M, D via `where` clause

Additionally:
- dynamics_params (NamedTuple with matrices) must be passed through SDEProblem's p argument
- User's f(t, x, params) receives dynamics_params explicitly, not via closure (closures break GPU)
=#


# Returns the appropriate backend based on numGPUs
function get_backend(numGPUs::Int)
    if numGPUs == 0
        return CPU()
    else
        return GPU(numGPUs)
    end
end

# Assign each worker to a different GPU (call after @everywhere using CUDA)
function assign_gpus_to_workers()
    numGPUs = min(length(workers()), length(CUDA.devices()))
    for (i, w) in enumerate(workers()[1:numGPUs])
        remotecall_wait(w) do
            CUDA.device!(i - 1)
        end
    end
    @info "GPUs assigned" Devices=join(["Worker $w → CuDevice($(i-1))" for (i, w) in enumerate(workers()[1:numGPUs])], ", ")
end

# Remove any existing Distributed workers and free GPU memory
function cleanup_gpu_environment()
    @info "Removing Stale Workers and Shedding Memory"
    if nprocs() > 1
        rmprocs(workers())
    end
    GC.gc()
    CUDA.reclaim()
end

# Set up GPU workers for multi-GPU ensemble simulations
# max_GPUs: 0 = CPU only, 1 = single GPU (default), N > 1 = multi-GPU
# Returns: numGPUs (actual number of GPUs being used)
function setup_gpu_workers(; max_GPUs = 1)

    cleanup_gpu_environment()

    @assert max_GPUs isa Integer && max_GPUs >= 0 "max_GPUs must be a non-negative integer"

    available_GPUs = length(CUDA.devices())
    numGPUs = min(max_GPUs, available_GPUs)

    # Setup Distributed workers for multi-GPU
    if numGPUs > 1 && nprocs() == 1
        addprocs(numGPUs)
        @info "Multi-GPU mode: $numGPUs workers spawned"
    elseif numGPUs == 1
        @info "Single GPU mode" Detected=available_GPUs Assigned=numGPUs Device="CuDevice(0): $(CUDA.name(CUDA.CuDevice(0)))"
    else
        @info "CPU only mode" Detected=available_GPUs Assigned=numGPUs Threads=Threads.nthreads()
    end

    return numGPUs
end
