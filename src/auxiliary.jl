####################################################################################
# Auxiliary Functions (Backend Selection & Cleanup)
#
# Usage:
#   using CUDA, L1DRAC
#   backend = get_backend(numGPUs)  # GPU(N) for N GPUs, CPU() for 0
#   # ... run simulations ...
#   cleanup_environment(backend)
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

"""
    get_numGPUs(max_GPUs::Int) -> Int

Converts user's max_GPUs request to actual numGPUs.
Returns min(max_GPUs, length(CUDA.devices())).

- max_GPUs = 0 → returns 0 (CPU mode)
- max_GPUs > available → caps to available, logs warning
- max_GPUs ≤ available → returns max_GPUs
"""
function get_numGPUs(max_GPUs::Int)
    @assert max_GPUs >= 0 "max_GPUs must be non-negative"

    if max_GPUs == 0
        @info "CPU mode requested"
        return 0
    end

    available_GPUs = Int(length(CUDA.devices()))

    if max_GPUs > available_GPUs
        @warn "Requested $max_GPUs GPUs but only $available_GPUs available. Using $available_GPUs."
        return available_GPUs
    else
        @info "GPU mode" max_GPUs=max_GPUs available_GPUs=available_GPUs
        return max_GPUs
    end
end

# Cleanup methods (multiple dispatch based on backend)
function cleanup_environment(::CPU)
    GC.gc()
    @info "Memory reclaimed (CPU)"
end

function cleanup_environment(::GPU)
    GC.gc()
    CUDA.reclaim()
    @info "GPU memory reclaimed"
end

#= OLD Distributed.jl functions (removed in @async refactor - see v1.1.0-GPU-parallel-PMAP)

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

# Set up GPU workers for multi-GPU ensemble simulations
function setup_gpu_workers(; max_GPUs = 1)
    cleanup_gpu_environment()
    @assert max_GPUs isa Integer && max_GPUs >= 0 "max_GPUs must be a non-negative integer"
    available_GPUs = length(CUDA.devices())
    numGPUs = min(max_GPUs, available_GPUs)
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
=#
