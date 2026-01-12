# test/NestedFunctionGPU.jl
# Minimal test to verify GPU compilation theory:
# - Runtime dimensions (n::Int64) cause "dynamic function invocation" error
# - Compile-time dimensions (Val{N}) work correctly
# - Nested function calls work fine when types are stable

using CUDA
using DiffEqGPU
using StochasticDiffEq
using StaticArrays
using Distributions
using LinearAlgebra

# Struct for Test 3 (must be at module scope)
struct SimDims
    n::Int
    d::Int
end

# ============================================================================
# TEST 1: GPU with parameters passed through p (like OLD code)
# ============================================================================

function test_nested_calls(; Ntraj=10)
    @info "TEST 1: GPU with parameters through p"
    
    # Drift - parameters from p, no captured globals
    function my_drift(X, p, t)
        α = p[1]
        ω = p[2]
        d1 = -α * X[1] + sin(ω * t) * X[2]
        d2 = -α * X[2]
        return @SVector [d1, d2]
    end

    # Diffusion - parameters from p
    function my_diffusion(X, p, t)
        σ = p[3]
        return @SMatrix [σ 0.0f0; 0.0f0 σ]
    end


    u0 = @SVector [1.0f0, 0.5f0]
    tspan = (0.0f0, 1.0f0)
    p = @SVector [0.5f0, 2.0f0, 0.1f0]  # α, ω, σ

    prob = SDEProblem(my_drift, my_diffusion, u0, tspan, p,
                      noise_rate_prototype = @SMatrix [0.0f0 0.0f0; 0.0f0 0.0f0])
    ensemble_prob = EnsembleProblem(prob)

    sol = solve(ensemble_prob, GPUEM(), EnsembleGPUKernel(CUDA.CUDABackend()),
                dt=0.01f0, trajectories=Ntraj, saveat=0.1f0, adaptive=false)

    @info "✓ TEST 1 PASSED" trajectories=length(sol)
    return true
end

# ============================================================================
# TEST 2: Runtime n passed through p - does SVector{n} work?
# ============================================================================

function test_hardcoded_n(; Ntraj=10)
    @info "TEST 2: Hardcoded n (should work on GPU)"

    # Diffusion with hardcoded dimension
    function diffusion(X, p, t)
        return @SMatrix [0.1f0 0.0f0; 0.0f0 0.1f0]
    end

    # Drift with hardcoded 2 instead of runtime n
    function drift_hardcoded(X, p, t)
        return SVector{2}(ntuple(i -> -0.5f0 * X[i], 2))
    end

    u0 = @SVector [1.0f0, 0.5f0]
    tspan = (0.0f0, 1.0f0)

    prob = SDEProblem(drift_hardcoded, diffusion, u0, tspan, nothing,
                      noise_rate_prototype = @SMatrix [0.0f0 0.0f0; 0.0f0 0.0f0])
    ensemble_prob = EnsembleProblem(prob)

    @info "  Testing hardcoded n=2 in SVector{2}..."
    sol = solve(ensemble_prob, GPUEM(), EnsembleGPUKernel(CUDA.CUDABackend()),
                dt=0.01f0, trajectories=Ntraj, saveat=0.1f0, adaptive=false)
    @info "  ✓ TEST 2 PASSED" trajectories=length(sol)

    return true
end

# ============================================================================
# TEST 3: Package pattern - bridging runtime struct to compile-time Val
# ============================================================================

function test_package_pattern(; Ntraj=10)
    @info "TEST 3: Package pattern (struct dims → Val{N})"

    dims = SimDims(2, 2)

    # Matrices defined locally, will be passed through p
    A_mat = @SMatrix Float32[-0.5 0.1; 0.0 -0.3]
    σ_mat = @SMatrix Float32[0.1 0.0; 0.0 0.2]

    # User dynamics take parameters from params
    # params[1] = A_mat, params[2] = σ_mat
    user_f(t, x, params) = params[1] * x
    user_g(t, x, params) = params[2]

    # Package function: bridge runtime dims to compile-time
    function setup_gpu_problem(f, g, dims::SimDims, init_dist, params)
        _setup_gpu_problem(f, g, Val(dims.n), Val(dims.d), init_dist, params)
    end

    function _setup_gpu_problem(f, g, ::Val{N}, ::Val{D}, init_dist, params) where {N, D}
        drift_gpu(X, params, t) = f(t, X, params)
        diffusion_gpu(X, params, t) = g(t, X, params)

        u0 = SVector{N}(Float32.(rand(init_dist)))
        tspan = (0.0f0, 1.0f0)

        prob = SDEProblem(drift_gpu, diffusion_gpu, u0, tspan, params,
                          noise_rate_prototype = SMatrix{N,D}(zeros(Float32, N, D)))

        function prob_func(prob, i, repeat)
            remake(prob, u0 = SVector{N}(Float32.(rand(init_dist))))
        end

        return EnsembleProblem(prob, prob_func=prob_func)
    end

    # Run it
    init_dist = MvNormal(zeros(dims.n), 0.1 * I(dims.n))
    params = (A_mat, σ_mat)
    ensemble_prob = setup_gpu_problem(user_f, user_g, dims, init_dist, params)

    sol = solve(ensemble_prob, GPUEM(), EnsembleGPUKernel(CUDA.CUDABackend()),
                dt=0.01f0, trajectories=Ntraj, saveat=0.1f0, adaptive=false)

    @info "✓ TEST 3 PASSED" trajectories=length(sol)
    return true
end

# # ============================================================================
# # RUN ALL
# # ============================================================================

# function run_all_tests()
#     println("=" ^ 60)
#     println("GPU Compilation Tests - Val{N} Pattern")
#     println("=" ^ 60)
#     println()

#     test_nested_calls()
#     println()

#     test_val_vs_runtime()
#     println()

#     test_package_pattern()
#     println()

#     println("=" ^ 60)
#     println("All tests completed")
#     println("=" ^ 60)
# end
