####################################################################################
# GPU L1-DRAC Simulation Functions (EnsembleGPUKernel)
# - METHOD 3: L1-DRAC System (ensemble only)
#
# State vector: Z = [X, Xhat, Xfilter, Λhat] (size 3n + m)
# Adaptive law uses sample-and-hold dynamics embedded in drift (no callbacks).
# See Writeups/PieceWiseConstant.pdf for derivation.
#
# Note: Drift/diffusion functions are defined locally inside each method to avoid
# passing structs with functions as parameters (GPU requires isbits parameters).
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

##### METHOD 3: L1-DRAC closed-loop simulation (GPU) #####

# Outer function: dispatches on L1DRACParams, bridges runtime dims to compile-time
function system_simulation(simulation_parameters::SimParams, true_system::TrueSystem, L1params::L1DRACParams, ::GPU; kwargs...)
    # converts n, m, d to FIXED n_gpu, m_gpu, d_gpu via Val pattern for GPU compatibility
    @unpack n, m, d = getfield(true_system, :sys_dims)
    _system_simulation(simulation_parameters, true_system, L1params, Val(n), Val(m), Val(d))
end

# Inner function: n_gpu/m_gpu/d_gpu are compile-time constants via `where` clause
function _system_simulation(simulation_parameters, true_system::TrueSystem, L1params::L1DRACParams,
                            ::Val{n_gpu}, ::Val{m_gpu}, ::Val{d_gpu}) where {n_gpu, m_gpu, d_gpu}

    # (n_gpu, m_gpu, d_gpu) = (n, m, d) as compile-time constants

    prog_steps = 1000
    @unpack tspan, Δₜ, Ntraj, Δ_saveat = simulation_parameters
    @unpack true_ξ₀ = getfield(true_system, :init_dists)
    @unpack f, g, g_perp, p, dynamics_params = getfield(true_system, :nom_vec_fields)
    @unpack Λμ, Λσ = getfield(true_system, :unc_vec_fields)
    @unpack ω, Tₛ, λₛ = L1params

    # Build params tuple for SDEProblem (only isbits data, Float32 for GPU)
    L1_params_tuple = (dynamics_params, Float32(ω), Float32(Tₛ), Float32(λₛ), Float32(Δₜ))

    # L1-DRAC drift wrapper: Z = [X, Xhat, Xfilter, Λhat]
    # NOTE: Must use SOneTo/SUnitRange for indexing SVectors to preserve static typing.
    # Regular Z[1:n] returns Vector (not SVector), breaking GPU compatibility.
    #
    # IMPORTANT: Variable names inside closures must NOT shadow outer scope variables.
    # Julia's closure capture mechanism grabs outer variables when inner ones shadow by name,
    # breaking isbits requirement for GPU. Use unique names (e.g., dynamics_params_gpu, ω_gpu)
    # to avoid capturing the outer scope's dynamics_params, ω, Tₛ, λₛ, Δₜ.
    function drift_L1_gpu(Z, params, t)
        dynamics_params_gpu, ω_gpu, Tₛ_gpu, λₛ_gpu, Δₜ_gpu = params

        # Unpack state vector using compile-time dimensions
        X = Z[SOneTo(n_gpu)]
        Xhat = Z[StaticArrays.SUnitRange(n_gpu+1, 2n_gpu)]
        Xfilter = Z[StaticArrays.SUnitRange(2n_gpu+1, 2n_gpu+m_gpu)]
        Λhat = Z[StaticArrays.SUnitRange(2n_gpu+m_gpu+1, 3n_gpu+m_gpu)]

        # Controller: Θ(t) = [I_m, 0] * inv([g(t), g_perp(t)])
        gbar_t = hcat(g(t, X, dynamics_params_gpu), g_perp(t, X, dynamics_params_gpu))
        Θ_t = hcat(SMatrix{m_gpu, m_gpu}(I), @SMatrix(zeros(eltype(Z), m_gpu, n_gpu-m_gpu))) * inv(gbar_t)
        Λhat_m = Θ_t * Λhat
        uₐ = m_gpu == 1 ? -only(Xfilter) : -Xfilter

        # Derivatives
        dXfilter = -ω_gpu * Xfilter + ω_gpu * Λhat_m
        dXhat = -λₛ_gpu * (Xhat - X) + f(t, X, dynamics_params_gpu) + g(t, X, dynamics_params_gpu) * uₐ + Λhat
        dX = f(t, X, dynamics_params_gpu) + g(t, X, dynamics_params_gpu) * uₐ + Λμ(t, X, dynamics_params_gpu)

        # Sample-and-hold for Λhat update
        is_crossover = (floor(t / Tₛ_gpu) > floor((t - Δₜ_gpu) / Tₛ_gpu)) && (t >= Tₛ_gpu)
        if is_crossover
            Λhat_new = (λₛ_gpu / (1 - exp(λₛ_gpu * Tₛ_gpu))) * (Xhat - X)
            dΛhat = (Λhat_new - Λhat) / Δₜ_gpu
        else
            dΛhat = zero(Λhat)
        end

        return vcat(dX, dXhat, dXfilter, dΛhat)
    end

    # L1-DRAC diffusion wrapper: only X component has noise
    function diffusion_L1_gpu(Z, params, t)
        dynamics_params_gpu = params[1]
        X = Z[SOneTo(n_gpu)]

        # Only X has diffusion: p(t,X) + Λσ(t,X)
        dX_noise = SMatrix{n_gpu, d_gpu}(p(t, X, dynamics_params_gpu) + Λσ(t, X, dynamics_params_gpu))

        # Pad with zeros for Xhat, Xfilter, Λhat
        return vcat(dX_noise,
                    @SMatrix(zeros(eltype(Z), n_gpu, d_gpu)),
                    @SMatrix(zeros(eltype(Z), m_gpu, d_gpu)),
                    @SMatrix(zeros(eltype(Z), n_gpu, d_gpu)))
    end

    # Initial condition: [X₀, X₀, 0, 0] with compile-time dimensions
    true_init = rand(true_ξ₀)
    u0 = vcat(SVector{n_gpu}(Float32.(true_init)),
              SVector{n_gpu}(Float32.(true_init)),
              SVector{m_gpu}(zeros(Float32, m_gpu)),
              SVector{n_gpu}(zeros(Float32, n_gpu)))

    # Total state size for noise_rate_prototype
    total_state_size = 3 * n_gpu + m_gpu

    # SDEProblem: L1_params_tuple passed as p argument
    L1_problem = SDEProblem(drift_L1_gpu, diffusion_L1_gpu, u0, Float32.(tspan), L1_params_tuple,
                            noise_rate_prototype = SMatrix{total_state_size, d_gpu}(zeros(Float32, total_state_size, d_gpu)))

    # prob_func for ensemble: uses compile-time n_gpu, m_gpu for SVector
    function L1_prob_func(prob, i, repeat)
        rand_init = SVector{n_gpu}(Float32.(rand(true_ξ₀)))
        new_u0 = vcat(rand_init, rand_init,
                      SVector{m_gpu}(zeros(Float32, m_gpu)),
                      SVector{n_gpu}(zeros(Float32, n_gpu)))
        remake(prob, u0 = new_u0)
    end
    ensemble_L1_problem = EnsembleProblem(L1_problem, prob_func = L1_prob_func)

    @info "Running Ensemble Simulation of L1-DRAC System (GPU)"
    @CUDA.time L1_sol = solve(ensemble_L1_problem, GPUEM(), DiffEqGPU.EnsembleGPUKernel(CUDA.CUDABackend()),
                              dt=Float32(Δₜ), trajectories=Ntraj, progress=true, progress_steps=prog_steps,
                              saveat=Float32(Δ_saveat), adaptive=false)
    @info "Done"
    return L1_sol
end
