####################################################################################
# L1-DRAC System Simulation (True System with L1 Adaptive Control)
#
# State vector: Z = [X, Xhat, Xfilter, Λhat] (size 3n + m)
# Adaptive law uses sample-and-hold dynamics embedded in drift (no callbacks).
# See Writeups/PieceWiseConstant.pdf for derivation.
####################################################################################

##### HELPER FUNCTIONS (CPU) #####

function _L1_drift!(dZ, Z, (true_system, L1params, Δₜ), t)
    @unpack n, m = getfield(true_system, :sys_dims)
    @unpack f, g, g_perp, dynamics_params = getfield(true_system, :nom_vec_fields)
    @unpack Λμ = getfield(true_system, :unc_vec_fields)
    @unpack ω, Tₛ, λₛ = L1params

    X = Z[1:n]
    Xhat = Z[n+1:2n]
    Xfilter = Z[2n+1:2n+m]
    Λhat = Z[2n+m+1:3n+m]

    # Controller
    gbar_t = hcat(g(t, X, dynamics_params), g_perp(t, X, dynamics_params))
    Θ_t = hcat(I(m), zeros(m, n-m)) * inv(gbar_t)
    Λhat_m = Θ_t * Λhat
    uₐ = m == 1 ? -only(Xfilter) : -Xfilter

    dXfilter = -ω*Xfilter + ω*Λhat_m
    dXhat = -λₛ*(Xhat - X) + f(t, X, dynamics_params) + g(t, X, dynamics_params)*uₐ + Λhat
    dX = f(t, X, dynamics_params) + g(t, X, dynamics_params)*uₐ + Λμ(t, X, dynamics_params)

    # Sample-and-hold for Λhat
    is_crossover = (floor(t/Tₛ) > floor((t - Δₜ)/Tₛ)) && (t >= Tₛ)
    if is_crossover
        Λhat_new = (λₛ / (1 - exp(λₛ * Tₛ))) * (Xhat - X)
        dΛhat = (Λhat_new - Λhat) / Δₜ
    else
        dΛhat = zero(Λhat)
    end

    dZ[1:n] = dX
    dZ[n+1:2n] = dXhat
    dZ[2n+1:2n+m] = dXfilter
    dZ[2n+m+1:3n+m] = dΛhat
end

function _L1_diffusion!(dZ, Z, (true_system, L1params, Δₜ), t)
    @unpack n, m, d = getfield(true_system, :sys_dims)
    @unpack p, dynamics_params = getfield(true_system, :nom_vec_fields)
    @unpack Λσ = getfield(true_system, :unc_vec_fields)

    X = Z[1:n]
    dX = p(t, X, dynamics_params) + Λσ(t, X, dynamics_params)

    for j in 1:d
        dZ[1:n, j] = dX[:, j]
        dZ[n+1:2n, j] .= 0.0
        dZ[2n+1:2n+m, j] .= 0.0
        dZ[2n+m+1:3n+m, j] .= 0.0
    end
end

##### METHODS #####

# Method 1: CPU (single trajectory + ensemble)
function system_simulation(simulation_parameters::SimParams, true_system::TrueSystem, L1params::L1DRACParams, ::CPU; kwargs...)
    prog_steps = 1000
    @unpack tspan, Δₜ, Ntraj, Δ_saveat = simulation_parameters
    @unpack n, m, d = getfield(true_system, :sys_dims)
    @unpack true_ξ₀ = getfield(true_system, :init_dists)
    true_init = rand(true_ξ₀)

    if haskey(kwargs, :simtype) && kwargs[:simtype] == :ensemble
        L1_init = vcat(true_init, true_init, zeros(m), zeros(n))
        L1_problem = SDEProblem(_L1_drift!, _L1_diffusion!, L1_init, tspan,
                                noise_rate_prototype = zeros(3n+m, d),
                                (true_system, L1params, Δₜ))
        solver = EM()
        ensemble_alg = EnsembleThreads()
        function L1_prob_func_cpu(prob, i, repeat)
            rand_init = rand(true_ξ₀)
            remake(prob, u0 = vcat(rand_init, rand_init, zeros(m), zeros(n)))
        end
        ensemble_L1_problem = EnsembleProblem(L1_problem, prob_func = L1_prob_func_cpu)
        @info "Running Ensemble Simulation of L1-DRAC System (CPU)"
        L1_sol = solve(ensemble_L1_problem, solver, ensemble_alg, dt=Δₜ,
                      trajectories = Ntraj, progress = true, progress_steps = prog_steps,
                      saveat = Δ_saveat)
    else
        # Single trajectory
        L1_init = vcat(true_init, true_init, zeros(m), zeros(n))
        L1_problem = SDEProblem(_L1_drift!, _L1_diffusion!, L1_init, tspan,
                                noise_rate_prototype = zeros(3n+m, d),
                                (true_system, L1params, Δₜ))
        @info "Running Single Trajectory Simulation of L1-DRAC System"
        L1_sol = solve(L1_problem, EM(), dt=Δₜ, progress = true, progress_steps = prog_steps,
                      saveat = Δ_saveat)
    end
    @info "Done"
    return L1_sol
end

# Method 2: GPU - dispatches to inner private methods for single/multi GPU
function system_simulation(simulation_parameters::SimParams, true_system::TrueSystem, L1params::L1DRACParams, gpu::GPU; kwargs...)
    @unpack n, m, d = getfield(true_system, :sys_dims)
    if gpu.numGPUs == 1
        _system_simulation_L1_gpu(simulation_parameters, true_system, L1params, Val(n), Val(m), Val(d))
    else
        _system_simulation_L1_gpu(simulation_parameters, true_system, L1params, Val(n), Val(m), Val(d), gpu.numGPUs)
    end
end

##### INNER PRIVATE METHODS (GPU) #####

# Inner Private Method 1: Single GPU (Val pattern for compile-time dimensions)
function _system_simulation_L1_gpu(simulation_parameters, true_system::TrueSystem, L1params::L1DRACParams,
                                   ::Val{n_gpu}, ::Val{m_gpu}, ::Val{d_gpu}) where {n_gpu, m_gpu, d_gpu}

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
    # IMPORTANT: Variable names inside closures must NOT shadow outer scope variables.
    function drift_L1_gpu(Z, params, t)
        dynamics_params_gpu, ω_gpu, Tₛ_gpu, λₛ_gpu, Δₜ_gpu = params

        X = Z[SOneTo(n_gpu)]
        Xhat = Z[StaticArrays.SUnitRange(n_gpu+1, 2n_gpu)]
        Xfilter = Z[StaticArrays.SUnitRange(2n_gpu+1, 2n_gpu+m_gpu)]
        Λhat = Z[StaticArrays.SUnitRange(2n_gpu+m_gpu+1, 3n_gpu+m_gpu)]

        # Controller: Θ(t) = [I_m, 0] * inv([g(t), g_perp(t)])
        gbar_t = hcat(g(t, X, dynamics_params_gpu), g_perp(t, X, dynamics_params_gpu))
        Θ_t = hcat(SMatrix{m_gpu, m_gpu}(I), @SMatrix(zeros(eltype(Z), m_gpu, n_gpu-m_gpu))) * inv(gbar_t)
        Λhat_m = Θ_t * Λhat
        uₐ = m_gpu == 1 ? -only(Xfilter) : -Xfilter

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

        dX_noise = SMatrix{n_gpu, d_gpu}(p(t, X, dynamics_params_gpu) + Λσ(t, X, dynamics_params_gpu))

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

    total_state_size = 3 * n_gpu + m_gpu

    L1_problem = SDEProblem(drift_L1_gpu, diffusion_L1_gpu, u0, Float32.(tspan), L1_params_tuple,
                            noise_rate_prototype = SMatrix{total_state_size, d_gpu}(zeros(Float32, total_state_size, d_gpu)))

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

# Inner Private Method 2: Multi-GPU (Val pattern + batch_size distribution)
function _system_simulation_L1_gpu(simulation_parameters, true_system::TrueSystem, L1params::L1DRACParams,
                                   ::Val{n_gpu}, ::Val{m_gpu}, ::Val{d_gpu}, numGPUs::Int) where {n_gpu, m_gpu, d_gpu}

    prog_steps = 1000
    @unpack tspan, Δₜ, Ntraj, Δ_saveat = simulation_parameters
    @unpack true_ξ₀ = getfield(true_system, :init_dists)
    @unpack f, g, g_perp, p, dynamics_params = getfield(true_system, :nom_vec_fields)
    @unpack Λμ, Λσ = getfield(true_system, :unc_vec_fields)
    @unpack ω, Tₛ, λₛ = L1params

    L1_params_tuple = (dynamics_params, Float32(ω), Float32(Tₛ), Float32(λₛ), Float32(Δₜ))

    function drift_L1_gpu(Z, params, t)
        dynamics_params_gpu, ω_gpu, Tₛ_gpu, λₛ_gpu, Δₜ_gpu = params

        X = Z[SOneTo(n_gpu)]
        Xhat = Z[StaticArrays.SUnitRange(n_gpu+1, 2n_gpu)]
        Xfilter = Z[StaticArrays.SUnitRange(2n_gpu+1, 2n_gpu+m_gpu)]
        Λhat = Z[StaticArrays.SUnitRange(2n_gpu+m_gpu+1, 3n_gpu+m_gpu)]

        gbar_t = hcat(g(t, X, dynamics_params_gpu), g_perp(t, X, dynamics_params_gpu))
        Θ_t = hcat(SMatrix{m_gpu, m_gpu}(I), @SMatrix(zeros(eltype(Z), m_gpu, n_gpu-m_gpu))) * inv(gbar_t)
        Λhat_m = Θ_t * Λhat
        uₐ = m_gpu == 1 ? -only(Xfilter) : -Xfilter

        dXfilter = -ω_gpu * Xfilter + ω_gpu * Λhat_m
        dXhat = -λₛ_gpu * (Xhat - X) + f(t, X, dynamics_params_gpu) + g(t, X, dynamics_params_gpu) * uₐ + Λhat
        dX = f(t, X, dynamics_params_gpu) + g(t, X, dynamics_params_gpu) * uₐ + Λμ(t, X, dynamics_params_gpu)

        is_crossover = (floor(t / Tₛ_gpu) > floor((t - Δₜ_gpu) / Tₛ_gpu)) && (t >= Tₛ_gpu)
        if is_crossover
            Λhat_new = (λₛ_gpu / (1 - exp(λₛ_gpu * Tₛ_gpu))) * (Xhat - X)
            dΛhat = (Λhat_new - Λhat) / Δₜ_gpu
        else
            dΛhat = zero(Λhat)
        end

        return vcat(dX, dXhat, dXfilter, dΛhat)
    end

    function diffusion_L1_gpu(Z, params, t)
        dynamics_params_gpu = params[1]
        X = Z[SOneTo(n_gpu)]

        dX_noise = SMatrix{n_gpu, d_gpu}(p(t, X, dynamics_params_gpu) + Λσ(t, X, dynamics_params_gpu))

        return vcat(dX_noise,
                    @SMatrix(zeros(eltype(Z), n_gpu, d_gpu)),
                    @SMatrix(zeros(eltype(Z), m_gpu, d_gpu)),
                    @SMatrix(zeros(eltype(Z), n_gpu, d_gpu)))
    end

    true_init = rand(true_ξ₀)
    u0 = vcat(SVector{n_gpu}(Float32.(true_init)),
              SVector{n_gpu}(Float32.(true_init)),
              SVector{m_gpu}(zeros(Float32, m_gpu)),
              SVector{n_gpu}(zeros(Float32, n_gpu)))

    total_state_size = 3 * n_gpu + m_gpu

    L1_problem = SDEProblem(drift_L1_gpu, diffusion_L1_gpu, u0, Float32.(tspan), L1_params_tuple,
                            noise_rate_prototype = SMatrix{total_state_size, d_gpu}(zeros(Float32, total_state_size, d_gpu)))

    function L1_prob_func(prob, i, repeat)
        rand_init = SVector{n_gpu}(Float32.(rand(true_ξ₀)))
        new_u0 = vcat(rand_init, rand_init,
                      SVector{m_gpu}(zeros(Float32, m_gpu)),
                      SVector{n_gpu}(zeros(Float32, n_gpu)))
        remake(prob, u0 = new_u0)
    end
    ensemble_L1_problem = EnsembleProblem(L1_problem, prob_func = L1_prob_func)

    batch_size = cld(Ntraj, numGPUs)
    @info "Running Ensemble Simulation of L1-DRAC System on $numGPUs GPUs with batch size of $batch_size per GPU"
    L1_sol = solve(ensemble_L1_problem, GPUEM(), DiffEqGPU.EnsembleGPUKernel(CUDA.CUDABackend()),
                   dt=Float32(Δₜ), trajectories=Ntraj, batch_size=batch_size,
                   progress=true, progress_steps=prog_steps,
                   saveat=Float32(Δ_saveat), adaptive=false)
    @info "Done"
    return L1_sol
end
