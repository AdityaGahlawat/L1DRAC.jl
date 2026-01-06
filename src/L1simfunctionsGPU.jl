####################################################################################
# GPU L1-DRAC Simulation Functions (EnsembleGPUKernel)
# - METHOD 3: L1-DRAC System (ensemble only)
#
# State vector: Z = [X, Xhat, Xfilter, Λhat] (size 3n + m)
# Adaptive law uses sample-and-hold dynamics embedded in drift (no callbacks).
# See Writeups/PieceWiseConstant.pdf for derivation.
####################################################################################

# Out-of-place versions (GPU) - EnsembleGPUKernel requires returning values
# Type-generic: preserves Float32/Float64 based on input Z
function _L1_drift_oop(Z, (true_system, L1params, Δₜ), t)
    @unpack n, m = getfield(true_system, :sys_dims)
    @unpack f, g, g_perp = getfield(true_system, :nom_vec_fields)
    @unpack Λμ = getfield(true_system, :unc_vec_fields)
    @unpack ω, Tₛ, λₛ = L1params

    X = Z[SOneTo(n)]
    Xhat = Z[n+1:2n]
    Xfilter = Z[2n+1:2n+m]
    Λhat = Z[2n+m+1:3n+m]

    # Controller
    gbar_t = hcat(g(t), g_perp(t))
    Θ_t = hcat(I(m), zeros(eltype(Z), m, n-m)) * inv(gbar_t)
    Λhat_m = Θ_t * Λhat
    uₐ = m == 1 ? -only(Xfilter) : -Xfilter

    dXfilter = -ω*Xfilter + ω*Λhat_m
    dXhat = -λₛ*(Xhat - X) + f(t, X) + g(t)*uₐ + Λhat
    dX = f(t, X) + g(t)*uₐ + Λμ(t, X)

    # Sample-and-hold for Λhat
    is_crossover = (floor(t/Tₛ) > floor((t - Δₜ)/Tₛ)) && (t >= Tₛ)
    if is_crossover
        Λhat_new = (λₛ / (1 - exp(λₛ * Tₛ))) * (Xhat - X)
        dΛhat = (Λhat_new - Λhat) / Δₜ
    else
        dΛhat = zero(Λhat)
    end

    return vcat(SVector{n}(dX...), SVector{n}(dXhat...),
                SVector{m}(dXfilter...), SVector{n}(dΛhat...))
end

function _L1_diffusion_oop(Z, (true_system, L1params, Δₜ), t)
    @unpack n, m, d = getfield(true_system, :sys_dims)
    @unpack p = getfield(true_system, :nom_vec_fields)
    @unpack Λσ = getfield(true_system, :unc_vec_fields)

    X = Z[SOneTo(n)]
    dX = SMatrix{n,d}(p(t, X) + Λσ(t, X))

    return vcat(dX, @SMatrix(zeros(eltype(Z), n, d)), @SMatrix(zeros(eltype(Z), m, d)), @SMatrix(zeros(eltype(Z), n, d)))
end

####################################################################################
# METHOD 3: L1-DRAC closed-loop simulation
function system_simulation(simulation_parameters::SimParams, true_system::TrueSystem, L1params::L1DRACParams, ::GPU; kwargs...)
    prog_steps = 1000
    @unpack tspan, Δₜ, Ntraj, Δ_saveat = simulation_parameters
    @unpack n, m, d = getfield(true_system, :sys_dims)
    @unpack true_ξ₀ = getfield(true_system, :init_dists)
    true_init = rand(true_ξ₀)

    if haskey(kwargs, :simtype) && kwargs[:simtype] == :ensemble
        backend = get(kwargs, :backend, :cpu)

        if backend == :cpu
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
            @info "Running Ensemble Simulation of L1 System" backend=backend
            L1_sol = solve(ensemble_L1_problem, solver, ensemble_alg, dt=Δₜ,
                          trajectories = Ntraj, progress = true, progress_steps = prog_steps,
                          saveat = Δ_saveat)

        elseif backend == :gpu
            L1_init = vcat(SVector{n}(Float32.(true_init)...), SVector{n}(Float32.(true_init)...),
                          SVector{m}(zeros(Float32, m)...), SVector{n}(zeros(Float32, n)...))
            L1_problem = SDEProblem(_L1_drift_oop, _L1_diffusion_oop, L1_init, Float32.(tspan),
                                    (true_system, L1params, Float32(Δₜ)))
            solver = GPUEM()
            ensemble_alg = DiffEqGPU.EnsembleGPUKernel(CUDA.CUDABackend())
            function L1_prob_func_gpu(prob, i, repeat)
                rand_init = Float32.(rand(true_ξ₀))
                remake(prob, u0 = vcat(SVector{n}(rand_init...), SVector{n}(rand_init...),
                                       SVector{m}(zeros(Float32, m)...), SVector{n}(zeros(Float32, n)...)))
            end
            ensemble_L1_problem = EnsembleProblem(L1_problem, prob_func = L1_prob_func_gpu)
            @info "Running Ensemble Simulation of L1 System" backend=backend
            L1_sol = solve(ensemble_L1_problem, solver, ensemble_alg, dt=Float32(Δₜ),
                          trajectories = Ntraj, saveat = Float32(Δ_saveat), adaptive = false)

        else
            error("Unknown backend: $backend. Supported: :cpu, :gpu")
        end
    else
        # Single trajectory
        L1_init = vcat(true_init, true_init, zeros(m), zeros(n))
        L1_problem = SDEProblem(_L1_drift!, _L1_diffusion!, L1_init, tspan,
                                noise_rate_prototype = zeros(3n+m, d),
                                (true_system, L1params, Δₜ))
        @info "Running Single Trajectory Simulation of L1 System"
        L1_sol = solve(L1_problem, EM(), dt=Δₜ, progress = true, progress_steps = prog_steps,
                      saveat = Δ_saveat)
    end
    @info "Done"
    return L1_sol
end
