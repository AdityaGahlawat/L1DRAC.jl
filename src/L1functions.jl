# L1-DRAC specific functions 
function _L1_drift!(dZ, Z, (true_system, L1params, Λhat), t; kwargs...)
    @unpack n, m = getfield(true_system, :sys_dims)
	@unpack f, g, g_perp = getfield(true_system, :nom_vec_fields)
    @unpack Λμ = getfield(true_system, :unc_vec_fields)
    @unpack λₛ, Tₛ = L1params
    # Need the following unrefined concatenation for StaticArrays and GPU compatibility
    X = Z[1:n] 
    Xhat = Z[n+1:2n]
    ## Placeholders
    m == 1 ? uₐ = 0.0 : uₐ = zeros(m)   
    ##########################
    # System
    dX = f(t, X) + g(t)*uₐ + Λμ(t,X)  
    # Adaptive Estimate 
    # Predictor
    if haskey(kwargs, :predictor_mode) && kwargs[:predictor_mode] == :test
        # ---Test Mode: Passing Λμ(t,X) to the Predictor---"
        dXhat = -λₛ*(Xhat-X) + f(t, X) + g(t)*uₐ + Λμ(t,X)
    else 
        dXhat = -λₛ*(Xhat-X) + f(t, X) + g(t)*uₐ + Λhat
    end
    dZ[1:n] = dX[1:n]
    dZ[n+1:2n] = dXhat[1:n]
end
function _L1_diffusion!(dZ, Z, (true_system, L1params), t; kwargs...)
    @unpack n, d = getfield(true_system, :sys_dims)
	@unpack p = getfield(true_system, :nom_vec_fields)
	@unpack Λσ = getfield(true_system, :unc_vec_fields)
    @unpack λₛ = L1params
    # Need the following unrefined concatenation for StaticArrays and GPU compatibility
    X = Z[1:n] 
    Xhat = Z[n+1:2n] 
    # System
    Fσ(t, X) = p(t,X) + Λσ(t,X)
    dX = Fσ(t, X) 
    # Predictor
    if haskey(kwargs, :predictor_mode) && kwargs[:predictor_mode] == :test
        # ---Test Mode: Passing Fσ(t, X) and dWₜ to the Predictor---
        dXhat = Fσ(t, X)
    else 
        dXhat = zeros(n,d) # Predictor is an ODE (drift only)
    end
    concat_diffusion = vcat(dX, dXhat)
    for i in 1:2n
		for j in 1:d
            dZ[i,j] = concat_diffusion[i,j]
		end
	end
end
####################################################################################
# METHOD 3: simulation of L1-DRAC closed-loop system
# Methods 1 and 2 are in \src/simfunctions.jl
function system_simulation(simulation_parameters::SimParams, true_system::TrueSystem, L1params::L1DRACParams; kwargs...)
	prog_steps = 1000
	@unpack tspan, Δₜ, Ntraj = simulation_parameters
	@unpack n, d = getfield(true_system, :sys_dims)
	@unpack true_ξ₀ = getfield(true_system, :init_dists)
    @unpack λₛ, Tₛ = L1params
	true_init = rand(true_ξ₀)
    L1_init = vcat(true_init, true_init) # System and predictor initialized by the same initial condition
	# Error samples for the adaptation law
    Λhat = zeros(n) # Adaptive estimate Over the first interval [0, Tₛ)
    iₐ = 0.0 # index for the piecewise interval
    #Define the problem
	L1_problem = SDEProblem(_L1_drift!, _L1_diffusion!, L1_init, tspan, noise_rate_prototype = zeros(2n, d), (true_system, L1params, Λhat, iₐ))
    #############################################################
    # ADAPTIVE LAW CALLBACK
    integrator = init(L1_problem, EM(), dt=Δₜ)
    function _adaptive_law_condition(u,t,integrator)  
        # Triggers to update Xtilde at t = i*Tₛ, i = floor(t/Tₛ)
        current_iₐ = integrator.p[4]
        return floor(t/Tₛ) - current_iₐ ≥ 1 # returns Boolean true or false, becomes true when t moves to a new interval 
    end 
    # Update the adaptive estimate Λhat when _adaptive_law_condition is true
    function _adaptive_law_affect!(integrator)
        X = integrator.u[1:n]
        Xhat = integrator.u[n+1:2n]
        Xtilde_Tₛ = Xhat - X
        Λhat = ( λₛ/(1-exp(λₛ*Tₛ)) )*Xtilde_Tₛ # Update the adaptive estimate
        iₐ = floor(integrator.t/Tₛ) # Update the adaptive interval index
        integrator.p = (integrator.p[1], integrator.p[2], Λhat, iₐ) 
        # @show integrator.t, integrator.p[3], integrator.p[4]
    end
    adaptive_law_callback = DiscreteCallback(_adaptive_law_condition, _adaptive_law_affect!)
    # Storing the adaptive estimates 
    _save_adaptive_estimates(u, t, integrator) = integrator.p[3]
    adaptive_estimates_values = SavedValues(Float64, Vector{Float64})
    saved_adaptive_estimates_callback = SavingCallback(_save_adaptive_estimates, adaptive_estimates_values; saveat=Δₜ)
    callbacks = CallbackSet(adaptive_law_callback, saved_adaptive_estimates_callback)
    #############################################################
	# Solve the problem
	if haskey(kwargs, :simtype) && kwargs[:simtype] == :ensemble
        @info "Running Ensemble Simulation of L1 System"
        function L1_prob_func(prob, i, repeat)
            rand_init = rand(true_ξ₀)
            remake(prob, u0 = vcat(rand_init, rand_init)) # System and predictor initialized by the same initial condition
        end
        ensemble_L1_problem = EnsembleProblem(L1_problem, prob_func = L1_prob_func)
        L1_sol = solve(ensemble_L1_problem, EM(), dt=Δₜ, trajectories = Ntraj, progress = true, progress_steps = prog_steps)
    else
        @info "Running Single Trajectory Simulation of L1 System" 
	    L1_sol = solve(L1_problem, EM(), dt=Δₜ, progress = true, progress_steps = prog_steps, callback = callbacks)
    end
	@info "Done"
	return L1_sol, adaptive_estimates_values
end
####################################################################################
### Test Functions
# Predictor test 
function predictor_test(simulation_parameters::SimParams, true_system::TrueSystem, L1params::L1DRACParams)
	@warn "---Predictor Test Mode Active---"
    prog_steps = 1000
	@unpack tspan, Δₜ, Ntraj = simulation_parameters
	@unpack n, d = getfield(true_system, :sys_dims)
	@unpack true_ξ₀ = getfield(true_system, :init_dists)
	true_init = rand(true_ξ₀)
    L1_init = vcat(true_init, true_init) # System and predictor initialized by the same initial condition    
	
    @info "---Test Mode: Passing Λμ(t,X), Fσ(t, X), and dWₜ to the Predictor---"
    Xtilde_Tₛ = zeros(n) 
    _predictor_test_drift!(dZ, Z, (true_system, L1params), t) = _L1_drift!(dZ, Z, (true_system, L1params, Xtilde_Tₛ), t; predictor_mode = :test)
    _predictor_test_diffusion!(dZ, Z, (true_system, L1params), t) = _L1_diffusion!(dZ, Z, (true_system, L1params), t; predictor_mode = :test)
	#Define the problem
    L1_problem = SDEProblem(_predictor_test_drift!, _predictor_test_diffusion!, L1_init, tspan, noise_rate_prototype = zeros(2n, d), (true_system, L1params))
    # Solve the problem
    sol = solve(L1_problem, EM(), dt=Δₜ, progress = true, progress_steps = prog_steps)
    isequal(sol[1:n,:], sol[n+1:2n,:]) == true ? (@info "Predictor Test: PASSED") : (@error "Predictor Test FAILED: Predictor does not match System")
    @info "Returning solution arrays"
	return sol # For plotting 
end
# Adaptive law and Predictor Performance 






