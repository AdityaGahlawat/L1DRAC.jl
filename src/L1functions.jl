# L1-DRAC specific functions
# Algebraic Predictor 
# To be called within _L1_drift! 
function _adaptation_law!(X, Xhat, Xtilde_Tₛ, t, n, λₛ, Tₛ) 
    iₐ(t) = floor(t/Tₛ) # index for the piecewise interval
    if iₐ(t)*Tₛ == t 
        Xtilde_Tₛ = X - Xhat 
        Λhat = λₛ*(1-exp(λₛ*Tₛ))*Xtilde_Tₛ
    else 
        Λhat = λₛ*(1-exp(λₛ*Tₛ))*Xtilde_Tₛ
    end 
    return Λhat, Xtilde_Tₛ
end
# Differential Equation Components
function _L1_drift!(dZ, Z, (true_system, L1params, _adaptation_law), t; kwargs...)
    @unpack n, m = getfield(true_system, :sys_dims)
	@unpack f, g, g_perp = getfield(true_system, :nom_vec_fields)
    @unpack Λμ = getfield(true_system, :unc_vec_fields)
    @unpack λₛ, Tₛ = L1params
    # Need the following unrefined concatenation for StaticArrays and GPU compatibility
    X = Z[1:n] 
    Xhat = Z[n+1:2n]
    ## Placeholders
    m == 1 ? uₐ = 0.0 : uₐ = zeros(m)   
    n == 1 ? Λhat = 0. : Λhat = zeros(n)
    adapttest =  _adaptation_law!(X, Xhat, zeros(n), t, n, λₛ, Tₛ)
    ##########################
    # System
    dX = f(t, X) + g(t)*uₐ + Λμ(t,X)  
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
	true_init = rand(true_ξ₀)
    L1_init = vcat(true_init, true_init) # System and predictor initialized by the same initial condition
	#Define the problem
	L1_problem = SDEProblem(_L1_drift!, _L1_diffusion!, L1_init, tspan, noise_rate_prototype = zeros(2n, d), (true_system, L1params, _adaptation_law))
	# Solve the problem
	if haskey(kwargs, :simtype) && kwargs[:simtype] == :ensemble
        println("---Running Ensemble Simulation of L1 System")
        function L1_prob_func(prob, i, repeat)
            rand_init = rand(true_ξ₀)
            remake(prob, u0 = vcat(rand_init, rand_init)) # System and predictor initialized by the same initial condition
        end
        ensemble_L1_problem = EnsembleProblem(L1_problem, prob_func = L1_prob_func)
        L1_sol = solve(ensemble_L1_problem, EM(), dt=Δₜ, trajectories = Ntraj, progress = true, progress_steps = prog_steps)
    else
        println("---Running Single Trajectory Simulation of L1 System") 
	    L1_sol = solve(L1_problem, EM(), dt=Δₜ, progress = true, progress_steps = prog_steps)
    end
	println("---Done---")
	return L1_sol
end
####################################################################################
# Test Functions
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
    _predictor_test_drift!(dZ, Z, (true_system, L1params), t) = _L1_drift!(dZ, Z, (true_system, L1params), t; predictor_mode = :test)
    _predictor_test_diffusion!(dZ, Z, (true_system, L1params), t) = _L1_diffusion!(dZ, Z, (true_system, L1params), t; predictor_mode = :test)
	#Define the problem
    L1_problem = SDEProblem(_predictor_test_drift!, _predictor_test_diffusion!, L1_init, tspan, noise_rate_prototype = zeros(2n, d), (true_system, L1params))
    # Solve the problem
    sol = solve(L1_problem, EM(), dt=Δₜ, progress = true, progress_steps = prog_steps)
    isequal(sol[1:n,:], sol[n+1:2n,:]) == true ? (@info "Predictor Test: PASSED") : (@error "Predictor Test FAILED: Predictor does not match System")
    @info "Returning solution arrays"
	return sol # For plotting 
end






