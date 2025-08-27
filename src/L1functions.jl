# L1-DRAC specific functions
function _L1_drift!(D, Z, params, t)
    true_system = params[1] # Can also pass true_system, since we only need nominal parts
    L1params = params[2]
    @unpack X, Xhat = Z
    @unpack n, m = getfield(true_system, :sys_dims)
	@unpack f, g = getfield(true_system, :nom_vec_fields)
    @unpack Λμ = getfield(true_system, :unc_vec_fields)
    @unpack λₛ = L1params
    ## Placeholders
    m == 1 ? uₐ = 0.0 : uₐ = zeros(m)   
    n == 1 ? Λhat = 0. : Λhat = zeros(n) 
    ##
    D.X[1:n] = (f(t, X) + g(t)*uₐ + Λμ(t,X))[1:n]
    D.Xhat[1:n] = (-λₛ*(Xhat-X) + f(t, X) + g(t)*uₐ + Λhat)[1:n] # Predictor
end
# METHOD 3: simulation of L1-DRAC closed-loop system
# Methods 1 and 2 are in \src/simfunctions.jl
function system_simulation(simulation_parameters, true_system::TrueSystem, L1params::L1DRACParams; kwargs...)
	prog_steps = 1000
	@unpack tspan, Δₜ, Ntraj = simulation_parameters
	@unpack n, d = getfield(true_system, :sys_dims)
	@unpack true_ξ₀ = getfield(true_system, :init_dists)
	true_init = rand(true_ξ₀)	
	#Define the problem
	true_system = params[1] # Can also pass true_system, since we only need nominal parts
    L1params = params[2]
	true_problem = SDEProblem(_true_drift!, _true_diffusion!, true_init, tspan, noise_rate_prototype = zeros(n, d), params)
	# Solve the problem
	if haskey(kwargs, :simtype) && kwargs[:simtype] == :ensemble
        println("---Running Ensemble Simulation of True System")
        function true_prob_func(prob, i, repeat)
            remake(prob, u0 = rand(true_ξ₀))
        end
        ensemble_true_problem = EnsembleProblem(true_problem, prob_func = true_prob_func)
        true_sol = solve(ensemble_true_problem, EM(), dt=Δₜ, trajectories = Ntraj, progress = true, progress_steps = prog_steps)
    else
        println("---Running Single Trajectory Simulation of True System") 
	    true_sol = solve(true_problem, EM(), dt=Δₜ, progress = true, progress_steps = prog_steps)
    end
	println("---Done---")
	return true_sol
end
