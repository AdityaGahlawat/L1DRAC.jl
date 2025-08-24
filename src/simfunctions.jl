
# function systems()
function _nominal_drift!(dX, X, params, t)
	dX[1:Main.n] = Main.f(t,X)[1:Main.n]
end
function _nominal_diffusion!(dX, X, params, t)
	for i in 1:Main.n
		for j in 1:Main.d
			dX[i,j] = Main.p(t,X)[i,j]
		end
	end
end
function _true_drift!(dX, X, params, t)
	Fμ(t, X) = Main.f(t,X) + Main.Λμ(t,X)
	dX[1:Main.n] = Fμ(t,X)[1:Main.n]
end
function _true_diffusion!(dX, X, params, t)
	Fσ(t, X) = Main.p(t,X) + Main.Λσ(t,X)
	for i in 1:Main.n
		for j in 1:Main.d
			dX[i,j] = Fσ(t,X)[i,j]
		end
	end
end

##### MAIN SIMULATION FUNCTION #####
# method 1: simulation of nominal system
function system_simulation(simulation_parameters, nominal_system::NominalSystem; kwargs...)
	@unpack tspan, Δₜ, Ntraj = simulation_parameters
	@unpack n, m, d = getfield(nominal_system, :sys_dims)
	@unpack f, g, g_perp, p = getfield(nominal_system, :nom_vec_fields)
	@unpack nominal_ξ₀, true_ξ₀ = getfield(nominal_system, :init_dists)
	nom_init = rand(nominal_ξ₀)
	#Define the problem
    nominal_problem = SDEProblem(_nominal_drift!, _nominal_diffusion!, nom_init, tspan, noise_rate_prototype = zeros(n, d))
    # Solve the problem
    if haskey(kwargs, :simtype) && kwargs[:simtype] == :ensemble
        println("Running Ensemble Simulation of Nominal System")
        function nominal_prob_func(prob, i, repeat)
            remake(prob, u0 = rand(nominal_ξ₀))
        end
        ensemble_nominal_problem = EnsembleProblem(nominal_problem, prob_func = nominal_prob_func)
        nominal_sol = solve(ensemble_nominal_problem, EM(), dt=Δₜ, trajectories = Ntraj)
    else
        println("Running Single Trajectory Simulation of Nominal System") 
	    nominal_sol = solve(nominal_problem, EM(), dt=Δₜ)
    end
	return nominal_sol
end
# method 2: simulation of true system
function system_simulation(simulation_parameters, true_system::TrueSystem)
	@unpack tspan, Δₜ, Ntraj = simulation_parameters
	@unpack n, m, d = getfield(true_system, :sys_dims)
	@unpack f, g, g_perp, p = getfield(true_system, :nom_vec_fields)
    @unpack Λμ, Λσ = getfield(true_system, :unc_vec_fields)
	@unpack nominal_ξ₀, true_ξ₀ = getfield(true_system, :init_dists)
	true_init = rand(true_ξ₀)	
	#Define the problem
	true_problem = SDEProblem(_true_drift!, _true_diffusion!, true_init, tspan, noise_rate_prototype = zeros(n, d))
	true_sol = solve(true_problem, EM(), dt=Δₜ)
	return true_sol
end