
# function systems()
# Drift Functions
function _nominal_drift!(dX, X, (nominal_system, ), t)
	@unpack n = getfield(nominal_system, :sys_dims)
	@unpack f = getfield(nominal_system, :nom_vec_fields)
	dX[1:n] = f(t,X)[1:n]
end
function _uncertain_drift!(dX, X, (true_system, ), t)
	@unpack n = getfield(true_system, :sys_dims)
	@unpack Λμ = getfield(true_system, :unc_vec_fields)
	dX[1:n] = Λμ(t,X)[1:n]
end
function _true_drift!(dX, X, (true_system, ), t)
	@unpack n = getfield(true_system, :sys_dims)
	@unpack f = getfield(true_system, :nom_vec_fields)
	@unpack Λμ = getfield(true_system, :unc_vec_fields)
	Fμ(t, X) = f(t,X) + Λμ(t,X)
	dX[1:n] = Fμ(t,X)[1:n]
end
# Diffusion Functions
function _nominal_diffusion!(dX, X, (nominal_system, ), t)
	@unpack n, d = getfield(nominal_system, :sys_dims)
	@unpack p = getfield(nominal_system, :nom_vec_fields)
	for i in 1:n
		for j in 1:d
			dX[i,j] = p(t,X)[i,j]
		end
	end
end
function _uncertain_diffusion!(dX, X, (true_system, ), t)
	@unpack n, d = getfield(true_system, :sys_dims)
	@unpack Λσ = getfield(true_system, :unc_vec_fields)
	for i in 1:n
		for j in 1:d
			dX[i,j] = Λσ(t,X)[i,j]
		end
	end
end
function _true_diffusion!(dX, X, (true_system, ), t)
	@unpack n, d = getfield(true_system, :sys_dims)
	@unpack p = getfield(true_system, :nom_vec_fields)
	@unpack Λσ = getfield(true_system, :unc_vec_fields)
	Fσ(t, X) = p(t,X) + Λσ(t,X)
	for i in 1:n
		for j in 1:d
			dX[i,j] = Fσ(t, X)[i,j]
		end
	end
end

##### MAIN SIMULATION FUNCTION (Multiple Dispatch) #####
# METHOD 1: simulation of nominal system
function system_simulation(simulation_parameters::SimParams, nominal_system::NominalSystem; kwargs...)
	prog_steps = 1000
	@unpack tspan, Δₜ, Ntraj = simulation_parameters
	@unpack n, d = getfield(nominal_system, :sys_dims)
	@unpack nominal_ξ₀ = getfield(nominal_system, :init_dists)
	nom_init = rand(nominal_ξ₀)
	#Define the problem
    nominal_problem = SDEProblem(_nominal_drift!, _nominal_diffusion!, nom_init, tspan, noise_rate_prototype = zeros(n, d), (nominal_system,))
    # Solve the problem
    if haskey(kwargs, :simtype) && kwargs[:simtype] == :ensemble
        @info "Running Ensemble Simulation of Nominal System"
        function nominal_prob_func(prob, i, repeat)
            remake(prob, u0 = rand(nominal_ξ₀))
        end
        ensemble_nominal_problem = EnsembleProblem(nominal_problem, prob_func = nominal_prob_func)
        nominal_sol = solve(ensemble_nominal_problem, EM(), dt=Δₜ, trajectories = Ntraj, progress = true, progress_steps = prog_steps)
    else
        @info "Running Single Trajectory Simulation of Nominal System" 
	    nominal_sol = solve(nominal_problem, EM(), dt=Δₜ, progress = true, progress_steps = prog_steps)
    end
	@info "Done"
	return nominal_sol
end
# METHOD 2: simulation of true system
function system_simulation(simulation_parameters::SimParams, true_system::TrueSystem; kwargs...)
	prog_steps = 1000
	@unpack tspan, Δₜ, Ntraj = simulation_parameters
	@unpack n, d = getfield(true_system, :sys_dims)
	@unpack true_ξ₀ = getfield(true_system, :init_dists)
	true_init = rand(true_ξ₀)	
	#Define the problem
	true_problem = SDEProblem(_true_drift!, _true_diffusion!, true_init, tspan, noise_rate_prototype = zeros(n, d), (true_system, ))
	# Solve the problem
	if haskey(kwargs, :simtype) && kwargs[:simtype] == :ensemble
        @info "Running Ensemble Simulation of True System"
        function true_prob_func(prob, i, repeat)
            remake(prob, u0 = rand(true_ξ₀))
        end
        ensemble_true_problem = EnsembleProblem(true_problem, prob_func = true_prob_func)
        true_sol = solve(ensemble_true_problem, EM(), dt=Δₜ, trajectories = Ntraj, progress = true, progress_steps = prog_steps)
    else
        @info "Running Single Trajectory Simulation of True System" 
	    true_sol = solve(true_problem, EM(), dt=Δₜ, progress = true, progress_steps = prog_steps)
    end
	@info "Done"
	return true_sol
end
