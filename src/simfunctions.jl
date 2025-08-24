
# function systems()
# method 1: simulation of nominal system
function system_simulation(simulation_parameters, nominal_system::NominalSystem)
	@unpack tspan, Δₜ, Ntraj = simulation_parameters
	@unpack n, m, d = getfield(nominal_system, :sys_dims)
	@unpack f, g, g_perp, p = getfield(nominal_system, :nom_vec_fields)
	@unpack nominal_ξ₀, true_ξ₀ = getfield(nominal_system, :init_dists)

	nom_init = rand(nominal_ξ₀)

	function _nominal_drift!(dX, X, params, t)
		dX[1:n] = f(t,X)[1:n]
	end
	function _nominal_diffusion!(dX, X, params, t)
		for i in 1:n
			for j in 1:d
				dX[i,j] = p(t,X)[i,j]
			end
		end
	end
	
	#Define the problem
	nominal_problem = SDEProblem(_nominal_drift!, _nominal_diffusion!, nom_init, tspan, noise_rate_prototype = zeros(n, d))
	nominal_sol = solve(nominal_problem, EM(), dt=Δₜ)
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

    Fμ(t, X) = f(t,X) + Λμ(t,X)
    Fσ(t, X) = p(t,X) + Λσ(t,X)

	function _true_drift!(dX, X, params, t)
		dX[1:n] = Fμ(t,X)[1:n]
	end
	function _true_diffusion!(dX, X, params, t)
		for i in 1:n
			for j in 1:d
				dX[i,j] = Fσ(t,X)[i,j]
			end
		end
	end
	
	#Define the problem
	true_problem = SDEProblem(_true_drift!, _true_diffusion!, true_init, tspan, noise_rate_prototype = zeros(n, d))
	true_sol = solve(true_problem, EM(), dt=Δₜ)
	return true_sol
end