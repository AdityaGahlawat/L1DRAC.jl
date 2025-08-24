
# function systems()
function nominal_simulation(simulation_parameters, nominal_system)
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