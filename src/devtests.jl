
## Test Functions for development and debugging
using Plots
using LaTeXStrings
using DifferentialEquations
using UnPack
# import StochasticDiffEq as SDE




# For quick evaluations
function mytest() 
    A = [0 1; 1 0]
    B = rand(2, 2)
    C = A-B
    x = rand(2)
    # return A*x - [A[1,1]*x[1] + A[1,2]*x[2]; A[2,1]*x[1] + A[2,2]*x[2]]
    return C[1,1] - (A-B)[1,1]
end

function DeterministicPlot(simulation_parameters, nominal_components, trck_traj)

	@unpack tspan, Δₜ, Ntraj = simulation_parameters
	@unpack f, g, g_perp, p = nominal_components

	X₀ = zeros(2)
	#Define the problem
	function NominalDeterministic(dX, X, p, t, ) 
		dX[1] = f(t, X)[1] 
		dX[2] = f(t, X)[2]
	end
	#Pass to solvers
	prob = ODEProblem(NominalDeterministic, X₀, tspan)
	sol = solve(prob, ImplicitEuler())

	# Plot
	
	l = @layout [a; b]
    Ref(t) = trck_traj(t)

	
	PositionPhasePlot = plot(sol[1,:], sol[2,:], linewidth = 2, label=false, xlabel = L"X^\star_{t,1}", ylabel = L"X^\star_{t,2}" )
	
	
	
	
	


	PositionTimePlot = plot(sol.t, sol[1,:], color=1, linewidth = 2, label=L"X^\star_{t,1}")
	plot!(PositionTimePlot, sol.t, sol[2,:], color =2, linewidth = 2, label=L"X^\star_{t,2}")
	# plot!(PositionTimePlot, sol.t, sol[3,:], color =3, linewidth = 2, alpha = 0.3, label=L"X^\star_{t,3}")
	# plot!(PositionTimePlot, sol.t, sol[4,:], color =4, linewidth = 2, alpha = 0.3, label=L"X^\star_{t,4}")
	
	plot!(PositionTimePlot, sol.t, map(t -> Ref(t)[1], sol.t), color = 1, linewidth = 2, linestyle = :dash, label=L"p_{ref_{t,1}}")
	
	# plot!(PositionTimePlot, sol.t, map(t -> Ref(t)[2], sol.t), color = 2, linewidth = 2, linestyle = :dash, label=L"p_{ref_{t,1}}")
	
	# plot!(PositionTimePlot, sol.t, map(t -> Xref(t)[2], sol.t), color = 2, linewidth = 2, linestyle = :dash, label=L"X_{ref_{t,2}}")
	
	mainplot = plot(PositionPhasePlot, PositionTimePlot; layout = l, size=(600,800))

	return mainplot
end

function stochastic_simulation(simulation_parameters, nominal_system)
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

function nomsys_simplot(sol)
	l = @layout [a; b]

	PositionPhasePlot = plot(sol[1,:], sol[2,:], linewidth = 2, label=false, xlabel = L"X^\star_{t,1}", ylabel = L"X^\star_{t,2}" )
	
	PositionTimePlot = plot(sol.t, sol[1,:], color=1, linewidth = 2, label=L"X^\star_{t,1}")
	plot!(PositionTimePlot, sol.t, sol[2,:], color =2, linewidth = 2, label=L"X^\star_{t,2}")

	mainplot = plot(PositionPhasePlot, PositionTimePlot; layout = l, size=(600,800))
	
	return mainplot
end

function driftsetup!()
	@show dX[1:n] = f(t,X)[1:n] 
end



