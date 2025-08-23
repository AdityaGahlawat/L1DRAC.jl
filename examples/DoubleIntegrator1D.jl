## L1DRAC for a 1D Double Integrator
using Revise
using L1DRAC

using LinearAlgebra
using Distributions
using ControlSystemsBase
using DifferentialEquations
using Plots
using LaTeXStrings

################################
## USER INPUT

# Simulation Parameters
tspan = (0.0, 10.0)
Δₜ = 0.01 # Time step size
Ntraj = 1000 # Number of trajectories in ensemble simulation

# System Dimensions 
n=2
m=1
d=2

# Nominal Vector Fields
function trck_traj(t) 
    return [3*sin(t); 0.]
end
function stbl_cntrl()
    A = [0 1.0; 0 0]
    B = [0; 1.0] 
    C = I(2)
    D = 0.0 
    sys = ss(A, B, C, D)
    DesiredPoles = 3*[-2+0.5im, -2-0.5im]
    K = place(sys, DesiredPoles) # Poles of A-B*K
    return K
end
function f(t,x)
    A = [0 1.0; 0 0]
    B = [0; 1.0]
    K = stbl_cntrl()
    return (A-B*K)*x + B*K*trck_traj(t)
end
# f(t,x) = [x[2]; 0]
g(t) = [0; 1]
g_perp(t) = [1; 0];
p(t,x) = 0.08*I(2)

# Uncertain Vector Fields
Λμ(t,x) = [0; 0] 
Λσ(t,x) = [0 0; 0 0]

# Initial distributions
nominal_ξ₀ = MvNormal(zeros(2), 5*I(2))
true_ξ₀ = MvNormal(10*ones(2), 0.1*I(2))

###################################################################
## COMPUTATION START
##################################################################

simulation_parameters = sim_params(tspan, Δₜ, Ntraj)
system_dimensions = sys_dims(n, m, d)
nominal_components = nominal_vector_fields(f, g, g_perp, p)
uncertain_components = uncertain_vector_fields(Λμ, Λσ)
initial_distributions = init_dist(nominal_ξ₀, true_ξ₀)

#####
# For quick evaluations
function mytest() 
    A = [0 1; 1 0]
    B = rand(2, 2)
    C = A-B
    x = rand(2)
    # return A*x - [A[1,1]*x[1] + A[1,2]*x[2]; A[2,1]*x[1] + A[2,2]*x[2]]
    return C[1,1] - (A-B)[1,1]
end

function DeterministicPlot()

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
DeterministicPlot()