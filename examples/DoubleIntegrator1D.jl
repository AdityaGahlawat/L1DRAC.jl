## L1DRAC for a 1D Double Integrator
using Revise

#### MAIN CODE ####
using L1DRAC
using LinearAlgebra
using Distributions
using ControlSystemsBase
using Plots
using LaTeXStrings

################################
## USER INPUT

# Simulation Parameters
tspan = (0.0, 5.0)
Δₜ = 0.01 # Time step size
Ntraj = 1000 # Number of trajectories in ensemble simulation

# System Dimensions 
n=2
m=1
d=2

# Nominal Vector Fields
function trck_traj(t) # Reference trajectory for Nominal deterministic system to track 
    return [3*sin(t); 0.]
end
function stbl_cntrl() # Stabilizing controller via pole placement
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
Λμ(t,x) = [sin(x[1]); 10+norm(x)] 
Λσ(t,x) = [cos(x[2]) 0; 0 sqrt(norm(x))]

# Initial distributions
nominal_ξ₀ = MvNormal(zeros(2), 0.1*I(2))
true_ξ₀ = MvNormal(1*ones(2), 10*I(2))

###################################################################
## COMPUTATION START
##################################################################

simulation_parameters = sim_params(tspan, Δₜ, Ntraj)
system_dimensions = sys_dims(n, m, d)
nominal_components = nominal_vector_fields(f, g, g_perp, p)
uncertain_components = uncertain_vector_fields(Λμ, Λσ)
initial_distributions = init_dist(nominal_ξ₀, true_ξ₀)

nominal_system = nom_sys(system_dimensions, nominal_components, initial_distributions)
true_system = true_sys(system_dimensions, nominal_components, uncertain_components, initial_distributions)

nominal_sol = system_simulation(simulation_parameters, nominal_system)
true_sol = system_simulation(simulation_parameters, true_system)

###################### PLOTS #########################


# Plot Functions
function simplot(sol::RODESolution; xlabelstring::LaTeXString, ylabelstring::LaTeXString)
	l = @layout [a; b]

	PositionPhasePlot = plot(sol[1,:], sol[2,:], linewidth = 2, label=false, xlabel = xlabelstring, ylabel = ylabelstring )
	
	PositionTimePlot = plot(sol.t, sol[1,:], color=1, linewidth = 2, label=xlabelstring)
	plot!(PositionTimePlot, sol.t, sol[2,:], color =2, linewidth = 2, label=ylabelstring)

	mainplot = plot(PositionPhasePlot, PositionTimePlot; layout = l, size=(600,800))
	
	return mainplot
end


function simplot(sol1::RODESolution, sol2::RODESolution; labelstring1::LaTeXString, labelstring2::LaTeXString)
	l = @layout [a; b; c]

	PositionPhasePlotTime = plot(sol1[1,:], sol1[2,:], sol1.t, color=1, linewidth = 2, linestyle = :dash, label=false)
    plot!(PositionPhasePlotTime, sol2[1,:], sol2[2,:], sol2.t, color=1, linewidth = 2, label=false, zlabel = L"Time~\rightarrow")

    PositionPhasePlot = plot(sol1[1,:], sol1[2,:], color=1, linewidth = 2, linestyle = :dash, label=labelstring1)
    plot!(PositionPhasePlot, sol2[1,:], sol2[2,:], color=1, linewidth = 2, label=labelstring2)
	
	PositionTimePlot = plot(sol1.t, sol1[1,:], color=1, linewidth = 2, linestyle=:dash, label=false)
	plot!(PositionTimePlot, sol1.t, sol1[2,:], color=2, linewidth = 2, linestyle=:dash, label=false)
    plot!(PositionTimePlot, sol2.t, sol2[1,:], color=1, linewidth = 2, label=false)
	plot!(PositionTimePlot, sol2.t, sol2[2,:], color=2, linewidth = 2, label=false, xlabel = L"Time~\rightarrow")

	mainplot = plot(PositionPhasePlotTime, PositionPhasePlot, PositionTimePlot; layout = l, size=(600,1200))
	
	return mainplot
end



simplot(nominal_sol; xlabelstring = L"X^\star_{t,1}", ylabelstring = L"X^\star_{t,2}")
simplot(true_sol; xlabelstring = L"X_{t,1}", ylabelstring = L"X_{t,2}")
simplot(nominal_sol, true_sol; labelstring1 = L"X^\star_{t}", labelstring2 = L"X_{t}")


###################### TESTS #########################
include("../src/devtests.jl")


DeterministicPlot(simulation_parameters, nominal_components, trck_traj)


nomsys_simplot(nominal_sol)
nomsys_simplot(true_sol)

