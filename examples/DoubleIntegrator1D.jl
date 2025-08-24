## L1DRAC for a 1D Double Integrator
using Revise

#### MAIN CODE ####
using L1DRAC
using LinearAlgebra
using Distributions
using ControlSystemsBase



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
Λμ(t,x) = [0; 0] 
Λσ(t,x) = [0 0; 0 0]

# Initial distributions
nominal_ξ₀ = MvNormal(zeros(2), 0.1*I(2))
true_ξ₀ = MvNormal(10*ones(2), 0.1*I(2))

###################################################################
## COMPUTATION START
##################################################################

simulation_parameters = sim_params(tspan, Δₜ, Ntraj)
system_dimensions = sys_dims(n, m, d)
nominal_components = nominal_vector_fields(f, g, g_perp, p)
uncertain_components = uncertain_vector_fields(Λμ, Λσ)
initial_distributions = init_dist(nominal_ξ₀, true_ξ₀)

nominal_system = nom_sys(system_dimensions, nominal_components, initial_distributions)
nominal_sol = nominal_simulation(simulation_parameters, nominal_system)


##### TESTS ##### 
include("../src/devtests.jl")


DeterministicPlot(simulation_parameters, nominal_components, trck_traj)


nomsys_simplot(nominal_sol)


