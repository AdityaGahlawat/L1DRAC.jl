## L1DRAC for a 1D Double Integrator
using Revise

#### MAIN CODE ####
using L1DRAC
using LinearAlgebra
using Distributions
using ControlSystemsBase
include("ex1plotfunctions.jl")


################################
## USER INPUT

# Simulation Parameters
tspan = (0.0, 5.0)
Δₜ = 0.005 # Time step size
Ntraj = 100 # Number of trajectories in ensemble simulation
simulation_parameters = sim_params(tspan, Δₜ, Ntraj)

# System Dimensions 
n=2
m=1
d=2
system_dimensions = sys_dims(n, m, d)

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
g(t) = [0; 1]
g_perp(t) = [1; 0];
p(t,x) = 0.08*I(2)
nominal_components = nominal_vector_fields(f, g, g_perp, p)

# Uncertain Vector Fields
Λμ(t,x) = [1+sin(x[1]); norm(x)] 
Λσ(t,x) = [0.1+sin(x[2]) 0; 0 0.1+cos(x[2])+sqrt(norm(x))]
uncertain_components = uncertain_vector_fields(Λμ, Λσ)

# Initial distributions
nominal_ξ₀ = MvNormal(zeros(2), 0.01*I(2))
true_ξ₀ = MvNormal([-1.; 2.], 0.1*I(2))
initial_distributions = init_dist(nominal_ξ₀, true_ξ₀)

# L1 DRAC Parameters  
ω = 50.0    
Tₛ = 1*Δₜ # Needs to be a integer multiple of Δₜ
λₛ = 10.0 # Predictor Stability Parameter 
L1params = drac_params(ω, Tₛ, λₛ)
###################################################################
## COMPUTATION START
##################################################################

nominal_system = nom_sys(system_dimensions, nominal_components, initial_distributions)
true_system = true_sys(system_dimensions, nominal_components, uncertain_components, initial_distributions)

# Single Trajectories
nominal_sol = system_simulation(simulation_parameters, nominal_system);
true_sol = system_simulation(simulation_parameters, true_system);

# Ensemble Trajectories
ensemble_nominal_sol = system_simulation(simulation_parameters, nominal_system; simtype = :ensemble);
ensemble_true_sol = system_simulation(simulation_parameters, true_system; simtype = :ensemble);


###################### PLOTS #########################

# Single Trajectories
simplot(nominal_sol; xlabelstring = L"X^\star_{t,1}", ylabelstring = L"X^\star_{t,2}")
simplot(true_sol; xlabelstring = L"X_{t,1}", ylabelstring = L"X_{t,2}")
simplot(nominal_sol, true_sol; labelstring1 = L"X^\star_{t}", labelstring2 = L"X_{t}")

# Ensemble Trajectories 
simplot(ensemble_nominal_sol; xlabelstring = L"X^\star_{t,1}", ylabelstring = L"X^\star_{t,2}")
simplot(ensemble_true_sol; xlabelstring = L"X_{t,1}", ylabelstring = L"X_{t,2}")
simplot(ensemble_nominal_sol, ensemble_true_sol; labelstring1 = L"X^\star_{t}", labelstring2 = L"X_{t}")

###################### TESTS #########################
mutable struct DummyStruct
    x::Vector{Float64}
    y::Vector{Float64}
end

function mytest()

    @show Z = DummyStruct(ones(2), ones(2))
    @show SD = DummyStruct(zeros(2), zeros(2))

    function sys!(SD,Z)
        SD.x = Z.x
        return SD
    end
    sys!(SD,Z)
    @show SD.x
end
mytest()


