## L1DRAC for a 1D Double Integrator
using Revise

#### MAIN CODE ####
using L1DRAC
using LinearAlgebra
using Distributions
using ControlSystemsBase


# Simulation Parameters
tspan = (0.0, 5.0)
Δₜ = 1e-4 # Time step size
Ntraj = 10 # Number of trajectories in ensemble simulation
Δ_saveat = 1e2*Δₜ # Needs to be a integer multiple of Δₜ
simulation_parameters = sim_params(tspan, Δₜ, Ntraj, Δ_saveat)


# System Dimensions 
n=2
m=1
d=2
system_dimensions = sys_dims(n, m, d)

# Nominal Vector Fields
λ = 3.0 # Stability of nominal system 
function trck_traj(t) # Reference trajectory for Nominal deterministic system to track 
    return [5*sin(t) + 3*cos(2*t); 0.]
end

function stbl_cntrl(λ) # Stabilizing controller via pole placement
    A = [0 1.0; 0 0]
    B = [0; 1.0] 
    C = I(2)
    D = 0.0 
    sys = ss(A, B, C, D)
    DesiredPoles = -λ*ones(2)
    K = place(sys, DesiredPoles) # Poles of A-B*K
    return K, norm(A - B*K, 2), A-B*K
end

function f(t,x)
    A = [0 1.0; 0 0]
    B = [0; 1.0]
    K= stbl_cntrl(λ)[1]
    return (A-B*K)*x + B*K*trck_traj(t)
end

g(t) = [0; 1]
g_perp(t) = [1; 0]
g_bar(t)=[g(t) g_perp(t)]
Θ_ad(t) = [I(m) zeros(m, n - m)]*inv(g_bar(t)) 
p_um(t,x) = 2.0*[0.01 0.1]
p_m(t,x) = 1.0*[0.0 0.8]
p(t,x) = vcat(p_um(t,x), p_m(t,x)) 
nominal_components = nominal_vector_fields(f, g, g_perp, p)

# Uncertain Vector Fields 
Λμ_um(t,x) = 1e-2*(1+sin(x[1]))
Λμ_m(t,x) = 1.0*(5+10*cos(x[2])+5*norm(x))
Λμ(t,x) = vcat(Λμ_um(t,x), Λμ_m(t,x)) 
Λσ_um(t,x) = 1e-2*[0.1+cos(x[2]) 2]
Λσ_m(t,x) = 1.0*[0.0 5+sin(x[2])+5.0*sqrt(norm(x))]
Λσ(t,x) = vcat(Λσ_um(t,x), Λσ_m(t,x))
uncertain_components = uncertain_vector_fields(Λμ, Λσ)

# Initial distributions
nominal_ξ₀ = MvNormal(20.0*ones(n), 1e2*I(n))
true_ξ₀ = MvNormal(-2.0*ones(n), 1e1*I(n))
initial_distributions = init_dist(nominal_ξ₀, true_ξ₀)


# Constants 
order_p = 1
constants = assumption_constants(
    λ = λ, 
    Δf = stbl_cntrl(λ)[2],
    Δg = 1.0, 
    Δg_dot = 0.0, 
    Δg_perp = 1.0, 
    Δ_Θ = 1.0,
    Δp = norm([0.02 0.2; 0.0 0.8]), 
    Lhat_p = 0.0, 
    L_p = 0.0,
    Δσ = sqrt(121.000521), 
    Δσ_parallel = 11.0, 
    Δσ_perp = sqrt(0.000521),
    Δμ = sqrt(400.0004), 
    Δμ_parallel = 20.0, 
    Δμ_perp = 0.02,
    Δp_parallel = 0.8, 
    Δp_perp = sqrt(0.0404), 
    Lhat_p_parallel = 0.0, 
    L_p_parallel = 0.0, 
    Lhat_p_perp = 0.0, 
    L_p_perp = 0.0,
    L_μ = sqrt(225.0001), 
    Lhat_μ = 0.0, 
    L_μ_parallel = 15.0, 
    Lhat_μ_parallel = 0.0, 
    L_μ_perp = 0.01, 
    Lhat_μ_perp = 0.0, 
    L_σ = sqrt(49.0004), 
    Lhat_σ = 0.0, 
    L_σ_parallel = 7.0, 
    Lhat_σ_parallel = 0.0, 
    L_σ_perp = 0.02, 
    Lhat_σ_perp = 0.0, 
    L_f = 5 + sqrt(34), 
    Lhat_f = 0.0,
    order_p = order_p,
    Δ_star = Delta_star_Linear_Nominal(stbl_cntrl(λ)[3], [0.01 0.1; 0.0 0.8], simulation_parameters, nominal_ξ₀, 2*order_p)
)



# L1 DRAC Parameters  
ω = 50.    
Tₛ = 10*Δₜ # Needs to be a integer multiple of Δₜ
λₛ = 100. # Predictor Stability Parameter 
L1params = drac_params(ω, Tₛ, λₛ)

###################################################################
## COMPUTATION 
##################################################################

# Define the systems
nominal_system = nom_sys(system_dimensions, nominal_components, initial_distributions)
true_system = true_sys(system_dimensions, nominal_components, uncertain_components, initial_distributions)

# Checking the completeness of defining the constants
validate(constants, true_system)

# Solve for Single Sample Paths
nom_sol = system_simulation(simulation_parameters, nominal_system);
tru_sol = system_simulation(simulation_parameters, true_system);
L1_sol = system_simulation(simulation_parameters, true_system, L1params);

# Solve for Ensembles of Ntraj Sample Paths
@time ens_nom_sol = system_simulation(simulation_parameters, nominal_system; simtype = :ensemble);
@time ens_tru_sol = system_simulation(simulation_parameters, true_system; simtype = :ensemble);
@time ens_L1_sol = system_simulation(simulation_parameters, true_system, L1params; simtype = :ensemble);
###################### PLOTS #########################
include("plotutils.jl")
plotfunc()



# ###################### TESTS #########################
# # Predictor Visualization 
# predictorplot(L1_sol; predictor_mode = :performance)
# # Predictor test 
# prd_test_sol = predictor_test(simulation_parameters, true_system, L1params);
# predictorplot(prd_test_sol; predictor_mode = :test)





