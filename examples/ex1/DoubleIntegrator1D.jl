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
    # DesiredPoles = 3*[-2+0.5im, -2-0.5im]
    DesiredPoles = -λ*ones(2)
    K = place(sys, DesiredPoles) # Poles of A-B*K
    return K
end
function f(t,x)
    A = [0 1.0; 0 0]
    B = [0; 1.0]
    K = stbl_cntrl(λ)
    return (A-B*K)*x + B*K*trck_traj(t)
end
g(t) = [0; 1]
g_perp(t) = [1; 0];
p_um(t,x) = 2.0*[0.01 0.1]
p_m(t,x) = 1.0*[0.0 0.8]
p(t,x) = vcat(p_um(t,x), p_m(t,x)) 
nominal_components = nominal_vector_fields(f, g, g_perp, p)

# Uncertain Vector Fields 
Λμ_um(t,x) = 1e-2*(1+sin(x[1]))
Λμ_m(t,x) = 1.1*(10+10*cos(x[2])+6*norm(x))
Λμ(t,x) = vcat(Λμ_um(t,x), Λμ_m(t,x)) 
Λσ_um(t,x) = 1e-2*[0.1+cos(x[2]) 2]
Λσ_m(t,x) = 2.8*[0.0 5+sin(x[2])+10*sqrt(norm(x))]
Λσ(t,x) = vcat(Λσ_um(t,x), Λσ_m(t,x))
uncertain_components = uncertain_vector_fields(Λμ, Λσ)

# Initial distributions
nominal_ξ₀ = MvNormal(500.0*ones(n), 1e3*I(n))
true_ξ₀ = MvNormal(-500*ones(n), 1e3*I(n))
initial_distributions = init_dist(nominal_ξ₀, true_ξ₀)

# L1 DRAC Parameters  
ω = 50.    
Tₛ = 10*Δₜ # Needs to be a integer multiple of Δₜ
λₛ = 100. # Predictor Stability Parameter 
L1params = drac_params(ω, Tₛ, λₛ)
###################################################################
## COMPUTATION START
##################################################################

# Define the systems
nominal_system = nom_sys(system_dimensions, nominal_components, initial_distributions)
true_system = true_sys(system_dimensions, nominal_components, uncertain_components, initial_distributions)

# Solve for Single Sample Paths
nom_sol = system_simulation(simulation_parameters, nominal_system);
tru_sol = system_simulation(simulation_parameters, true_system);
L1_sol = system_simulation(simulation_parameters, true_system, L1params);

# Solve for Ensembles of Ntraj Sample Paths
ens_nom_sol = system_simulation(simulation_parameters, nominal_system; simtype = :ensemble);
ens_tru_sol = system_simulation(simulation_parameters, true_system; simtype = :ensemble);
ens_L1_sol = system_simulation(simulation_parameters, true_system, L1params; simtype = :ensemble);
###################### PLOTS #########################

l = @layout [a1 a2; b1 b2]

# Plotting num_to_plot random trajectories from the ensemble
num_to_plot = 10
plt_idxs = rand(1:Ntraj, num_to_plot)
function ensemble_norm_error_plots()
    p1 = plot()
    for i ∈ plt_idxs
        plot!(p1, ens_nom_sol[i].t, map(x->norm(x), ens_nom_sol[i].u-ens_tru_sol[i].u), lw=2, lalpha=0.1, color = 1, label = :false)
        ens_L1_state = [ens_L1_sol[i][j][1:2] for j in 1:length(ens_L1_sol[i])]
        plot!(p1, ens_nom_sol[i].t, map(x->norm(x), ens_nom_sol[i].u-ens_L1_state), lw=2, lalpha=0.1, color = 2, label = :false)
    end
    return p1
end
p1 = ensemble_norm_error_plots()
p2=plot()
for i ∈ plt_idxs
    plot!(p2, ens_nom_sol[i], idxs = (0,1), lw=2, lalpha=0.1, color = 13, label = :false)
    plot!(p2, ens_tru_sol[i], idxs = (0,1), lw=2, lalpha=0.1, color = 7, label = :false)
    plot!(p2, ens_L1_sol[i], idxs = (0,1), lw=2, lalpha=0.1, color = 25, label = :false)
end
p2
p3=plot()
for i ∈ plt_idxs
    plot!(p3, ens_nom_sol[i], idxs = (0,2), lw=2, lalpha=0.1, color = 13, label = :false)
    plot!(p3, ens_tru_sol[i], idxs = (0,2), lw=2, lalpha=0.1, color = 7, label = :false)
    plot!(p3, ens_L1_sol[i], idxs = (0,2), lw=2, lalpha=0.1, color = 25, label = :false)
end
p3


# ###################### TESTS #########################
# # Predictor Visualization 
# predictorplot(L1_sol; predictor_mode = :performance)
# # Predictor test 
# prd_test_sol = predictor_test(simulation_parameters, true_system, L1params);
# predictorplot(prd_test_sol; predictor_mode = :test)





