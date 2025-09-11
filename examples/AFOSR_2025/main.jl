## L1DRAC for a 1D Double Integrator for AFOSR 2025

#### MAIN CODE ####
using L1DRAC
using LinearAlgebra
using Distributions
using ControlSystemsBase
using JLD2, FileIO
import L1DRAC: L1DRACParams, InitialDistributions, SysDims

include("constanttypes.jl")
include("sysconstants.jl")
include("Tsconstants.jl")
include("computebounds.jl")
include("utils.jl")


results_directory= "/Users/sambhu/Desktop/L1DRAC.jl/examples/AFOSR_2025/Results/"
plots_directory= "/Users/sambhu/Desktop/GitRepos/L1DRAC.jl/examples/AFOSR_2025/"

#exp_name= "matched_drift_un_"
#exp_name= "matched_drift_matched_diff_un_"
# exp_name= "unmatched_drift_matched_diff_un_"
exp_name= "unmatched_drift_unmatched_diff_un_n"

ens_nom_sol_filename = results_directory  * exp_name  * "ens_nom_sol.jld2"
ens_tru_sol_filename = results_directory  * exp_name  * "ens_true_sol.jld2"
ens_L1_sol_filename =  results_directory  * exp_name  * "ens_l1_sol.jld2"

# System Dimensions 
n=2
m=1
d=2
system_dimensions = sys_dims(n, m, d)

# Nominal Vector Fields
#  Stability of nominal system 
λ=3.0
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
Δf=75
L_f=10.9

g(t) = [0; 1]
Δg=1.0
g_perp(t) = [1; 0];
Δg_perp=1.0

p_m(t,x)  = 1.0*[0.0 0.8]
p_um(t,x) = 2.0*[0.01 0.1]
Δp_parallel=0.8
Δp_perp=0.2
p(t,x) = vcat(p_um(t,x), p_m(t,x)) 
Δp=0.83

# Simulation Parameters
tspan = (0.0, 5.0)
Δₜ = 1e-4 # Time step size
Tₛ = Δₜ # INITIAL CONDITION FOR Ts (NOT Ts!!!!)
Ntraj = 100 # Number of trajectories in ensemble simulation
Δ_saveat = 1e2*Δₜ # Needs to be a integer multiple of Δₜ
simulation_parameters = sim_params(tspan, Δₜ, Ntraj, Δ_saveat)
order_p=1     

# Initial distributions
nominal_ξ₀ = MvNormal(2.0*ones(2), I(2))
true_ξ₀    = MvNormal(-15.0*ones(2), 5*I(2))
initial_distributions = init_dist(nominal_ξ₀, true_ξ₀)


#Define the nominal system
nominal_components = nominal_vector_fields(f, g, g_perp, p)
nominal_system = nom_sys(system_dimensions, nominal_components, initial_distributions)

# Simulate nominal system
ens_nom_sol = nom_system_simulation(ens_nom_sol_filename, simulation_parameters, nominal_system; overwrite=true)
Δ_star = Delta_star_computation(ens_nom_sol, order_p ,tspan, Δ_saveat)

# Uncertain Vector Fields 
# Drift 
K_mu = 10.0
K_mu_perp= 0.3
Λμ_um(t,x)  = K_mu_perp* sqrt(1 + norm(x)^2)
Λμ_m(t,x)   = K_mu * sqrt(1 + norm(x)^2)
Λμ(t,x) = vcat(Λμ_um(t,x), Λμ_m(t,x)) 
Δμ=sqrt(K_mu^2 + K_mu_perp^2 )
Δμ_parallel = K_mu
Δμ_perp = 0.01#K_mu_perp
L_μ=sqrt(K_mu^2 + K_mu_perp^2 )
L_μ_parallel =K_mu
L_μ_perp = K_mu_perp

# Diffusion
K_sigma = 5.0
K_sigma_perp = 0.1
Λσ_um(t,x) = K_sigma_perp * [0.0 1.0+sqrt(norm(x))]
Λσ_m(t,x) =  K_sigma * [0.0 1.0+sqrt(norm(x))]

Λσ(t,x) = vcat(Λσ_um(t,x), Λσ_m(t,x))
uncertain_components = uncertain_vector_fields(Λμ, Λσ)
Δσ=sqrt(K_sigma^2 + K_sigma_perp^2) 
Δσ_parallel = K_sigma
Δσ_perp = K_sigma_perp
L_σ= sqrt(K_sigma^2+K_sigma_perp^2 )
L_σ_parallel =K_sigma
L_σ_perp =K_sigma_perp

α = alpha_computation(initial_distributions,system_dimensions, Ntraj) 

# Constants ε_r, ε_a from Sec. 3.2
ϵ_r = 1e-3
ϵ_a =1e-3

assumption_constants = assump_consts(;
    Δg, Δg_perp, Δf, Δ_star,
    Δp, Δp_parallel, Δp_perp,
    Δμ, Δμ_parallel, Δμ_perp,
    Δσ, Δσ_parallel, Δσ_perp,
    L_μ, L_μ_parallel, L_μ_perp,
    L_σ, L_σ_parallel,L_σ_perp,
    L_f, λ, m, ϵ_r, ϵ_a)


# # # # # Reference Process Analysis constants (Sec. A.1)
ref_system_constants =  ref_sys_constants(assumption_constants) 

# # # # True Process Analysis constants (Sec. A.2)
true_system_constants = true_sys_constants(assumption_constants)

##################################################################
# COMPUTATION 
#################################################################

# Define the True system
true_system = true_sys(system_dimensions, nominal_components, uncertain_components, initial_distributions)


# L1 DRAC Parameters  
ρᵣ, ρₐ, ρ, ω =  rho_and_filter_bandwidth_computation(α, assumption_constants, ref_system_constants, true_system_constants);  
λₛ = 100. # Predictor Stability Parameter 
L1params = drac_params(ω, Tₛ ,λₛ)
# @show 

Tₛ= sampling_period_computation!(ρₐ, ρᵣ, ω, Ts_min= Δₜ )
# if Tₛ ≥ 4*Δₜ 
#     Tₛ = 4*Δₜ
# end
# #Need to fix this, calling twice to update Tₛ with new value
L1params = drac_params(ω, Tₛ, λₛ)

@show ρ, ω, Tₛ

# Simulate a single sample path of the true system with L1 control off and on.

tru_sol = system_simulation(simulation_parameters, true_system);
L1_sol = system_simulation(simulation_parameters, true_system, L1params);

# Solve for Ensembles of Ntraj Sample Paths

@time ens_tru_sol = system_simulation(simulation_parameters, true_system; simtype = :ensemble);
@time ens_L1_sol = system_simulation(simulation_parameters, true_system, L1params; simtype = :ensemble);
ens_tru_sol_file =results_directory  * exp_name  * "ens_tru_sol.jld2"
ens_L1_sol_file  =results_directory   * exp_name   * "ens_L1_sol.jld2"

###################### SAVE TRAJECTORIES #########################

jldsave(ens_tru_sol_filename; ens_tru_sol=ens_tru_sol) 
jldsave(ens_L1_sol_filename; ens_L1_sol=ens_L1_sol) 

# # # # # ###################### PLOTS #########################
include("plotutils.jl")
plot_ylims= (0, 900.)

ens_filename = plots_directory * exp_name * "ens_plot.png"
plotfunc(ens_filename,(0.0, 1.0),plot_ylims)

###################### Wasserstein PLOTS #########################
include("wasserstein_plots.jl")
# # # ens_nom_sol, ens_tru_sol, ens_L1_sol= load_simulation_data(exp_name);
wassd_filename = plots_directory * exp_name * "ens_plot_wasserstein.png"
plot_wasserstein(exp_name, (0.0,1.0), ρᵣ, ρₐ; plot_ylims= (0, 250))

# # # # plot_wasserstein_from_saved_data(exp_name, tspan, ρᵣ, ρₐ)

# # comparison_plot_filename = plots_directory * exp_name * "m_drift_vs_m_drift_m_diff_comparison.png"
# # plot_wasserstein_overlay("matched_drift_un_","matched_drift_matched_diff_un_", tspan )
# plot_wasserstein_overlay("matched_drift_un_","matched_drift_matched_diff_un_","unmatched_drift_unmatched_diff_un_n" ,tspan )