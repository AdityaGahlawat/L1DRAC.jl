## L1DRAC for a 1D Double Integrator for AFOSR 2025

#### MAIN CODE ####
using L1DRAC
using LinearAlgebra
using Distributions
using ControlSystemsBase
using JLD2
import L1DRAC: L1DRACParams

include("constanttypes.jl")
include("sysconstants.jl")
include("Tsconstants.jl")
include("computebounds.jl")
include("utils.jl")

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

nominal_components = nominal_vector_fields(f, g, g_perp, p)
Δ_star =3.1
# Uncertain Vector Fields 
Λμ_um(t,x)  = 1e-4
Λμ_m(t,x)   = 1.5* (1 + norm(x))
Δμ_parallel = 1.5
Δμ_perp =0.01
Λμ(t,x) = vcat(Λμ_um(t,x), Λμ_m(t,x)) 
Δμ=1.5
L_μ=1.5
L_μ_parallel =1.5
L_μ_perp = 1e-5

Λσ_um(t,x) = [1e-5 1e-5]
Λσ_m(t,x) =  [1e-5 1e-5]
Λσ(t,x) = vcat(Λσ_um(t,x), Λσ_m(t,x))
uncertain_components = uncertain_vector_fields(Λμ, Λσ)

# Initial distributions
nominal_ξ₀ = MvNormal(2.0*ones(2), I(2))
true_ξ₀ = MvNormal(-15.0*ones(2), 5*I(2))
initial_distributions = init_dist(nominal_ξ₀, true_ξ₀)

# The Wasserstein metric of order 2 between two Normal distributions. Currently, α supports only Normal distributions.
α = alpha(initial_distributions) 

# Constants ε_r, ε_a from Sec. 3.2
ϵ_r=0.1
ϵ_a=0.1

assumption_constants = assump_consts(; Δg, Δg_perp, Δf, Δ_star,
                                       Δp,Δp_parallel,Δp_perp, Δμ, Δμ_parallel, Δμ_perp ,
                                       L_μ, L_μ_parallel,L_μ_perp, L_f, λ, m, ϵ_r, ϵ_a )

# # Reference Process Analysis constants (Sec. A.1)
ref_system_constants =  ref_sys_constants(assumption_constants) 

# # True Process Analysis constants (Sec. A.2)
true_system_constants = true_sys_constants(assumption_constants)

# # ###################################################################
# # ## COMPUTATION 
# # ##################################################################

# # Define the systems
nominal_system = nom_sys(system_dimensions, nominal_components, initial_distributions)
true_system = true_sys(system_dimensions, nominal_components, uncertain_components, initial_distributions)

# # # Simulation Parameters
tspan = (0.0, 3.0)
Δₜ = 1e-4 # Time step size
### need to fix this initalizing 
Tₛ= 1e-3
Ntraj = 1000 # Number of trajectories in ensemble simulation
Δ_saveat = 1e2*Δₜ # Needs to be a integer multiple of Δₜ
simulation_parameters = sim_params(tspan, Δₜ, Ntraj, Δ_saveat)

# # # L1 DRAC Parameters  
ρᵣ, ρₐ, ρ, ω =  rho_and_filter_bandwidth_computation(α, assumption_constants, ref_system_constants, true_system_constants)   
λₛ = 100. # Predictor Stability Parameter 
L1params = drac_params(ω, Tₛ ,λₛ)

Tₛ= sampling_period_computation(ρₐ, ρᵣ, ω)

#Need to fix this, calling twice to update Tₛ with new value
L1params = drac_params(ω, Tₛ, λₛ)

@show ρᵣ, ρₐ, ρ, ω
@show Tₛ


# The plots below are retained from the DoubleIntegrator example. To be modified.

# Solve for Single Sample Paths
# nom_sol = system_simulation(simulation_parameters, nominal_system);
# tru_sol = system_simulation(simulation_parameters, true_system);
# L1_sol = system_simulation(simulation_parameters, true_system, L1params);

# # Solve for Ensembles of Ntraj Sample Paths
# @time ens_nom_sol = system_simulation(simulation_parameters, nominal_system; simtype = :ensemble);
# @time ens_tru_sol = system_simulation(simulation_parameters, true_system; simtype = :ensemble);
# @time ens_L1_sol = system_simulation(simulation_parameters, true_system, L1params; simtype = :ensemble);

# # # # ###################### SAVE TRAJECTORIES #########################
# ens_nom_sol = Array(ens_nom_sol)
# jldsave("ens_nom_solution.jld2"; ens_nom_sol)
# # jldsave("ens_tru_solution.jld2";ens_tru_sol)
# ens_L1_sol= Array(ens_L1_sol)
# jldsave("ens_L1_solution.jld2";ens_L1_sol)

# # ###################### PLOTS #########################
# include("plotutils.jl")
# plotfunc()

