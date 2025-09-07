## L1DRAC for a 1D Double Integrator
#using Revise

#### MAIN CODE ####
using L1DRAC
using UnPack
using LinearAlgebra
using Distributions
using ControlSystemsBase

include("constanttypes.jl")
include("sysconstants.jl")
include("computebounds.jl")
include("utils.jl")

# System Dimensions 
n=2
m=1
d=2
system_dimensions = sys_dims(n, m, d)

# # Nominal Vector Fields
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

# # Uncertain Vector Fields 
Λμ_um(t,x) = 1e-5
Λμ_m(t,x) = 1.5* (1 + norm(x))
Λμ(t,x) = vcat(Λμ_um(t,x), Λμ_m(t,x)) 
Λσ_um(t,x) = [1e-5 1e-5]
Λσ_m(t,x) = 1.0*[0.0 0.5*sqrt(norm(x))]
Λσ(t,x) = vcat(Λσ_um(t,x), Λσ_m(t,x))
uncertain_components = uncertain_vector_fields(Λμ, Λσ)

# # Initial distributions
nominal_ξ₀ = MvNormal(2.0*ones(2), I(2))
true_ξ₀ = MvNormal(-2.0*ones(2), I(2))
initial_distributions = init_dist(nominal_ξ₀, true_ξ₀)


assumption_constants = assump_consts(
    Δg=1.0, 
    Δg_dot=0.0, 
    Δg_perp=1.0,
    Δf=75,
    Δ_star=10,

    Δp=0.9, 
    Δp_parallel=0.8, 
    Δp_perp =0.3, 

    Δμ=1.5,
    Δμ_parallel=1.5,
    Δμ_perp=0.0,

    Δσ=0.5, 
    Δσ_parallel=0.5, 
    Δσ_perp=0.0, 

    L_p=0.0, 
    L_p_parallel=0.0,
    L_p_perp=0.0, 

    L_μ=1.5, 
    L_μ_parallel=1.5,
    L_μ_perp=0.0,

    L_σ=0.5,
    L_σ_parallel=0.5,
    L_σ_perp=0.0,
    
    L_f=10.9,
    λ=3.0, m=1.0, 
    ϵ_r=0.2, ϵ_a=0.2
)
ref_sys_constants =  RefSystemConstants(assumption_constants) 
true_sys_constants = TrueSystemConstants(assumption_constants) 
# ###################################################################
# ## COMPUTATION 
# ##################################################################

# Define the systems
nominal_system = nom_sys(system_dimensions, nominal_components, initial_distributions)
true_system = true_sys(system_dimensions, nominal_components, uncertain_components, initial_distributions)

# Simulation Parameters
tspan = (0.0, 5.0)
Δₜ = 1e-4 # Time step size
Ntraj = 10 # Number of trajectories in ensemble simulation
Δ_saveat = 1e2*Δₜ # Needs to be a integer multiple of Δₜ
simulation_parameters = sim_params(tspan, Δₜ, Ntraj, Δ_saveat)

# L1 DRAC Parameters  
ω = 90.    
Tₛ = 10*Δₜ # Needs to be a integer multiple of Δₜ
λₛ = 100. # Predictor Stability Parameter 
L1params = drac_params(ω, Tₛ, λₛ)


rho_r, rho_a, rho= find_rho(initial_distributions,assumption_constants,true_sys_constants ,ref_sys_constants, L1params)
@show rho

ω_condns_satisfied= filter_bandwidth_conditions(rho_r, rho_a, initial_distributions, assumption_constants, ref_sys_constants, true_sys_constants, L1params)
@show ω_condns_satisfied