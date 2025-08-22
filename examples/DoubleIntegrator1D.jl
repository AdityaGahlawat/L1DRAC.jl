## L1DRAC for a 1D Double Integrator
using Revise
using L1DRAC
using LinearAlgebra
using Distributions

## USER INPUT

# Sim time 
tspan = (0.0, 10.0)

# Number of trajectories in ensemble simulation
Ntraj = 1000

# System Dimensions 
n=2
m=1
d=2

# Nominal Vector Fields 
f(x) = [x[2]; 0]
g(x) = [0; 1]
g_perp(x) = [1; 0];
p(x) = 0.08*I(2)

# Uncertain Vector Fields
Λμ(x) = [0; 0] 
Λσ(x) = [0 0; 0 0]

# Initial distributions
nominal_ξ₀ = MvNormal(zeros(2), 5*I(2))
true_ξ₀ = MvNormal(10*ones(2), 0.1*I(2))

###################################################################
## COMPUTATION START
##################################################################

system_dimensions = sys_dims(n, m, d)
nominal_components = nominal_vector_fields(f, g, g_perp, p)
uncertain_components = uncertain_vector_fields(Λμ, Λσ)

#####
# For quick evaluations
function mytest() 
    A = [0 1; 1 0]
    eval = eigvals(A)
    println("The eigen values are $eval")
end