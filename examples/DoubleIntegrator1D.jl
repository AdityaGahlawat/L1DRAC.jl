## L1DRAC for a 1D Double Integrator
using Revise
using L1DRAC
using LinearAlgebra

# Nominal System 
f(x) = [x[2]; 0]
g(x) = [0; 1]
p(x) = 0.08*I(2)

nominal_system = nominal_vector_fields(f, g, p)