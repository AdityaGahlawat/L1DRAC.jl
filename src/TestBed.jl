using Revise
using Pkg; Pkg.activate("L1DRAC.jl/") # Relative path to the julia initiation folder 
# Adding deps to Project.toml and Manifest.toml
Pkg.add("LinearAlgebra")
Pkg.add("OrdinaryDiffEq")
Pkg.add("DifferentialEquations")

using L1DRAC # Revise.jl before L1DRAC during dev


# plot(1:10, rand(10), label="Random Data") 
