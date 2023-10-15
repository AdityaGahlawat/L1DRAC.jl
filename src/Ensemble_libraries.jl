using Distributed

if myid() == 1
    println(">>> Loading libraries")
end

using DiffEqGPU, CUDA, OrdinaryDiffEq, Test, Random, StaticArrays, Distributions, LaTeXStrings, LinearAlgebra, Revise, PrecompileTools



