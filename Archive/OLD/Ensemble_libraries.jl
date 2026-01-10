using Distributed

if myid() ≠ 1
    println("I'm loading libraries")
end

using DiffEqGPU, CUDA, OrdinaryDiffEq, Test, Random, StaticArrays, Distributions, LaTeXStrings, LinearAlgebra, Revise, PrecompileTools



