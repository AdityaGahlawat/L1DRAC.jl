# Simple Pendulum Problem from https://docs.sciml.ai/DiffEqDocs/stable/examples/classical_physics/#Second-Order-Non-linear-ODE

import OrdinaryDiffEq as ODE, Plots
using  LaTeXStrings

#Define the problem
function simplependulum(du, u, p, t)    
    #Constants
    g = 9.81
    L = 1.0

    θ, ω = u
    du[1] = ω
    du[2] = -(g / L) * sin(θ)
end

function simplependulum_PCA(du, u, p, t)    

    # u = [θ, ω, θ_pca, ω_pca]

    #Constants
    g = 9.81
    L = 1.0

    θ = u[1]
    ω = u[2]
    du[1] = ω
    du[2] = -(g / L) * sin(θ)
    if (floor(t/p.Tₛ) > floor((t-p.Δₜ)/p.Tₛ) && t ≥ p.Tₛ) == true
        du[3] = θ/p.Δₜ # dθ_pca 
        du[4] = ω/p.Δₜ # dω_pca
        # du[3] = θ # dθ_pca 
        # du[4] = ω # dω_pca
        @info "Trigerred at" t
        @info floor(t/p.Tₛ), floor((t-p.Δₜ)/p.Tₛ), t ≥ p.Tₛ
        @info "Tₛ =", p.Tₛ
    else
        du[3] = 0.0 #dθ_pca
        du[4] = 0.0 #dω_pca
    end    
end

function PCA_test()

    Δₜ = 0.01

    #Parameters
    p = (Δₜ, Tₛ = 50*Δₜ)

    #Initial Conditions
    u₀ = [0, π / 2, 0.0, 0.0] # [θ, ω, θ_pca, ω_pca]
    tspan = (0.0, 5.0)

    #PCA Problem 
    prob = ODE.ODEProblem(simplependulum_PCA, u₀, tspan, p)
    sol = ODE.solve(prob, ODE.Euler(), dt = Δₜ)

    #Plots
    p1 = Plots.plot(sol.t, sol[1, :], linewidth = 2, xaxis = "Time", label = L"\theta", dpi = 1000)
    Plots.plot!(p1, sol.t, sol[3, :], linewidth = 2, xaxis = "Time", label = L"\hat{\theta}", dpi = 1000)

    p2 = Plots.plot(sol.t, sol[2, :], linewidth = 2, xaxis = "Time", label = L"\dot{\theta}", dpi = 1000)
    Plots.plot!(p2, sol.t, sol[4, :], linewidth = 2, xaxis = "Time", label = L"\hat{\dot{\theta}}", dpi = 1000)

    return Plots.plot(p1, p2, layout=(2, 1))


    #Plot
    # return Plots.plot(sol.t, sol[1:4, :]', linewidth = 2, title = "Simple Pendulum PCA", xaxis = "Time",
        # yaxis = "Height", label = [L"\theta" L"\dot{\theta}" L"\hat{\theta}" L"\hat{\dot{\theta}}"], dpi = 1000)
end
PCA_test()
# Plots.savefig(PCA_test(), "PCA_test.png")