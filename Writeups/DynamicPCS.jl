# Demonstration of simple PCA law on the Lorenz Equations

import DifferentialEquations as DE
using LinearAlgebra
import Plots 
using LaTeXStrings

σ  = 10.0
ρ  = 28.0
β  = 8 / 3
λₛ = -1e1
Δt = 1e-4
N  = 5e3
Tₛ = N * Δt
p  = [σ, ρ, β, λₛ, Tₛ, Δt]


function lorenz_PCA!(dZ, Z, p, t)
    X    = Z[1:3]
    Xhat = Z[4:6]
    Λhat = Z[7:9]

    Tₛ = p[5]
    Δt = p[6]

    f₁(X,p) = p[1] * (X[2] - X[1])
    f₂(X,p) = X[1] * (p[2] - X[3]) - X[2]
    f₃(X,p) = X[1] * X[2] - p[3] * X[3]
    f(X,p) = [f₁(X,p), f₂(X,p), f₃(X,p)]

    dX    = f(X, p)
    dXhat = Λhat + p[4] * (Xhat - X)

    # PCA adaptation law (forward-Euler sample-hold)
    if (floor(t / Tₛ) > floor((t - Δt) / Tₛ)) && (t >= Tₛ)
        gain  = p[4] / (1 - exp(p[4] * Tₛ))
        dΛhat = (gain * (Xhat - X) - Λhat) / Δt
    else
        dΛhat = zeros(3)
    end

    dZ[1:3] = dX
    dZ[4:6] = dXhat
    dZ[7:9] = dΛhat
end

function lorenz_solve()
    u0 = [10.0; 10.0; 10.0; 10.0; 10.0; 10.0; 0.0; 0.0; 0.0]
    tspan = (0.0, 5.0)
    prob = DE.ODEProblem(lorenz_PCA!, u0, tspan, p)
    sol = DE.solve(prob, DE.Euler(); dt = Δt, adaptive = false)
    return sol
end

function lorenz_plots()
    p1 = Plots.plot(sol, idxs = (1, 2, 3), legend = false, title = "Lorenz Attractor", xlabel = L"x", ylabel = L"y", zlabel = L"z", dpi=600);
    p2 = Plots.plot(sol, idxs = (4, 5, 6), legend = false, title = "Predictor", xlabel = L"\hat{x}", ylabel = L"\hat{y}", zlabel = L"\hat{z}", color = 2, dpi=600);

    StatePlot = Plots.plot(p1, p2, layout = (1, 2))

    # Prediction Error Plot

    ErrorPhasePlot = Plots.plot(sol, idxs = ((x1, x2, x3, xhat1, xhat2, xhat3) -> (xhat1 - x1, xhat2 - x2, xhat3 - x3), 1, 2, 3, 4, 5, 6), legend = false, title = "State Prediction Error" , xlabel = L"\tilde{x}", ylabel = L"\tilde{y}", zlabel = L"\tilde{z}", color = 3, dpi=600);
    ErrorNormPlot = Plots.plot(sol, idxs = ((t, x1, x2, x3, xhat1, xhat2, xhat3) -> (t, norm([xhat1 - x1, xhat2 - x2, xhat3 - x3])), 0, 1, 2, 3, 4, 5, 6), legend = false, title = "State Prediction Error Norm", xlabel = L"t", ylabel = L"\left\Vert \tilde{X} \right\Vert", color = 4, dpi=600);
    ErrorPlot = Plots.plot(ErrorPhasePlot, ErrorNormPlot, layout = (1, 2))

    ErrorStatePlot = Plots.plot(sol, idxs = ((t, x1, xhat1) -> (t, xhat1 - x1), 0, 1, 4), label = L"\tilde{x}", legend = :topright, xlabel = L"t", ylabel = L"\tilde{X}", color = 5, dpi=600);
    Plots.plot!(ErrorStatePlot, sol, idxs = ((t, x2, xhat2) -> (t, xhat2 - x2), 0, 2, 5), label = L"\tilde{y}", color = 6, dpi=600);
    Plots.plot!(ErrorStatePlot, sol, idxs = ((t, x3, xhat3) -> (t, xhat3 - x3), 0, 3, 6), label = L"\tilde{z}", color = 7, dpi=600)

    # Adaptive Estimation Plot

    AdEst1 = Plots.plot(sol, idxs = ((t, x1, x2, x3) -> (t, σ * (x2 - x1)), 0, 1, 2, 3),                              
        label = L"f_1(X)", color = 2, xlabel = L"t",  dpi=600);                                          
    Plots.plot!(AdEst1, sol, idxs = (0, 7), label = L"\hat{\Lambda}_1", color = 2, linestyle = :dash, dpi=600);

    AdEst2 = Plots.plot(sol, idxs = ((t, x1, x2, x3) -> (t, x1 * (p[2] - x3) - x2), 0, 1, 2, 3),                              
        label = L"f_2(X)", color = 6, xlabel = L"t",  dpi=600);                                          
    Plots.plot!(AdEst2, sol, idxs = (0, 8), label = L"\hat{\Lambda}_2", color = 6, linestyle = :dash, dpi=600);

    AdEst3 = Plots.plot(sol, idxs = ((t, x1, x2, x3) -> (t, x1 * x2 - p[3] * x3), 0, 1, 2, 3),                              
        label = L"f_3(X)", color = 7, xlabel = L"t",  dpi=600);                                          
    Plots.plot!(AdEst3, sol, idxs = (0, 9), label = L"\hat{\Lambda}_3", color = 7, linestyle = :dash, dpi=600);
    AdaptiveEstimationPlot = Plots.plot(AdEst1, AdEst2, AdEst3, layout = (3, 1))

    return StatePlot, ErrorPlot, ErrorStatePlot, AdaptiveEstimationPlot
end

sol = lorenz_solve()
StatePlot, ErrorPlot, ErrorStatePlot, AdaptiveEstimationPlot = lorenz_plots()

StatePlot
ErrorPlot
ErrorStatePlot
AdaptiveEstimationPlot

##############################################