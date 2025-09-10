# Plotting functions for example 1# Plot Functions
using Plots
using LaTeXStrings
import DifferentialEquations as DE

###############################################
## Multiple Dispatch
# Single system, single sample path
function simplot(sol::RODESolution; xlabelstring::LaTeXString, ylabelstring::LaTeXString)
	l = @layout [a; b]
    ###
	PositionPhasePlot = plot(sol[1,:], sol[2,:], linewidth = 2, label=false, xlabel = xlabelstring, ylabel = ylabelstring )
	###
	PositionTimePlot = plot(sol.t, sol[1,:], color=1, linewidth = 2, label=xlabelstring)
	plot!(PositionTimePlot, sol.t, sol[2,:], color =2, linewidth = 2, label=ylabelstring)
    ###
	mainplot = plot(PositionPhasePlot, PositionTimePlot; layout = l, size=(600,800))
	return mainplot
end
# comparison of systems, single sample path
function simplot(sol1::RODESolution, sol2::RODESolution; labelstring1::LaTeXString, labelstring2::LaTeXString)
	l = @layout [a; b; c]
    ###
	PositionPhasePlotTime = plot(sol1[1,:], sol1[2,:], sol1.t, color=1, linewidth = 2, linestyle = :dash, label=false)
    plot!(PositionPhasePlotTime, sol2[1,:], sol2[2,:], sol2.t, color=1, linewidth = 2, label=false, zlabel = L"Time~\rightarrow")
    ###
    PositionPhasePlot = plot(sol1[1,:], sol1[2,:], color=1, linewidth = 2, linestyle = :dash, label=labelstring1)
    plot!(PositionPhasePlot, sol2[1,:], sol2[2,:], color=1, linewidth = 2, label=labelstring2)
	###
	PositionTimePlot = plot(sol1.t, sol1[1,:], color=1, linewidth = 2, linestyle=:dash, label=false)
	plot!(PositionTimePlot, sol1.t, sol1[2,:], color=2, linewidth = 2, linestyle=:dash, label=false)
    plot!(PositionTimePlot, sol2.t, sol2[1,:], color=1, linewidth = 2, label=false)
	plot!(PositionTimePlot, sol2.t, sol2[2,:], color=2, linewidth = 2, label=false, xlabel = L"Time~\rightarrow")
    ###
	mainplot = plot(PositionPhasePlotTime, PositionPhasePlot, PositionTimePlot; layout = l, size=(600,1200))
	return mainplot
end
# comparison of ALL systems, single sample path
function simplot(sol1::RODESolution, sol2::RODESolution, sol3::RODESolution; labelstring1::String, labelstring2::String, labelstring3::LaTeXString)
	l = @layout [a; b; c]
	lw = 2.5
	lα = 0.6
    ###
    PositionPhasePlot = plot(sol1[1,:], sol1[2,:], color=13, linewidth = lw, linealpha = lα, label=labelstring1)
    plot!(PositionPhasePlot, sol2[1,:], sol2[2,:], color=7, linewidth = lw, linealpha = lα, label=labelstring2)
	plot!(PositionPhasePlot, sol3[1,:], sol3[2,:], color=25, linewidth = lw, linealpha = lα, label=labelstring3)
	###
	PositionTimePlot1 = plot(sol1.t, sol1[1,:], color=13, linewidth = lw, linealpha = lα, label=false)
    plot!(PositionTimePlot1, sol2.t, sol2[1,:], color=7, linewidth = lw, linealpha = lα, label=false)
	plot!(PositionTimePlot1, sol3.t, sol3[1,:], color=25, linewidth = lw, linealpha = lα, label=false, xlabel = L"Time~\rightarrow", title = L"X_{t,1}")
	###
	PositionTimePlot2 = plot(sol1.t, sol1[2,:], color=13, linewidth = lw, linealpha = lα, label=false)
	plot!(PositionTimePlot2, sol2.t, sol2[2,:], color=7, linewidth = lw, linealpha = lα, label=false)
	plot!(PositionTimePlot2, sol3.t, sol3[2,:], color=25, linewidth = lw, linealpha = lα, label=false, xlabel = L"Time~\rightarrow", title = L"X_{t,2}")
    ###
	mainplot = plot(PositionPhasePlot, PositionTimePlot1, PositionTimePlot2; layout = l, size=(600,1200))
	trackingerrorplot = plot(sol1.t, abs.(sol1[1,:]-sol2[1,:]), color=7, linewidth = lw, linealpha = lα, label=false, xlabel = L"Time~\rightarrow", title = L"|X^\star_{t,1}-X_{t,1}|")
	return mainplot
end
# Single system, Ensemble
function simplot(sol::EnsembleSolution, xlabelstring::LaTeXString, ylabelstring::LaTeXString)
	lw = 1.5 # linewidth
	lα = 0.1 # linealpha
    l = @layout [a; b]
    ###
	PositionPhasePlot = plot(sol[1][1,:], sol[1][2,:], linewidth = lw, linealpha = lα, label=false, xlabel = xlabelstring, ylabel = ylabelstring )
    for i = 2:Main.Ntraj
		plot!(PositionPhasePlot, sol[i][1,:], sol[i][2,:], color=1, linewidth = lw, linealpha = lα,  label=false)
	end
	###
	PositionTimePlot = plot(sol[1].t, sol[1][1,:], color=1, linewidth = lw, linealpha = lα, label=xlabelstring)
	plot!(PositionTimePlot, sol[1].t, sol[1][2,:], color =2, linewidth = lw, linealpha = lα, label=ylabelstring)
    for j = 1:Main.n
		for i = 2:Main.Ntraj
			plot!(PositionTimePlot, sol[i].t, sol[i][j,:], color=j, linewidth = lw, linealpha = lα,  label=false)
		end
	end
    ###
	mainplot = plot(PositionPhasePlot, PositionTimePlot; layout = l, size=(600,800))
	return mainplot
end
# comparison of systems, Ensemble
function simplot(sol1::EnsembleSolution, sol2::EnsembleSolution; labelstring1::LaTeXString, labelstring2::LaTeXString)
    lw = 1.5 # linewidth
	lα = 0.1 # linealpha
	l = @layout [a; b]
    ###
    PositionPhasePlot = plot(sol1[1][1,:], sol1[1][2,:], color=1, linewidth = lw, linealpha = lα, linestyle = :dash, label=labelstring1)
    plot!(PositionPhasePlot, sol2[1][1,:], sol2[1][2,:], color=2, linewidth = lw, linealpha = lα, label=labelstring2)
    for i = 2:Main.Ntraj
		plot!(PositionPhasePlot, sol1[i][1,:], sol1[i][2,:], color=1, linewidth = lw, linealpha = lα,  label=false)
        plot!(PositionPhasePlot, sol2[i][1,:], sol2[i][2,:], color=2, linewidth = lw, linealpha = lα,  label=false)
	end
	###
	PositionTimePlot = plot(sol1[1].t, sol1[1][1,:], color=1, linewidth = lw, linealpha = lα, label=false)
	plot!(PositionTimePlot, sol1[1].t, sol1[1][2,:], color=1, linewidth = lw, linealpha = lα, label=false)
    plot!(PositionTimePlot, sol2[1].t, sol2[1][1,:], color=2, linewidth = lw, linealpha = lα, label=false)
	plot!(PositionTimePlot, sol2[1].t, sol2[1][2,:], color=2, linewidth = lw, linealpha = lα, label=false, xlabel = L"Time~\rightarrow")
    for j = 1:Main.n
		for i = 2:Main.Ntraj
			plot!(PositionTimePlot, sol1[i].t, sol1[i][j,:], color=1, linewidth = lw, linealpha = lα,  label=false)
            plot!(PositionTimePlot, sol2[i].t, sol2[i][j,:], color=2, linewidth = lw, linealpha = lα,  label=false)
		end
	end
    ###
	mainplot = plot(PositionPhasePlot, PositionTimePlot; layout = l, size=(600,800))
	return mainplot
end
###############################################
function predictorplot(sol::RODESolution; kwargs...)
	lα = 0.7
    ### LEGEND ###
	# system state = sol[1:n,:]
	# predictor state = sol[n+1:2n,:]
	##############
	PositionTimePlot = plot(sol.t, sol[1,:], color=1, linewidth = 2, label = false, linealpha = lα)
	labelstring1 = L"X_{t}"
	labelstring2 = L"predictor~\hat{X}_{t}"
	for i ∈ 2:n
		i == n ? plot!(PositionTimePlot, sol.t, sol[i,:], color=1, linewidth = 2, linealpha = lα, label = labelstring1) : plot!(PositionTimePlot, label = false) : plot!(PositionTimePlot, sol.t, sol[i,:], color=1, linewidth = 2, linealpha = lα)
	end
	for i ∈ n+1:2n
		i == 2n ? plot!(PositionTimePlot, sol.t, sol[i,:], color=2, linewidth = 3, label = false, linestyle = :dash, linealpha = lα) : plot!(PositionTimePlot, sol.t, sol[i,:], color=2, linewidth = 3, label = labelstring2, linestyle = :dash, linealpha = lα, xlabel = L"Time~\rightarrow")
	end
	if haskey(kwargs, :predictor_mode) && kwargs[:predictor_mode] == :test
        plot!(PositionTimePlot, title = "Predictor Test")
    elseif haskey(kwargs, :predictor_mode) && kwargs[:predictor_mode] == :performance
        plot!(PositionTimePlot, title = "Predictor Quality")
    end
	return PositionTimePlot
end
###############################################

## ENSEMBLE AND PATH PLOTS IN EXAMPLE 1 ##
function plotfunc(filename)
    l = @layout [a b; c d]
    nom_summ = DE.EnsembleSummary(ens_nom_sol)
    tru_summ = DE.EnsembleSummary(ens_tru_sol)
    L1_summ = DE.EnsembleSummary(ens_L1_sol)
    p1=plot(); p2=plot(); p3=plot(); p4=plot();
    for i ∈ 1:Ntraj
        plot!(p1, ens_nom_sol[i], idxs = (0,1), lw=2, lalpha=0.1, color = 13, label = :false)
        plot!(p1, ens_tru_sol[i], idxs = (0,1), lw=2, lalpha=0.1, color = 7, label = :false)
        plot!(p1, ens_L1_sol[i], idxs = (0,1), lw=2, lalpha=0.1, color = 25, label = :false, title = L"X_1-Nom(Blue), Tru(Orange) \mathcal{L}_1(Green)")

        plot!(p2, ens_nom_sol[i], idxs = (0,2), lw=2, lalpha=0.1, color = 13, label = :false)
        plot!(p2, ens_tru_sol[i], idxs = (0,2), lw=2, lalpha=0.1, color = 7, label = :false)
        plot!(p2, ens_L1_sol[i], idxs = (0,2), lw=2, lalpha=0.1, color = 25, label = :false, title = L"X_2")
    end
    plot!(p3, nom_summ, idxs = 1, color = 13, lw = 1, fillalpha = 0.2)
    plot!(p3, tru_summ, idxs = 1, color = 7, lw = 1, fillalpha = 0.2)
    plot!(p3, L1_summ, idxs = 1, color = 25, lw = 1, fillalpha = 0.2, xlabel = "t")

    plot!(p4, nom_summ, idxs = 2, color = 13, lw = 1, fillalpha = 0.2)
    plot!(p4, tru_summ, idxs = 2, color = 7, lw = 1, fillalpha = 0.2)
    plot!(p4, L1_summ, idxs = 2, color = 25, lw = 1, fillalpha = 0.2, xlabel= "t")

    finalplot = plot(p1, p2, p3, p4, layout = l, size = (650, 650))
	savefig(finalplot, filename)
end
