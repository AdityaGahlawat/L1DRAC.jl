# Plotting functions for example 1# Plot Functions
using Plots
using LaTeXStrings

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
# Single system, Ensemble
function simplot(sol::EnsembleSolution; xlabelstring::LaTeXString, ylabelstring::LaTeXString)
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
function predictorplot(sol::RODESolution; labelstring1::LaTeXString, labelstring2::LaTeXString)
	l = @layout [a; b]
	lα = 0.7
    ### LEGEND ###
	# system state = sol[1:n,:]
	# predictor state = sol[n+1:2n,:]
	##############
    PositionPhasePlot = plot(sol[1,:], sol[2,:], color=1, linewidth = 2, label = labelstring1, linealpha = lα)
    plot!(PositionPhasePlot, sol[3,:], sol[4,:], color=2, linewidth = 2, label = labelstring2, linealpha = lα)
	# ###
	PositionTimePlot = plot(sol.t, sol[1,:], color=1, linewidth = 2, label = false, linealpha = lα)
	plot!(PositionTimePlot, sol.t, sol[2,:], color=1, linewidth = 2, label = labelstring1, linealpha = lα)
    plot!(PositionTimePlot, sol.t, sol[3,:], color=2, linewidth = 3, label = false, linestyle = :dash, linealpha = lα)
	plot!(PositionTimePlot, sol.t, sol[4,:], color=2, linewidth = 3, label = labelstring2, linestyle = :dash, linealpha = lα, xlabel = L"Time~\rightarrow")
    # ###
	mainplot = plot(PositionPhasePlot, PositionTimePlot; layout = l, size=(600,800))
	return mainplot
end
