## L1DRAC for a 1D Double Integrator
using L1DRAC
using CUDA
using LinearAlgebra
using Distributions
using ControlSystemsBase
using StaticArrays
using Plots
using JLD2
using DependencyAtlas
using DifferentialEquations

# %% [markdown]
# # L1DRAC for a 1D Double Integrator
# - $\frac{1}{2}$ 

###################################################################
## SYSTEM SETUP
###################################################################
function setup_system(; Ntraj=10) # Ntraj = number of trajectories for ensemble sims, default val 10
    # Simulation Parameters
    tspan = (0.0, 5.0)
    Δₜ = 1e-4 # Time step size
    Δ_saveat = 1e2 * Δₜ # Needs to be an integer multiple of Δₜ
    simulation_parameters = sim_params(tspan, Δₜ, Ntraj, Δ_saveat)

    # System Dimensions
    n, m, d = 2, 1, 2
    system_dimensions = sys_dims(n, m, d)

    # Double integrator dynamics
    A = @SMatrix [0.0 1.0; 0.0 0.0]
    B = @SMatrix [0.0; 1.0]

    # Baseline controller via pole placement
    λ = 10.0 # Stability margin
    sys = ss(A, B, SMatrix{2,2}(1.0I), 0.0)
    K = SMatrix{1,2}(place(sys, -λ * ones(2)))
    dp = (; K) # Dynamics params for GPU

    function baseline_input(t, x, dp) # Tracking controller
        r = @SVector [5*sin(t) + 3*cos(2*t), 0.0] # Reference trajectory
        return dp.K * (r - x)
    end

    # Nominal Vector Fields
    f(t, x, dp) = A * x + B * baseline_input(t, x, dp)
    g(t, x, dp) = @SVector [0.0, 1.0]
    g_perp(t, x, dp) = @SVector [1.0, 0.0]

    p_um(t, x, dp) = 2.0 * @SMatrix [0.01 0.1]
    p_m(t, x, dp) = 1.0 * @SMatrix [0.0 0.8]
    p(t, x, dp) = vcat(p_um(t, x, dp), p_m(t, x, dp))

    nominal_components = nominal_vector_fields(f, g, g_perp, p, dp)

    # Uncertain Vector Fields
    Λμ_um(t, x, dp) = 1e-2 * (1 + sin(x[1]))
    Λμ_m(t, x, dp) = 3.0 * (5 + 10*cos(x[2]) + 5*norm(x))
    Λμ(t, x, dp) = @SVector [Λμ_um(t, x, dp), Λμ_m(t, x, dp)]

    Λσ_um(t, x, dp) = 1e-1 * @SMatrix [0.1+cos(x[2]) 2.0]
    Λσ_m(t, x, dp) = 6 * @SMatrix [0.0 5+sin(x[2])+
                        5.0*(norm(x) < 1 ? norm(x) : sqrt(norm(x)))]
    Λσ(t, x, dp) = vcat(Λσ_um(t, x, dp), Λσ_m(t, x, dp))

    uncertain_components = uncertain_vector_fields(Λμ, Λσ)

    # Initial Distributions
    nominal_ξ₀ = MvNormal(20.0 * ones(2), 1e2 * I(2))
    true_ξ₀ = MvNormal(-2.0 * ones(2), 1e1 * I(2))
    initial_distributions = init_dist(nominal_ξ₀, true_ξ₀)

    # Define Systems
    nominal_system = nom_sys(system_dimensions, nominal_components, 
                        initial_distributions)
    true_system = true_sys(system_dimensions, nominal_components, 
                        uncertain_components, initial_distributions)

    # L1-DRAC Parameters (PLACEHOLDER values)
    ω = 50.0 # Filter bandwidth
    Tₛ = 10 * Δₜ # Sample time (integer multiple of Δₜ)
    λₛ = 100.0 # Predictor stability
    L1params = drac_params(ω, Tₛ, λₛ)

    return (
        simulation_parameters = simulation_parameters,
        nominal_system = nominal_system,
        true_system = true_system,
        L1params = L1params,
        system_dimensions = system_dimensions
    )
end

###################################################################
## MAIN
###################################################################
function main(; Ntraj = Int(1e1), max_GPUs=10, 
                        systems=[:nominal_sys, :true_sys, :L1_sys]) 

    @info "Warmup run for JIT compilation"
    println("=====================================") 
    warmup_setup = setup_system(Ntraj = 10)
    run_simulations(warmup_setup; max_GPUs=max_GPUs, systems=systems);

    println("=====================================")
    @info "Complete run for Ntraj=$Ntraj" 
    println("=====================================")
    setup = setup_system(; Ntraj = Ntraj)
    solutions = run_simulations(setup; max_GPUs=max_GPUs, systems=systems)
    return setup, solutions
end


###################################################################
## DATA LOGGING
###################################################################
# Save after a run — one direct call to the package's state_logging:
#
#   state_logging(sols; path="sol_logs/")                      # all three systems
#   state_logging(sols; systems=[:L1_sys], path="sol_logs/")   # a subset, same Symbols as run_simulations
#
# path options:
# 1. No path → sol_logs/ where Julia was launched (the state_logging default).
# 2. joinpath(@__DIR__, "sol_logs") → sol_logs/ next to this script — as written IN a script;
#    typed in the REPL, @__DIR__ gives the launch folder instead.
# 3. Your own string → anywhere you want (relative = from launch folder, absolute = exact).

# log_state_results: the runnable version of the call above, pinned to this script's folder
# (a function body does NOT execute at include time — only when you call it).
# Usage after a run:  log_state_results(sols)
function log_state_results(sols)
    state_logging(sols; path=joinpath(@__DIR__, "sol_logs"))
end


###################################################################
## PLOTS
###################################################################

# idxs — state selection for plot(sol) / plot(ensemble_sol), one grammar, three rules:
#
# Rule 1: integers name state components; 0 means time.
#   idxs=1 is X1, idxs=2 is X2, idxs=0 is t.
#
# Rule 2: a vector means "each of these vs time", overlaid as separate curves.
#   plot(sol; idxs=[1,2])   # X1 vs t AND X2 vs t (the default plots all states)
#   plot(sol; idxs=2)       # just X2 vs t
#
# Rule 3: a tuple means "pair these as axes" (parametric plot, time implicit).
#   plot(sol; idxs=(1,2))   # phase plot: X1 on x-axis, X2 on y-axis
#   plot(sol; idxs=(0,1))   # t vs X1 — explicit form of the default
#   plot(sol; idxs=(1,2,3)) # 3D phase trajectory (n >= 3 systems)
#
# Summary: list = overlay vs time, tuple = plot against each other.
# Plots sample the continuous interpolant, not just the saveat points.
# For this double integrator: idxs=(1,2) is the position-velocity phase portrait,
# one curve per trajectory.

# load_states_for_plotting: rebuild the plotting inputs from files saved by
# log_state_results — the recommended route when plotting only states. The L1 file
# holds the full extended state; component=:L1_sys_states slices to X while loading,
# so summaries and plots do no work on the predictor/filter/estimate blocks.
# Returns the same NamedTuple shape as main's solutions, so plot_states_vs_time
# accepts live and loaded inputs interchangeably.
# Arguments: system_dimensions (setup.system_dimensions); path (folder of saved files)
# Returns: (; nominal_sol, true_sol, L1_sol)
function load_states_for_plotting(system_dimensions; path=joinpath(@__DIR__, "sol_logs"))
    nominal_sol = load_ensemble(joinpath(path, "states_nominal.jld2"))
    true_sol    = load_ensemble(joinpath(path, "states_true.jld2"))
    L1_sol      = load_ensemble(joinpath(path, "states_L1.jld2"); component=:L1_sys_states, system_dimensions=system_dimensions) # Since we only want to plot the sattes, we load only the L1 state and not the full extended state which would lead to computing compute-heavy statistics that are not relevant to the plots. 
    return (; nominal_sol, true_sol, L1_sol)
end

# ensemble_stats: threaded per-timepoint statistics over the FULL ensemble.
# Threads.@threads over timepoints (independent work, needs julia -t N > 1);
# each iteration owns its buffer, sorts once per component, and reads the δ,
# 0.5 (= median exactly), and 1-δ quantiles off the sorted values — so all three
# quantile curves cost a single sort. k-th moment: (E[|Xi|^k])^(1/k), as in the plots.
# Arguments: esol (one EnsembleSolution); k (moment order); δ (quantile level);
#            components (state components to process, default: all)
# Returns: (; t, moment, med, qlow, qhigh) — each stat a Matrix [component, timepoint]
function ensemble_stats(esol; k=2, δ=0.05, components=eachindex(esol.u[1].u[1]))
    t = esol.u[1].t
    Ntraj = length(esol.u)
    nc, nt = length(components), length(t)
    moment = Matrix{Float64}(undef, nc, nt)
    med    = Matrix{Float64}(undef, nc, nt)
    qlow   = Matrix{Float64}(undef, nc, nt)
    qhigh  = Matrix{Float64}(undef, nc, nt)
    Threads.@threads for j in 1:nt
        vals = Vector{Float64}(undef, Ntraj)   # owned by this iteration — no sharing
        for (ci, i) in enumerate(components)
            for tr in 1:Ntraj
                vals[tr] = esol.u[tr].u[j][i]
            end
            moment[ci, j] = (sum(v -> abs(v)^k, vals) / Ntraj)^(1 / k)
            sort!(vals)
            qlow[ci, j]  = quantile(vals, δ;     sorted=true)
            med[ci, j]   = quantile(vals, 0.5;   sorted=true)
            qhigh[ci, j] = quantile(vals, 1 - δ; sorted=true)
        end
    end
    return (; t, moment, med, qlow, qhigh)
end

# plot_states_vs_time: 3x2 figure (columns = X1 | X2), all three systems overlaid
# per panel (nominal=13, true=7, L1=9):
#   row 1 — sample paths (states vs t), first min(max_traj_plot, Ntraj) paths only
#   row 2 — empirical k-th moments per component: (E[|Xi|^k])^(1/k), the L_k norm
#   row 3 — quantiles (median + [δ, 1-δ] band)
# Rows 2-3 statistics: ONE threaded ensemble_stats pass per system, all trajectories.
# Saved next to this script.
# Arguments: sol (solutions NamedTuple from main, e.g. sols); k (moment order, k=2 -> RMS);
#            δ (probability level: band spans the δ and 1-δ quantiles);
#            max_traj_plot (cap on sample paths drawn in row 1; rows 2-3 always use all);
#            fname (output file name)
# Returns: the figure
function plot_states_vs_time(sol; k=2, δ=0.05, max_traj_plot=500, fname="states_vs_time.png")
    
    # row 1 draws only the first min(max_traj_plot, Ntraj) sample paths
    paths(esol) = 1:min(max_traj_plot, length(esol.u))

    p1 = plot(sol.nominal_sol; idxs=1, trajectories=paths(sol.nominal_sol), color=13,
        linewidth=0.5, linealpha=0.3, title="X1")
    plot!(p1, sol.true_sol; idxs=1, trajectories=paths(sol.true_sol), color=7,
        linewidth=0.5, linealpha=0.3)
    plot!(p1, sol.L1_sol; idxs=1, trajectories=paths(sol.L1_sol), color=9,
        linewidth=0.5, linealpha=0.3)
    
    p2 = plot(sol.nominal_sol; idxs=2, trajectories=paths(sol.nominal_sol), color=13,
        linewidth=0.5, linealpha=0.3, title="X2")
    plot!(p2, sol.true_sol; idxs=2, trajectories=paths(sol.true_sol), color=7,
        linewidth=0.5, linealpha=0.3)
    plot!(p2, sol.L1_sol; idxs=2, trajectories=paths(sol.L1_sol), color=9,
        linewidth=0.5, linealpha=0.3)
    
    # one threaded ensemble_stats pass per system over ALL trajectories:
    # k-th moments (row 2) + median and [δ, 1-δ] quantiles (row 3), states only
    st_nom = ensemble_stats(sol.nominal_sol; k=k, δ=δ, components=1:2)
    st_tru = ensemble_stats(sol.true_sol;    k=k, δ=δ, components=1:2)
    st_L1  = ensemble_stats(sol.L1_sol;      k=k, δ=δ, components=1:2)

    # row 2 — empirical k-th moments
    p3 = plot(st_nom.t, st_nom.moment[1, :]; color=13, lw=1.5, label=false, ylabel="Moment: $k")
    plot!(p3, st_tru.t, st_tru.moment[1, :]; color=7, lw=1.5, label=false)
    plot!(p3, st_L1.t, st_L1.moment[1, :]; color=9, lw=1.5, label=false)

    p4 = plot(st_nom.t, st_nom.moment[2, :]; color=13, lw=1.5, label=false, ylabel="Moment: $k")
    plot!(p4, st_tru.t, st_tru.moment[2, :]; color=7, lw=1.5, label=false)
    plot!(p4, st_L1.t, st_L1.moment[2, :]; color=9, lw=1.5, label=false)

    # row 3 — median line + [δ, 1-δ] quantile band; ribbon takes DISTANCES from
    # the center line, hence med - qlow and qhigh - med
    band(st, i) = (st.med[i, :] .- st.qlow[i, :], st.qhigh[i, :] .- st.med[i, :])

    p5 = plot(st_nom.t, st_nom.med[1, :]; ribbon=band(st_nom, 1), color=13, lw=1.5,
        fillalpha=0.2, label=false, xlabel="t", ylabel="Quantiles: [$δ, $(1 - δ)]")
    plot!(p5, st_tru.t, st_tru.med[1, :]; ribbon=band(st_tru, 1), color=7, lw=1.5,
        fillalpha=0.2, label=false)
    plot!(p5, st_L1.t, st_L1.med[1, :]; ribbon=band(st_L1, 1), color=9, lw=1.5,
        fillalpha=0.2, label=false)

    p6 = plot(st_nom.t, st_nom.med[2, :]; ribbon=band(st_nom, 2), color=13, lw=1.5,
        fillalpha=0.2, label=false, xlabel="t", ylabel="Quantiles: [$δ, $(1 - δ)]")
    plot!(p6, st_tru.t, st_tru.med[2, :]; ribbon=band(st_tru, 2), color=7, lw=1.5,
        fillalpha=0.2, label=false)
    plot!(p6, st_L1.t, st_L1.med[2, :]; ribbon=band(st_L1, 2), color=9, lw=1.5,
        fillalpha=0.2, label=false)

    fig = plot(p1, p2, p3, p4, p5, p6; layout=(3, 2), size=(900, 1100))
    savefig(fig, joinpath(@__DIR__, fname))
    return fig
end

## TO BE DEPRECATED 
include("plotting_utils.jl")

function generate_state_plots(; path=joinpath(@__DIR__, "sol_logs"), max_traj=500)
    nom = load(joinpath(path, "states_nominal.jld2"))
    tru = load(joinpath(path, "states_true.jld2"))
    L1  = load(joinpath(path, "states_L1.jld2"))

    fig = plot_results(nom, tru, L1; max_traj=max_traj)
    savefig(fig, joinpath(@__DIR__, "states_plot.png"))
    @info "Saved states_plot.png"
    return fig
end

