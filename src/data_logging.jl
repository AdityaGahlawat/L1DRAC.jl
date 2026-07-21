# data_logging.jl: Save ensemble simulation solutions to JLD2 files
#
# All systems (nominal, true, L1) save files with an identical two-key schema:
#   {t, u}
#   - t: Vector of time points, shared across trajectories
#   - u: Vector of trajectories, each trajectory a Vector of state vectors,
#        saved verbatim (no slicing, no statistics stored)
#
# For the L1 system, u holds the FULL extended state at each time point:
#   [X, Xhat, Xfilter, Λhat]  (length 3n+m)
# It is saved whole; the slice to X (first n components) is deferred to
# rebuild time and is NOT applied here.
#
# Statistics (mean, variance) and analysis objects are NOT stored. They are
# reconstructed at use time from the saved t and u arrays, via load_ensemble.
#
# HELPER FUNCTIONS:
#   _process_solution(sol)             - Process nominal/true into (t, u)
#   _process_solution_L1(sol, system_dimensions) - Process L1 into (t, u) with full extended state
#
# MAIN FUNCTION:
#   state_logging(system_dimensions; sol_nominal, sol_true, sol_L1, path)
#     - Saves each non-nothing solution to a JLD2 file holding {t, u}
#     - Returns named tuple of file paths
#
# LOAD FUNCTION:
#   load_ensemble(path; components) - Rebuild a toolbox-native EnsembleSolution from a
#     saved {t, u} file. Optional components slice (e.g. 1:n) selects state components AT
#     rebuild; default rebuilds on all components. L1 state analysis uses components=1:n (X).


# _process_solution: Process nominal or true system solution into (t, u) structure
# Input: a single EnsembleSolution (trajectories saved verbatim; no statistics computed or stored here)
# Returns: tuple (t, u) for JLD2 saving
#   - t: time points, shared across trajectories (taken from the first trajectory)
#   - u: vector of trajectories, each trajectory a vector of state vectors
function _process_solution(sol)
    t = sol.u[1].t
    u = [traj.u for traj in sol]
    return (t, u)
end


# _process_solution_L1: Process L1 system solution into (t, u) structure
# Input: a single EnsembleSolution of extended states [X, Xhat, Xfilter, Λhat] (length 3n+m),
#        saved verbatim; no slicing and no statistics computed or stored here
#        system_dimensions (SysDims struct) is retained but unused (see below)
# Returns: tuple (t, u) for JLD2 saving
#   - t: time points, shared across trajectories (taken from the first trajectory)
#   - u: vector of trajectories, each trajectory a vector of full extended state vectors
# Slice to X (first n components) happens at rebuild time (load_ensemble), not here
function _process_solution_L1(sol, system_dimensions)
    # system_dimensions unused: full extended state saved verbatim; slice to X moved to rebuild
    t = sol.u[1].t
    u = [traj.u for traj in sol]
    return (t, u)
end


# state_logging: Save ensemble simulation solutions to JLD2 files
# Each provided system is saved with the identical two-key {t, u} schema
#
# Arguments:
#   system_dimensions: SysDims struct containing n, m, d
#   sol_nominal: (kwarg) EnsembleSolution for nominal system
#   sol_true:    (kwarg) EnsembleSolution for true system
#   sol_L1:      (kwarg) EnsembleSolution for L1 system (full extended state)
#   path:        (kwarg) Output directory (created if doesn't exist)
#
# Returns named tuple of file paths: (nominal=..., true_sys=..., L1=...)
function state_logging(system_dimensions; sol_nominal=nothing, sol_true=nothing, sol_L1=nothing, path::String="sol_logs/")
    mkpath(path)

    nominal_path = nothing
    true_path = nothing
    L1_path = nothing

    # Save nominal solution with the {t, u} schema
    if sol_nominal !== nothing
        t, u = _process_solution(sol_nominal)
        nominal_path = joinpath(path, "states_nominal.jld2")
        jldsave(nominal_path; t, u)
    end

    # Save true system solution with the {t, u} schema
    if sol_true !== nothing
        t, u = _process_solution(sol_true)
        true_path = joinpath(path, "states_true.jld2")
        jldsave(true_path; t, u)
    end

    # Save L1 solution: full extended state saved verbatim (slice to X deferred to rebuild)
    if sol_L1 !== nothing
        t, u = _process_solution_L1(sol_L1, system_dimensions)
        L1_path = joinpath(path, "states_L1.jld2")
        jldsave(L1_path; t, u)
    end

    return (
        nominal = nominal_path,
        true_sys = true_path,
        L1 = L1_path
    )
end


# load_ensemble: rebuild a toolbox-native EnsembleSolution from a saved {t, u} JLD2 file.
# Path-B reconstruction (save-design-brief.md 2.2a): each saved trajectory is rebuilt into a
# genuine per-trajectory SciML solution (build_solution over a stub SDEProblem), so the FULL
# ensemble toolbox works on the result -- EnsembleSummary (on-grid AND off-grid), every
# EnsembleAnalysis timestep_*/timepoint_* statistic, and both Plots recipes -- in a fresh
# session with NO re-simulation. Off-grid queries stay correct because each trajectory is a
# real SciMLSolution carrying a LinearInterpolation over its own (t, u_i), so an off-grid
# EnsembleSummary(sim, ts) takes the interpolation branch (length-consistent by construction),
# never the DiffEqArray silent-corruption branch (brief 2.3).
#
# Arguments:
#   path       : path to a states_*.jld2 file written by state_logging (keys {t, u})
#   components : (kwarg) which state-vector components to rebuild on.
#                nothing (default) -> all components; saved state vectors used verbatim.
#                a range / index collection (e.g. 1:n) -> each state vector is sliced to those
#                components BEFORE its trajectory is built (A3), so the returned ensemble IS the
#                sub-state ensemble. L1 state analysis loads with components=1:n against a full
#                extended-state L1 file, and EnsembleSummary then runs directly on X.
# Returns: an EnsembleSolution of genuine per-trajectory solutions.
#
# In-module name resolution (save-design-brief.md D4, VBI-verified):
#   load                               - bare, via `using JLD2`
#   SDEProblem / EM / EnsembleSolution - bare, re-exported by `using DifferentialEquations`
#   build_solution / LinearInterpolation / ReturnCode - NOT bare in this module; reached as
#                                        DifferentialEquations.SciMLBase.*
function load_ensemble(path::AbstractString; components=nothing)
    data = load(path)
    t = data["t"]
    u = data["u"]

    # Stub drift/diffusion: never evaluated by the analysis (build_solution only accesses
    # prob.u0 / prob.f / parameter fields); zero(u) is type-stable on every path.
    fstub(uu, p, tt) = zero(uu)
    gstub(uu, p, tt) = zero(uu)

    trajs = map(u) do u_i
        # Component selection applied BEFORE building the trajectory (A3): slice each state
        # vector so the rebuilt trajectory carries only the requested components.
        u_sel = components === nothing ? u_i : [state[components] for state in u_i]
        prob = SDEProblem(fstub, gstub, u_sel[1], (t[1], t[end]))
        DifferentialEquations.SciMLBase.build_solution(
            prob, EM(), t, u_sel;
            interp = DifferentialEquations.SciMLBase.LinearInterpolation(t, u_sel),
            calculate_error = false,
            retcode = DifferentialEquations.SciMLBase.ReturnCode.Success)
    end

    return EnsembleSolution(trajs, 0.0, true)
end
