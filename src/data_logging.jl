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
#   _process_solution_L1(sol) - Process L1 into (t, u) with full extended state
#
# MAIN FUNCTION:
#   state_logging(solutions; systems=[:nominal_sys, :true_sys, :L1_sys], path)
#     - Saves each requested (in `systems`) non-nothing solution to a JLD2 file holding {t, u}
#     - Returns named tuple of file paths
#
# LOAD FUNCTION:
#   load_ensemble(path; component, system_dimensions) - Rebuild a toolbox-native
#     EnsembleSolution from a saved {t, u} file. `component` (a Symbol, package idiom) selects
#     one L1 extended-state block AT rebuild; requires `system_dimensions` (a SysDims) to resolve
#     its range. Valid L1-only selectors: :L1_sys_states (X), :L1_predictor (Xhat),
#     :L1_filter (Xfilter), :L1_adaptive_estimate (Λhat). Default (no component) rebuilds the
#     file verbatim: nominal/true as-is, L1 on its full extended state.


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
# Returns: tuple (t, u) for JLD2 saving
#   - t: time points, shared across trajectories (taken from the first trajectory)
#   - u: vector of trajectories, each trajectory a vector of full extended state vectors
# Slice to X (first n components) happens at rebuild time (load_ensemble), not here
function _process_solution_L1(sol)
    t = sol.u[1].t
    u = [traj.u for traj in sol]
    return (t, u)
end


# state_logging: Save ensemble simulation solutions to JLD2 files
# Each saved system is written with the identical two-key {t, u} schema
#
# Arguments:
#   solutions: (positional) the NamedTuple returned by run_simulations, with fields
#              nominal_sol, true_sol, L1_sol (each an EnsembleSolution or nothing)
#   systems:   (kwarg) Vector{Symbol} of systems to save; default = all three.
#              Valid entries: :nominal_sys, :true_sys, :L1_sys.
#              A system is written only if it is BOTH listed in `systems` AND its
#              solution field is non-nothing; a nothing field is silently skipped.
#   path:      (kwarg) Output directory (created if it doesn't exist)
#
# Symbol -> field -> file mapping:
#   :nominal_sys -> solutions.nominal_sol -> states_nominal.jld2 (via _process_solution)
#   :true_sys    -> solutions.true_sol    -> states_true.jld2    (via _process_solution)
#   :L1_sys      -> solutions.L1_sol      -> states_L1.jld2      (via _process_solution_L1,
#                   full extended state saved verbatim)
#
# Returns named tuple of file paths: (nominal=..., true_sys=..., L1=...), with
# nothing in any slot that was not saved.
function state_logging(solutions; systems::Vector{Symbol}=[:nominal_sys, :true_sys, :L1_sys], path::String="sol_logs/")
    # Validate systems kwarg (mirrors run_simulations)
    valid_systems = [:nominal_sys, :true_sys, :L1_sys]
    for s in systems
        s ∈ valid_systems || error("Invalid system: $s. Valid options: $valid_systems")
    end
    isempty(systems) && error("Must specify at least one system to log")

    mkpath(path)

    nominal_path = nothing
    true_path = nothing
    L1_path = nothing

    # Save nominal solution with the {t, u} schema
    if :nominal_sys ∈ systems && solutions.nominal_sol !== nothing
        t, u = _process_solution(solutions.nominal_sol)
        nominal_path = joinpath(path, "states_nominal.jld2")
        jldsave(nominal_path; t, u)
    end

    # Save true system solution with the {t, u} schema
    if :true_sys ∈ systems && solutions.true_sol !== nothing
        t, u = _process_solution(solutions.true_sol)
        true_path = joinpath(path, "states_true.jld2")
        jldsave(true_path; t, u)
    end

    # Save L1 solution: full extended state saved verbatim (slice to X deferred to rebuild)
    if :L1_sys ∈ systems && solutions.L1_sol !== nothing
        t, u = _process_solution_L1(solutions.L1_sol)
        L1_path = joinpath(path, "states_L1.jld2")
        jldsave(L1_path; t, u)
    end

    return (
        nominal = nominal_path,
        true_sys = true_path,
        L1 = L1_path
    )
end


# load_ensemble: rebuild a ready-to-use EnsembleSolution from a saved {t, u} JLD2 file.
#
# PRIMARY USE — any saved file (nominal, true, or L1), no keywords:
#
#   load_ensemble("sol_logs/states_nominal.jld2")   # loaded verbatim
#                                                   # (an L1 file: full extended state)
#
#   Fresh session, no re-simulation — EnsembleSummary, plot, and the ensemble
#   statistics work directly on the result.
#
# OPTIONAL — L1 files only: slice one block of the extended state while loading.
# Needs system_dimensions, since saved files carry no metadata:
#
#   load_ensemble("sol_logs/states_L1.jld2"; component=:L1_predictor, system_dimensions=dims)
#
#   :L1_sys_states        → X        (1:n)
#   :L1_predictor         → Xhat     (n+1:2n)
#   :L1_filter            → Xfilter  (2n+1:2n+m)
#   :L1_adaptive_estimate → Λhat     (2n+m+1:3n+m)
#
# Guards, each a loud error: unknown component; component without system_dimensions;
# L1 selector aimed at a nominal/true file (state length != 3n+m).
#
# Rebuild detail: every saved trajectory becomes a per-trajectory solution object with
# linear interpolation over t, so on- and off-grid queries behave like a fresh solve.
function load_ensemble(path::AbstractString; component=nothing, system_dimensions=nothing)
    data = load(path)
    t = data["t"]
    u = data["u"]

    # Resolve the named selector to a concrete index range ONCE (not per trajectory).
    # components stays nothing on the default path -> state vectors used verbatim.
    valid_components = (:L1_sys_states, :L1_predictor, :L1_filter, :L1_adaptive_estimate)
    components = nothing
    if component !== nothing
        # Guard 1: only the four L1 selectors are valid.
        component in valid_components || error(
            "load_ensemble: invalid component $(repr(component)). Valid selectors are " *
            "$(valid_components) (all L1-only), or omit `component` to load the file verbatim.")
        # Guard 2: a selector needs the system dimensions to resolve to a range.
        system_dimensions !== nothing || error(
            "load_ensemble: component=$(repr(component)) requires `system_dimensions` " *
            "(a SysDims) to resolve its index range; none was given.")
        @unpack n, m = system_dimensions
        # Guard 3: L1 selectors address the extended state, so they apply only to a 3n+m file.
        # Catches an L1 selector aimed at a nominal/true file. Assumes a non-empty file, as the
        # rebuild below already does (u[1][1] is the first state vector of the first trajectory).
        state_len = length(u[1][1])
        state_len == 3n + m || error(
            "load_ensemble: component=$(repr(component)) is an L1 extended-state selector, but " *
            "the file at $(path) has state length $(state_len) != 3n+m = $(3n + m). " *
            "L1 selectors apply only to L1 files, not nominal/true files.")
        components =
            component === :L1_sys_states ? (1:n) :
            component === :L1_predictor  ? (n+1:2n) :
            component === :L1_filter     ? (2n+1:2n+m) :
                                           (2n+m+1:3n+m)
    end

    # Stub drift/diffusion: never evaluated by the analysis (build_solution only accesses
    # prob.u0 / prob.f / parameter fields); zero(u) is type-stable on every path.
    fstub(uu, p, tt) = zero(uu)
    gstub(uu, p, tt) = zero(uu)

    trajs = map(u) do u_i
        # Component selection applied BEFORE building the trajectory (A3): slice each state
        # vector to the resolved range so the rebuilt trajectory carries only that block.
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
