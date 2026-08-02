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


# _input_ensemble: rebuild a toolbox-native EnsembleSolution from an already-computed
# (t, u) set of input trajectories, using the same rebuild sequence as load_ensemble.
#
# Arguments:
#   t: time points, shared across trajectories (the same shared grid the saved files use)
#   u: vector of trajectories, each trajectory a vector of m-dimensional input values,
#      used verbatim — no slicing, no conversion, any m >= 1, any value container
#
# Returns: ONE EnsembleSolution whose per-trajectory solution objects carry linear
# interpolation over t, so EnsembleSummary, the plot recipes, and on/off-grid queries
# behave exactly as they do on a freshly solved ensemble.
#
# Private helper: the values are already in memory, so this reads no file and applies no
# component selection. Argument validation is the caller's job, not this function's.
function _input_ensemble(t, u)
    # Stub drift/diffusion: never evaluated by the analysis (build_solution only accesses
    # prob.u0 / prob.f / parameter fields); zero(uu) is type-stable on every path.
    fstub(uu, p, tt) = zero(uu)
    gstub(uu, p, tt) = zero(uu)

    trajs = map(u) do u_i
        prob = SDEProblem(fstub, gstub, u_i[1], (t[1], t[end]))
        DifferentialEquations.SciMLBase.build_solution(
            prob, EM(), t, u_i;
            interp = DifferentialEquations.SciMLBase.LinearInterpolation(t, u_i),
            calculate_error = false,
            retcode = DifferentialEquations.SciMLBase.ReturnCode.Success)
    end

    return EnsembleSolution(trajs, 0.0, true)
end


# _baseline_trajectories: evaluate the baseline control input along every trajectory of an
# ensemble of state trajectories, producing input trajectories in the (t, u) shape.
#
# Arguments:
#   esol:           EnsembleSolution of state trajectories — a live solve or a load_ensemble result
#   baseline_input: the baseline control function, called as baseline_input(t, x, dp)
#   dp:             dynamics parameters, handed to baseline_input untouched
#   n:              state dimension; used only on the L1 path, to take the X block
#   is_L1:          false -> x is the whole saved state vector (nominal/true, length n)
#                   true  -> x is state[1:n], the X block of the L1 extended state
#                            [X, Xhat, Xfilter, Λhat] (length 3n+m)
#
# Returns: tuple (t, u), exactly the shape _input_ensemble consumes
#   - t: time points, shared across trajectories (taken from the first trajectory)
#   - u: vector of trajectories, each a vector of the m-dimensional values baseline_input
#        returned, stored verbatim — no conversion, no check of the returned length, any m >= 1
#
# Private helper: a plain walk-and-evaluate. It holds NO guards — the dimension and selection
# checks (including the L1 state-length rule) live in the exported orchestrator that calls it.
function _baseline_trajectories(esol, baseline_input, dp, n, is_L1)
    t = esol.u[1].t
    # Outer: one entry per trajectory. Inner: one input value per timepoint of the shared grid.
    u = [[baseline_input(t[j], is_L1 ? traj.u[j][1:n] : traj.u[j], dp) for j in eachindex(t)]
         for traj in esol]
    return (t, u)
end


# _adaptive_trajectories: extract the L1 adaptive control input along every trajectory of an
# ensemble of L1 extended-state trajectories, producing input trajectories in the (t, u) shape.
#
# The adaptive input is not recomputed here — it is already carried in the saved state. The L1
# extended state is [X, Xhat, Xfilter, Λhat] (length 3n+m), and the adaptive input is the
# NEGATED filter block, u_a = -Xfilter = -state[2n+1:2n+m].
#
# Arguments:
#   esol: EnsembleSolution of L1 extended-state trajectories — a live solve or a load_ensemble
#         result loaded verbatim (i.e. WITHOUT a component selection, so the full 3n+m state)
#   n:    state dimension, used to locate the Xfilter block
#   m:    input dimension, used to locate the Xfilter block
#
# Returns: tuple (t, u), exactly the shape _input_ensemble consumes
#   - t: time points, shared across trajectories (taken from the first trajectory)
#   - u: vector of trajectories, each a vector of the m-dimensional negated filter slices,
#        stored verbatim — no conversion, no reshaping, any m >= 1
#
# Private helper: a plain walk-and-slice. It holds NO guards — the dimension checks (including
# the L1 STRICT full-extended-length rule) live in the exported orchestrator that calls it.
function _adaptive_trajectories(esol, n, m)
    t = esol.u[1].t
    # Outer: one entry per trajectory. Inner: one input value per timepoint of the shared grid.
    u = [[-traj.u[j][2n+1:2n+m] for j in eachindex(t)]
         for traj in esol]
    return (t, u)
end


# compute_control_inputs: build the control-input ensembles that go with an already-computed set
# of state ensembles. Nothing is re-simulated: the baseline input is re-evaluated along the saved
# state trajectories, and the L1 adaptive input is read straight out of the saved L1 extended
# state. Every computed slot is handed back as a toolbox-native EnsembleSolution (via
# _input_ensemble), so EnsembleSummary, the plot recipes, and on/off-grid queries behave on inputs
# exactly as they do on states.
#
# Arguments:
#   solutions:         (positional) NamedTuple with fields nominal_sol, true_sol, L1_sol (each an
#                      EnsembleSolution or nothing) — a live run_simulations result, or one
#                      assembled from load_ensemble results
#   baseline_input:    (positional) the baseline control function, called as baseline_input(t, x, dp)
#   dp:                (positional) dynamics parameters, handed to baseline_input untouched
#   system_dimensions: (positional) a SysDims; n and m are read from it
#   systems:           (kwarg) Vector{Symbol} of state ensembles to draw on; default = all three.
#                      Valid entries: :nominal_sys, :true_sys, :L1_sys. Unlike state_logging, this
#                      is a hard request: every listed system must carry a non-nothing solution.
#   inputs:            (kwarg) Vector{Symbol} of input kinds to compute; default = all three.
#                      Valid entries: :baseline, :adaptive, :total.
#
# Returns a named tuple of five slots, nothing in every slot that was not requested:
#   baseline_nominal — :baseline and :nominal_sys — baseline_input along the nominal ensemble
#   baseline_true    — :baseline and :true_sys    — baseline_input along the true ensemble
#   baseline_L1      — :baseline and :L1_sys      — baseline_input along the L1 ensemble, on
#                                                   x = state[1:n], the X block
#   adaptive_L1      — :adaptive                  — u_a = -Xfilter, read off the L1 ensemble
#   total_L1         — :total                     — the L1 baseline plus the adaptive input, summed
#                                                   elementwise per trajectory per timepoint
#
# :total needs both of its summands, so it computes the L1 baseline and the adaptive input even when
# :baseline / :adaptive were not asked for; whatever gets computed is computed ONCE and shared.
#
# Guards, each a loud error, all of them run to completion before anything is computed:
#   1. selection — unknown or empty `systems` / `inputs`, and any system listed in `systems` whose
#      solution field is nothing (a hard error here, NOT state_logging's silent skip; `systems` is
#      the single selection axis and listing a system asserts it is present)
#   2. L1-only kinds — :adaptive and :total need BOTH :L1_sys ∈ systems AND a non-nothing L1_sol
#   3. dimensions — the state length of every listed ensemble: n for nominal/true, and STRICTLY the
#      full extended 3n+m for L1, since the X block (baseline) and the Xfilter block (adaptive) both
#      have to be addressable — a sliced L1 load is rejected
#
# There is deliberately NO grid check between the two pieces summed into :total: they are read from
# the same L1 ensemble, so they carry the same time grid by construction.
function compute_control_inputs(solutions, baseline_input, dp, system_dimensions;
                                systems::Vector{Symbol}=[:nominal_sys, :true_sys, :L1_sys],
                                inputs::Vector{Symbol}=[:baseline, :adaptive, :total])
    # Guard family 1, selection: validate both kwargs (mirrors state_logging / run_simulations)
    valid_systems = [:nominal_sys, :true_sys, :L1_sys]
    for s in systems
        s ∈ valid_systems || error("Invalid system: $s. Valid options: $valid_systems")
    end
    isempty(systems) && error("Must specify at least one system for the baseline input")

    valid_inputs = [:baseline, :adaptive, :total]
    for k in inputs
        k ∈ valid_inputs || error("Invalid input kind: $k. Valid options: $valid_inputs")
    end
    isempty(inputs) && error("Must specify at least one input kind to compute")

    # Symbol -> (symbol, field name, solution) resolution, done ONCE and reused by the
    # requested-but-nothing check below and by the dimension checks further down
    requested = [s === :nominal_sys ? (s, :nominal_sol, solutions.nominal_sol) :
                 s === :true_sys    ? (s, :true_sol,    solutions.true_sol)    :
                                      (s, :L1_sol,      solutions.L1_sol)
                 for s in systems]

    # Guard family 1, continued: a listed system must carry a solution. Unconditional on `inputs` —
    # listing a system is the request, whether or not a baseline slot ends up being computed for it.
    for (s, field, esol) in requested
        esol !== nothing || error(
            "compute_control_inputs: $s was requested but solutions.$field is nothing. " *
            "Run or load that system first.")
    end

    # Guard family 2: :adaptive and :total are read off the L1 extended state, so each needs BOTH
    # the system selected AND its solution present.
    for k in inputs
        if k === :adaptive || k === :total
            (:L1_sys ∈ systems && solutions.L1_sol !== nothing) || error(
                "compute_control_inputs: input kind $k is L1-only; it needs :L1_sys in `systems` " *
                "and a non-nothing solutions.L1_sol.")
        end
    end

    # Dimensions resolved the way load_ensemble resolves them
    @unpack n, m = system_dimensions

    # Guard family 3, dimensions: state length read the way load_ensemble reads it — the first state
    # vector of the first trajectory. Assumes a non-empty ensemble, as the rebuild path already does.
    for (s, _, esol) in requested
        len = length(esol.u[1].u[1])
        if s === :L1_sys
            # STRICT: the full extended state only. A sliced block cannot serve both the X read
            # (baseline) and the Xfilter read (adaptive).
            len == 3n + m || error(
                "compute_control_inputs: the L1 ensemble has state length $len, not the full extended " *
                "length 3n+m = $(3n+m). Load the full L1 file — load_ensemble(path) with no `component` " *
                "— not a sliced block.")
        else
            len == n || error(
                "compute_control_inputs: the $s ensemble has state length $len, not n = $n. " *
                "Check system_dimensions, or that this ensemble is really that system's state.")
        end
    end

    # ---- validation complete; from here on nothing can fail a selection or dimension check ----

    baseline_nominal = nothing
    baseline_true = nothing
    baseline_L1 = nothing
    adaptive_L1 = nothing
    total_L1 = nothing

    # Baseline along the nominal ensemble: is_L1 = false, the saved state IS x (length n)
    if :baseline ∈ inputs && :nominal_sys ∈ systems
        t, u = _baseline_trajectories(solutions.nominal_sol, baseline_input, dp, n, false)
        baseline_nominal = _input_ensemble(t, u)
    end

    # Baseline along the true ensemble: is_L1 = false as well
    if :baseline ∈ inputs && :true_sys ∈ systems
        t, u = _baseline_trajectories(solutions.true_sol, baseline_input, dp, n, false)
        baseline_true = _input_ensemble(t, u)
    end

    # The two L1 pieces, computed before any of the L1 slots are built so :total can reuse them
    t_baseline_L1 = nothing
    u_baseline_L1 = nothing
    t_adaptive_L1 = nothing
    u_adaptive_L1 = nothing

    # L1 baseline: is_L1 = true, so x is state[1:n], the X block of [X, Xhat, Xfilter, Λhat].
    # Computed for its own slot OR as the first summand of :total.
    if (:baseline ∈ inputs && :L1_sys ∈ systems) || :total ∈ inputs
        t_baseline_L1, u_baseline_L1 =
            _baseline_trajectories(solutions.L1_sol, baseline_input, dp, n, true)
    end

    # Adaptive input, u_a = -Xfilter. Computed for its own slot OR as the second summand of :total.
    if :adaptive ∈ inputs || :total ∈ inputs
        t_adaptive_L1, u_adaptive_L1 = _adaptive_trajectories(solutions.L1_sol, n, m)
    end

    if :baseline ∈ inputs && :L1_sys ∈ systems
        baseline_L1 = _input_ensemble(t_baseline_L1, u_baseline_L1)
    end

    if :adaptive ∈ inputs
        adaptive_L1 = _input_ensemble(t_adaptive_L1, u_adaptive_L1)
    end

    if :total ∈ inputs
        # Elementwise sum, per trajectory per timepoint, of the two pieces already computed above.
        # The baseline values and the adaptive slices add directly; no conversion is applied.
        u_total_L1 = [[u_baseline_L1[i][j] + u_adaptive_L1[i][j]
                       for j in eachindex(u_baseline_L1[i])]
                      for i in eachindex(u_baseline_L1)]
        # Both summands come from the same L1 ensemble, so the grid is shared by construction.
        total_L1 = _input_ensemble(t_baseline_L1, u_total_L1)
    end

    return (
        baseline_nominal = baseline_nominal,
        baseline_true = baseline_true,
        baseline_L1 = baseline_L1,
        adaptive_L1 = adaptive_L1,
        total_L1 = total_L1
    )
end


# control_input_logging: Save control-input ensembles to JLD2 files
# Each saved slot is written with the identical two-key {t, u} schema the state files use.
# Here u holds INPUT trajectories — each trajectory a vector of m-dimensional input values,
# saved verbatim (no slicing, no statistics stored), exactly as the state files hold states.
#
# Arguments:
#   control_inputs: (positional) the five-slot NamedTuple returned by compute_control_inputs, with
#                   fields baseline_nominal, baseline_true, baseline_L1, adaptive_L1, total_L1
#                   (each an EnsembleSolution or nothing)
#   path:           (kwarg) Output directory (created if it doesn't exist)
#
# `path` is the ONLY keyword: there is deliberately no save-time selection kwarg. Every
# non-nothing slot is written, and a nothing slot is silently skipped (state_logging's behavior).
# The selection was already made — and validated loudly — at compute time in
# compute_control_inputs; save time re-decides nothing and holds NO guards of its own.
#
# Field -> file mapping (every slot via _process_solution — inputs carry no extended state, so
# _process_solution_L1 has no role here):
#   baseline_nominal -> inputs_baseline_nominal.jld2
#   baseline_true    -> inputs_baseline_true.jld2
#   baseline_L1      -> inputs_baseline_L1.jld2
#   adaptive_L1      -> inputs_adaptive_L1.jld2
#   total_L1         -> inputs_total_L1.jld2
#
# Returns named tuple of file paths under the SAME five slot names:
# (baseline_nominal=..., baseline_true=..., baseline_L1=..., adaptive_L1=..., total_L1=...),
# with nothing in every slot that was not saved.
function control_input_logging(control_inputs; path::String="sol_logs/")
    mkpath(path)

    baseline_nominal_path = nothing
    baseline_true_path = nothing
    baseline_L1_path = nothing
    adaptive_L1_path = nothing
    total_L1_path = nothing

    # Baseline input along the nominal ensemble
    if control_inputs.baseline_nominal !== nothing
        t, u = _process_solution(control_inputs.baseline_nominal)
        baseline_nominal_path = joinpath(path, "inputs_baseline_nominal.jld2")
        jldsave(baseline_nominal_path; t, u)
    end

    # Baseline input along the true ensemble
    if control_inputs.baseline_true !== nothing
        t, u = _process_solution(control_inputs.baseline_true)
        baseline_true_path = joinpath(path, "inputs_baseline_true.jld2")
        jldsave(baseline_true_path; t, u)
    end

    # Baseline input along the L1 ensemble
    if control_inputs.baseline_L1 !== nothing
        t, u = _process_solution(control_inputs.baseline_L1)
        baseline_L1_path = joinpath(path, "inputs_baseline_L1.jld2")
        jldsave(baseline_L1_path; t, u)
    end

    # L1 adaptive input, u_a = -Xfilter
    if control_inputs.adaptive_L1 !== nothing
        t, u = _process_solution(control_inputs.adaptive_L1)
        adaptive_L1_path = joinpath(path, "inputs_adaptive_L1.jld2")
        jldsave(adaptive_L1_path; t, u)
    end

    # Total L1 input, the L1 baseline plus the adaptive input
    if control_inputs.total_L1 !== nothing
        t, u = _process_solution(control_inputs.total_L1)
        total_L1_path = joinpath(path, "inputs_total_L1.jld2")
        jldsave(total_L1_path; t, u)
    end

    return (
        baseline_nominal = baseline_nominal_path,
        baseline_true = baseline_true_path,
        baseline_L1 = baseline_L1_path,
        adaptive_L1 = adaptive_L1_path,
        total_L1 = total_L1_path
    )
end
