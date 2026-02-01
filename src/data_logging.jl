# data_logging.jl: Save ensemble simulation solutions with uniform structure
#
# All systems (nominal, true, L1) save files with identical structure:
#   (t, u, mean, var)
#   - t:    Vector of time points
#   - u:    Vector of trajectories (each trajectory is Vector of state vectors)
#   - mean: Vector of mean state at each time point (from EnsembleSummary)
#   - var:  Vector of variance at each time point (from EnsembleSummary)
#
# This uniform structure enables consistent plotting API across all systems.
#
# EXAMPLE with n=2 state dimension, m=1 control dimension:
#
#   Nominal/True system state at time t:
#     [X_1, X_2]                                        (length 2)
#
#   L1 extended state at time t:
#     [X_1, X_2, Xhat_1, Xhat_2, Xf_1, Xf_2, Λhat_1]   (length 3n+m = 7)
#     |-------|  |-----------|  |--------|  |-----|
#       X (n)     Xhat (n)      Xfilter(n)  Λhat(m)
#
#   For L1, we only want X (first n=2 components) in saved output.
#
# L1 EXTRACTION ORDER (CRITICAL):
#   1. Compute EnsembleSummary on FULL extended state (length 3n+m)
#      - This gives mean/var for all 7 components
#   2. Extract first n components from mean/var arrays
#      - mean = [m[1:2] for m in mean_full]
#   3. Extract first n components from each trajectory
#      - u = [[state[1:2] for state in traj] for traj in u_full]
#
#   WHY THIS ORDER: EnsembleSummary requires EnsembleSolution object.
#   Once we extract to arrays, we lose the ability to call EnsembleSummary.
#
# MULTI-GPU HANDLING:
#   Solutions arrive as Vector{EnsembleSolution}, one element per GPU.
#   - Concatenate trajectories across all GPUs into single list
#   - Combine statistics using pooled variance formula
#
# HELPER FUNCTIONS:
#   _concatenate_trajectories(sol_vec) - Flatten trajectories from all GPUs
#   _compute_statistics(sol_vec)       - Compute mean/var with pooled variance formula
#   _process_solution(sol_vec)         - Process nominal/true into (t, u, mean, var)
#   _process_solution_L1(sol_vec, n)   - Process L1 with extraction after stats
#
# MAIN FUNCTION:
#   state_logging(sol_nominal, sol_true, sol_L1, system_dimensions; path)
#     - Saves each non-nothing solution to JLD2 file
#     - Returns named tuple of file paths


# _concatenate_trajectories: Combine trajectories from multiple EnsembleSolutions
# Input: Vector{EnsembleSolution} (from multi-GPU or single-GPU)
# Returns: Vector of trajectories (each trajectory is the .u field from a single solve)
function _concatenate_trajectories(sol_vec)
    u = []
    for ensemble_sol in sol_vec
        for traj in ensemble_sol
            push!(u, traj.u)
        end
    end
    return u
end


# _compute_statistics: Compute mean/variance across all trajectories using EnsembleSummary
# Input: Vector{EnsembleSolution}
# Returns: (mean, var) tuple where each is Vector over timepoints
# Uses pooled variance formula to combine statistics from multiple GPUs
function _compute_statistics(sol_vec)
    # Compute EnsembleSummary for each GPU's solution
    summaries = [EnsembleSummary(ensemble_sol) for ensemble_sol in sol_vec]

    # Weights = number of trajectories per GPU
    weights = [length(ensemble_sol) for ensemble_sol in sol_vec]
    total_traj = sum(weights)

    T = length(summaries[1].u)
    mean = similar(summaries[1].u)
    var = similar(summaries[1].v)

    for t_idx in 1:T
        # Weighted mean across GPUs
        weighted_sum = sum(w * s.u[t_idx] for (w, s) in zip(weights, summaries))
        mean[t_idx] = weighted_sum / total_traj

        # Combined variance using pooled variance formula:
        # Var(combined) = weighted_avg(Var_i + (mean_i - mean_combined)^2)
        var[t_idx] = sum(w * (s.v[t_idx] .+ (s.u[t_idx] .- mean[t_idx]).^2)
                        for (w, s) in zip(weights, summaries)) / total_traj
    end

    return (mean, var)
end


# _process_solution: Extract time, trajectories, and statistics from solution
# Input: Vector{EnsembleSolution} (length 1 for single GPU/CPU, >1 for multi-GPU)
# Returns: Named tuple (t, u, mean, var) for JLD2 saving
function _process_solution(sol_vec)
    t = sol_vec[1][1].t
    u = _concatenate_trajectories(sol_vec)
    mean, var = _compute_statistics(sol_vec)
    return (t=t, u=u, mean=mean, var=var)
end


# state_logging: Save ensemble simulation solutions to JLD2 files
# Saves EnsembleSolution objects directly for nominal and true systems.
# For L1 solution, extracts only the first n state components (removes filter state).
#
# Arguments:
#   sol_nominal, sol_true, sol_L1: Solutions (Vector{EnsembleSolution} or nothing)
#   system_dimensions: SysDims struct containing n, m, d
#   path: Output directory (created if doesn't exist)
#
# Returns named tuple of file paths: (nominal=..., true_sys=..., L1=...)
function state_logging(sol_nominal, sol_true, sol_L1, system_dimensions; path::String="sol_logs/")
    @unpack n = system_dimensions
    mkpath(path)

    nominal_path = nothing
    true_path = nothing
    L1_path = nothing

    # Save nominal solution directly
    if sol_nominal !== nothing
        nominal_path = joinpath(path, "states_nominal.jld2")
        jldsave(nominal_path; sol=sol_nominal)
    end

    # Save true system solution directly
    if sol_true !== nothing
        true_path = joinpath(path, "states_true.jld2")
        jldsave(true_path; sol=sol_true)
    end

    # Save L1 solution - extract only first n state components
    if sol_L1 !== nothing
        L1_extracted = _extract_L1_state(sol_L1, n)
        L1_path = joinpath(path, "states_L1.jld2")
        jldsave(L1_path; sol=L1_extracted)
    end

    return (
        nominal = nominal_path,
        true_sys = true_path,
        L1 = L1_path
    )
end


# _extract_L1_state: Extract first n components from L1 solution
# L1 has extended state [X, Xhat, Xfilter, Λhat], we only want X (first n)
# sol_L1 is always Vector{EnsembleSolution} (even single GPU returns [sol])
function _extract_L1_state(sol_L1, n)
    return [_extract_L1_ensemble(ensemble_sol, n) for ensemble_sol in sol_L1]
end


# _extract_L1_ensemble: Extract first n components from each trajectory in an EnsembleSolution
# Returns (t, u) tuple where u has truncated states
function _extract_L1_ensemble(sol, n)
    t = sol[1].t
    u = [[state[1:n] for state in traj.u] for traj in sol]
    return (t=t, u=u)
end
