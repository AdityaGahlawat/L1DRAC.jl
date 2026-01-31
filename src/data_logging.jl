"""
    state_logging(sol_nominal, sol_true, sol_L1; n::Int, path::String="sol_logs/")

Save ensemble simulation results to JLD2 files for remote-compute to local-plot workflow.

Extracts time and state data from `EnsembleSolution` objects (or `Vector{EnsembleSolution}`
from multi-GPU runs) and saves them as portable JLD2 files that can be loaded on any machine
without needing the full DifferentialEquations environment.

# Arguments
- `sol_nominal`: Nominal system solution (EnsembleSolution, Vector{EnsembleSolution}, or nothing)
- `sol_true`: True system solution (EnsembleSolution, Vector{EnsembleSolution}, or nothing)
- `sol_L1`: L1 adaptive control solution (EnsembleSolution, Vector{EnsembleSolution}, or nothing)
- `n::Int`: State dimension - for L1 solution, extracts only first n components (removes filter state)
- `path::String="sol_logs/"`: Output directory path (created if doesn't exist)

# Returns
Named tuple of saved file paths (or nothing if solution was nothing):
```julia
(nominal = "sol_logs/states_nominal.jld2",
 true_sys = "sol_logs/states_true.jld2",
 L1 = "sol_logs/states_L1.jld2")
```

# File Format
Each saved file contains:
- `"t"`: Time vector (shared across all trajectories when using saveat)
- `"u"`: Vector of trajectory state arrays, where `u[i]` is the i-th trajectory

# Example
```julia
# Run simulations
setup = setup_system(Ntraj=1000)
solutions = run_simulations(setup; max_GPUs=1)

# Save data
paths = state_logging(solutions.nominal_sol, solutions.true_sol, solutions.L1_sol; n=2)

# Load on any machine
using JLD2
data = load(paths.nominal)
t = data["t"]      # Time vector
u = data["u"]      # Vector of trajectories
```

See also: [`run_simulations`](@ref)
"""
function state_logging(sol_nominal, sol_true, sol_L1; n::Int, path::String="sol_logs/")
    # Create output directory if it doesn't exist
    mkpath(path)

    # Initialize return paths
    nominal_path = nothing
    true_path = nothing
    L1_path = nothing

    # Save nominal solution
    if sol_nominal !== nothing
        t, u = _extract_solution_data(sol_nominal)
        nominal_path = joinpath(path, "states_nominal.jld2")
        jldsave(nominal_path; t=t, u=u)
    end

    # Save true system solution
    if sol_true !== nothing
        t, u = _extract_solution_data(sol_true)
        true_path = joinpath(path, "states_true.jld2")
        jldsave(true_path; t=t, u=u)
    end

    # Save L1 solution (extract only first n state components)
    if sol_L1 !== nothing
        t, u_full = _extract_solution_data(sol_L1)
        # Extract only first n components (remove filter_state and other L1 extended state)
        u = [[state[1:n] for state in traj] for traj in u_full]
        L1_path = joinpath(path, "states_L1.jld2")
        jldsave(L1_path; t=t, u=u)
    end

    return (
        nominal = nominal_path,
        true_sys = true_path,
        L1 = L1_path
    )
end


"""
    _extract_solution_data(sol)

Extract time and state data from solution, handling both single-GPU (EnsembleSolution)
and multi-GPU (Vector{EnsembleSolution}) formats.

# Returns
Tuple `(t, u)` where:
- `t`: Time vector (from first trajectory)
- `u`: Vector of trajectory state arrays
"""
function _extract_solution_data(sol)
    if sol isa Vector
        # Multi-GPU: Vector{EnsembleSolution} - concatenate trajectories across GPUs
        return _extract_multi_gpu_data(sol)
    else
        # Single GPU: EnsembleSolution
        return _extract_ensemble_data(sol)
    end
end


"""
    _extract_ensemble_data(sol::EnsembleSolution)

Extract time and state data from a single EnsembleSolution.

# Returns
Tuple `(t, u)` where:
- `t`: Time vector (shared across trajectories when using saveat)
- `u`: Vector of trajectory state arrays
"""
function _extract_ensemble_data(sol)
    # Time grid is shared across all trajectories when using saveat
    t = sol[1].t
    # Extract state arrays for each trajectory
    u = [traj.u for traj in sol]
    return (t, u)
end


"""
    _extract_multi_gpu_data(sols::Vector{EnsembleSolution})

Extract and concatenate time and state data from multi-GPU solutions.

# Returns
Tuple `(t, u)` where:
- `t`: Time vector (same for all GPUs, uses first)
- `u`: Concatenated vector of trajectory state arrays from all GPUs
"""
function _extract_multi_gpu_data(sols)
    # Time is same for all GPUs (use first)
    t = sols[1][1].t
    # Concatenate trajectories from all GPUs
    u = vcat([_extract_ensemble_data(sol)[2] for sol in sols]...)
    return (t, u)
end
