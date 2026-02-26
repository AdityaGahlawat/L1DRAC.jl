# plotting_utils.jl: State trajectory plotting for examples/ex1 (double integrator, 2 states)
# Extracts each state component (X1, X2) into flat T x Ntraj matrices for plotting.

using Plots
using LaTeXStrings
using StaticArrays

# Trajectory data arrives as named tuples with .u = Vector{Vector{SVector{2,Float64}}}
# Each state component is extracted separately into a flat Float64 matrix.

# extract_state_matrix: Extract the chosen state component into a T x n_show matrix
# n_show = min(Ntraj, max_traj) — user-chosen cap on trajectories to display
# columns = trajectories, rows = timepoints
function extract_state_matrix(data; state_component::Int, max_traj=500)
    u = data["u"]
    n_show = min(length(u), max_traj)
    return reduce(hcat, [getindex.(u[i], state_component) for i in 1:n_show])
end

# build_trajectory_panel: Plot individual trajectories for one state component
# Overlays all three systems (nom, tru, L1) on a single panel
# Colors: nominal=13, true=7, L1=25 (palette indices), low alpha for cloud effect
function build_trajectory_panel(nom, tru, L1; state_component::Int, max_traj=500,
                                 lw=1, lalpha=0.1, title_str="")
    t_nom = nom["t"]
    t_tru = tru["t"]
    t_L1  = L1["t"]

    nom_flat = extract_state_matrix(nom; state_component=state_component, max_traj=max_traj)
    tru_flat = extract_state_matrix(tru; state_component=state_component, max_traj=max_traj)
    L1_flat  = extract_state_matrix(L1;  state_component=state_component, max_traj=max_traj)

    p = plot(t_nom, nom_flat, color=13, lw=lw, linealpha=lalpha, label=false, legend=false)
    plot!(p, t_tru, tru_flat, color=7,  lw=lw, linealpha=lalpha, label=false)
    plot!(p, t_L1,  L1_flat,  color=25, lw=lw, linealpha=lalpha, label=false)

    plot!(p, title=title_str)
    return p
end
