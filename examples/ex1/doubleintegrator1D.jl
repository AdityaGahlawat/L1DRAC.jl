## L1DRAC for a 1D Double Integrator

using L1DRAC
using CUDA
using LinearAlgebra
using Distributions
using ControlSystemsBase
using StaticArrays
using Plots

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
    Λμ_m(t, x, dp) = 1.0 * (5 + 10*cos(x[2]) + 5*norm(x))
    Λμ(t, x, dp) = @SVector [Λμ_um(t, x, dp), Λμ_m(t, x, dp)]

    Λσ_um(t, x, dp) = 0.0 * @SMatrix [0.1+cos(x[2]) 2.0]
    Λσ_m(t, x, dp) = @SMatrix [0.0 5+sin(x[2])+5.0*(norm(x) < 1 ? norm(x) : sqrt(norm(x)))]
    Λσ(t, x, dp) = vcat(Λσ_um(t, x, dp), Λσ_m(t, x, dp))

    uncertain_components = uncertain_vector_fields(Λμ, Λσ)

    # Initial Distributions
    nominal_ξ₀ = MvNormal(1e-2 * ones(2), 1.0 * I(2))
    true_ξ₀ = MvNormal(-1.0 * ones(2), 1e-1 * I(2))
    initial_distributions = init_dist(nominal_ξ₀, true_ξ₀)

    # Define Systems
    nominal_system = nom_sys(system_dimensions, nominal_components, initial_distributions)
    true_system = true_sys(system_dimensions, nominal_components, uncertain_components, initial_distributions)

    # L1-DRAC Parameters (PLACEHOLDER values)
    ω = 50.0 # Filter bandwidth
    Tₛ = 10 * Δₜ # Sample time (integer multiple of Δₜ)
    λₛ = 100.0 # Predictor stability
    L1params = drac_params(ω, Tₛ, λₛ)

    return (
        simulation_parameters = simulation_parameters,
        nominal_system = nominal_system,
        true_system = true_system,
        L1params = L1params
    )
end

###################################################################
## MAIN
###################################################################
function main(; Ntraj = Int(1e1), max_GPUs=10, systems=[:nominal_sys, :true_sys, :L1_sys]) 

    @info "Warmup run for JIT compilation"
    println("=====================================") 
    warmup_setup = setup_system(; Ntraj = 10)
    run_simulations(warmup_setup; max_GPUs=max_GPUs, systems=systems)

    println("=====================================")
    @info "Complete run for Ntraj=$Ntraj" 
    println("=====================================")
    setup = setup_system(; Ntraj = Ntraj)
    solutions = run_simulations(setup; max_GPUs=max_GPUs, systems=systems)
    return solutions, systems
end

main();

###################################################################
## PLOTS
###################################################################
# function plot_results(solutions)
#     # TODO
# end