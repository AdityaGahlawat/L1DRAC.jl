# Setup for Double Integrator 1D benchmark

using L1DRAC
using LinearAlgebra
using Distributions
using ControlSystemsBase
using Dates
using DataFrames
using CSV
using CUDA
using Distributed
using StaticArrays


function setup_double_integrator(; Ntraj = 10)
    # Simulation Parameters
    tspan = (0.0, 5.0)
    Δₜ = 1e-4
    Δ_saveat = 1e2 * Δₜ
    simulation_parameters = sim_params(tspan, Δₜ, Ntraj, Δ_saveat)

    # System Dimensions
    n, m, d = 2, 1, 2
    system_dimensions = sys_dims(n, m, d)

    # System constants
    λ = 10.0
    σ_scale = 1.0

    # System matrices
    A = @SMatrix [0.0 1.0; 0.0 0.0]
    B = @SMatrix [0.0; 1.0]
    C = SMatrix{2,2}(1.0I)
    D = 0.0

    sys = ss(A, B, C, D)
    DesiredPoles = -λ * ones(2)
    K = SMatrix{1,2}(place(sys, DesiredPoles))
    A_cl = A - B * K

    # All qunatities whose size cannot be determined at compile-time need to go inside the dynamics_params tuple for GPU compatibility
    # E.g., instead of declaring global consts, we wish to keep A_cl, B, and K as variables, hence will be collected into dynamics_params
    # Build params tuple for GPU
    

    # Dynamics functions

    trck_traj(t) = @SVector [5*sin(t) + 3*cos(2*t), 0.0]

    dynamics_params = (; A_cl, B, K) # Collects items for dynamics whose size cannot be determined at compile-time (for GPU)
    f(t, x, dynamics_params) = dynamics_params.A_cl * x + dynamics_params.B * dynamics_params.K * trck_traj(t)

    g(t, x, dynamics_params) = @SVector [0.0, 1.0]
    g_perp(t, x, dynamics_params) = @SVector [1.0, 0.0]

    p_um(t, x, dynamics_params) = 2.0 * @SMatrix [0.01 0.1]
    p_m(t, x, dynamics_params) = 1.0 * @SMatrix [0.0 0.8]
    p(t, x, dynamics_params) = vcat(p_um(t, x, dynamics_params), p_m(t, x, dynamics_params))

    Λμ_um(t, x, dynamics_params) = 1e-2 * (1 + sin(x[1]))
    Λμ_m(t, x, dynamics_params) = 1.0 * (5 + 10*cos(x[2]) + 5*norm(x))
    Λμ(t, x, dynamics_params) = @SVector [Λμ_um(t, x, dynamics_params), Λμ_m(t, x, dynamics_params)]

    Λσ_um(t, x, dynamics_params) = 0.0 * @SMatrix [0.1+cos(x[2]) 2.0]
    Λσ_m(t, x, dynamics_params) = σ_scale * @SMatrix [0.0 5+sin(x[2])+5.0*(norm(x) < 1 ? norm(x) : sqrt(norm(x)))]
    Λσ(t, x, dynamics_params) = vcat(Λσ_um(t, x, dynamics_params), Λσ_m(t, x, dynamics_params))

    # Nominal Vector Fields
    nominal_components = nominal_vector_fields(f, g, g_perp, p, dynamics_params)

    # Uncertain Vector Fields
    uncertain_components = uncertain_vector_fields(Λμ, Λσ)

    # Initial distributions
    nominal_ξ₀ = MvNormal(1e-2 * ones(2), 1.0 * I(2))
    true_ξ₀ = MvNormal(-1.0 * ones(2), 1e-1 * I(2))
    initial_distributions = init_dist(nominal_ξ₀, true_ξ₀)

    # Define the systems
    nominal_system = nom_sys(system_dimensions, nominal_components, initial_distributions)
    true_system = true_sys(system_dimensions, nominal_components, uncertain_components, initial_distributions)

    # L1 DRAC Parameters
    ω = 50.0
    Tₛ = 10 * Δₜ
    λₛ = 100.0
    L1params = drac_params(ω, Tₛ, λₛ)

    return (
        simulation_parameters = simulation_parameters,
        system_dimensions = system_dimensions,
        dynamics_params = dynamics_params,
        nominal_system = nominal_system,
        true_system = true_system,
        L1params = L1params
    )
end
