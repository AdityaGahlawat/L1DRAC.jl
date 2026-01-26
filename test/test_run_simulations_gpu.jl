####################################################################################
# GPU Tests for run_simulations
#
# Testing: src/run_simulations.jl
# Setup: Reduced version of examples/ex1/doubleintegrator1D.jl
#
# Test Matrix (GPU - max_GPUs > 0):
# ┌─────┬───────┬──────────┬─────────────────────────────┬─────────────────────────────┐
# │ ID  │ Ntraj │ max_GPUs │ systems                     │ Expected                    │
# ├─────┼───────┼──────────┼─────────────────────────────┼─────────────────────────────┤
# │ T02 │ 1     │ 1        │ [:nominal_sys, :true_sys, :L1_sys]  │ Single traj, 1 GPU          │
# │ T05 │ 10    │ 1        │ [:nominal_sys, :true_sys, :L1_sys]  │ Ensemble, 1 GPU, len=1      │
# │ T06 │ 100   │ 1        │ [:nominal_sys, :true_sys, :L1_sys]  │ Larger ensemble, 1 GPU      │
# │ T07 │ 10    │ 3        │ [:nominal_sys, :true_sys, :L1_sys]  │ Ensemble, 3 GPUs, len=3     │
# │ T08 │ 100   │ 3        │ [:nominal_sys, :true_sys, :L1_sys]  │ Larger ensemble, 3 GPUs     │
# ├─────┼───────┼──────────┼─────────────────────────────┼─────────────────────────────┤
# │ X02 │ 10    │ 10       │ [:nominal_sys, :true_sys, :L1_sys]  │ Caps to available GPUs      │
# └─────┴───────┴──────────┴─────────────────────────────┴─────────────────────────────┘
#
# Note: These tests only run if CUDA.functional() returns true
####################################################################################

# Helper: Create minimal test setup (same as CPU tests)
function create_test_setup_gpu(; Ntraj=10, include_L1params=true)
    # Simulation Parameters
    tspan = (0.0, 0.1)  # Short for fast tests
    Δₜ = 1e-3
    Δ_saveat = 1e-2
    simulation_parameters = sim_params(tspan, Δₜ, Ntraj, Δ_saveat)

    # System Dimensions
    n, m, d = 2, 1, 2
    system_dimensions = sys_dims(n, m, d)

    # Simple dynamics
    K = @SMatrix [-10.0 -10.0]
    dp = (; K=K)

    f(t, x, dp) = @SVector [-dp.K[1]*x[1] - dp.K[2]*x[2], x[1]]
    g(t, x, dp) = @SVector [0.0, 1.0]
    g_perp(t, x, dp) = @SVector [1.0, 0.0]
    p(t, x, dp) = @SMatrix [0.01 0.1; 0.0 0.8]

    nominal_components = nominal_vector_fields(f, g, g_perp, p, dp)

    # Uncertain Vector Fields
    Λμ(t, x, dp) = @SVector [0.01, 1.0]
    Λσ(t, x, dp) = @SMatrix [0.0 0.1; 0.0 0.5]
    uncertain_components = uncertain_vector_fields(Λμ, Λσ)

    # Initial Distributions
    nominal_ξ₀ = MvNormal(zeros(2), 0.1 * I(2))
    true_ξ₀ = MvNormal(zeros(2), 0.1 * I(2))
    initial_distributions = init_dist(nominal_ξ₀, true_ξ₀)

    # Define Systems
    nominal_system = nom_sys(system_dimensions, nominal_components, initial_distributions)
    true_system = true_sys(system_dimensions, nominal_components, uncertain_components, initial_distributions)

    if include_L1params
        L1params = drac_params(50.0, 10 * Δₜ, 100.0)
        return (
            simulation_parameters = simulation_parameters,
            nominal_system = nominal_system,
            true_system = true_system,
            L1params = L1params
        )
    else
        return (
            simulation_parameters = simulation_parameters,
            nominal_system = nominal_system,
            true_system = true_system
        )
    end
end

# Get number of available GPUs
num_gpus_available = length(CUDA.devices())

@testset "run_simulations - GPU" begin

    @testset "T02: Single trajectory, 1 GPU" begin
        setup = create_test_setup_gpu(; Ntraj=1)
        solutions = run_simulations(setup; max_GPUs=1)

        @test solutions isa NamedTuple
        @test haskey(solutions, :nominal_sol)
        @test haskey(solutions, :true_sol)
        @test haskey(solutions, :L1_sol)
        @test length(solutions.nominal_sol) == 1
    end

    @testset "T05: Ensemble Ntraj=10, 1 GPU" begin
        setup = create_test_setup_gpu(; Ntraj=10)
        solutions = run_simulations(setup; max_GPUs=1)

        @test solutions isa NamedTuple
        @test length(solutions.nominal_sol) == 1
        @test length(solutions.nominal_sol[1]) == 10
    end

    @testset "T06: Ensemble Ntraj=100, 1 GPU" begin
        setup = create_test_setup_gpu(; Ntraj=100)
        solutions = run_simulations(setup; max_GPUs=1)

        @test length(solutions.nominal_sol[1]) == 100
        @test length(solutions.true_sol[1]) == 100
        @test length(solutions.L1_sol[1]) == 100
    end

    if num_gpus_available >= 3
        @testset "T07: Ensemble Ntraj=10, 3 GPUs" begin
            setup = create_test_setup_gpu(; Ntraj=10)
            solutions = run_simulations(setup; max_GPUs=3)

            @test length(solutions.nominal_sol) == 3
            # Total trajectories across all GPUs
            total_nominal = sum(length(s) for s in solutions.nominal_sol)
            @test total_nominal == 10
        end

        @testset "T08: Ensemble Ntraj=100, 3 GPUs" begin
            setup = create_test_setup_gpu(; Ntraj=100)
            solutions = run_simulations(setup; max_GPUs=3)

            @test length(solutions.nominal_sol) == 3
            total_nominal = sum(length(s) for s in solutions.nominal_sol)
            @test total_nominal == 100
        end
    else
        @info "Skipping multi-GPU tests (T07, T08) - only $num_gpus_available GPU(s) available"
    end

    @testset "X02: max_GPUs > available caps correctly" begin
        setup = create_test_setup_gpu(; Ntraj=10)
        # Request more GPUs than available
        solutions = run_simulations(setup; max_GPUs=10)

        # Should cap to available
        @test length(solutions.nominal_sol) == num_gpus_available
    end

end
