####################################################################################
# CPU Tests for run_simulations
#
# Testing: src/run_simulations.jl
# Setup: Reduced version of examples/ex1/doubleintegrator1D.jl
#
# Test Matrix (CPU - max_GPUs=0):
# ┌─────┬───────┬──────────┬─────────────────────────────┬─────────────────────────────┐
# │ ID  │ Ntraj │ max_GPUs │ systems                     │ Expected                    │
# ├─────┼───────┼──────────┼─────────────────────────────┼─────────────────────────────┤
# │ T01 │ 1     │ 0        │ [:nominal_sys, :true_sys, :L1_sys]  │ Single traj, 3 solutions    │
# │ T03 │ 10    │ 0        │ [:nominal_sys, :true_sys, :L1_sys]  │ Ensemble, length(sol) = 1   │
# │ T04 │ 100   │ 0        │ [:nominal_sys, :true_sys, :L1_sys]  │ Larger ensemble             │
# ├─────┼───────┼──────────┼─────────────────────────────┼─────────────────────────────┤
# │ T09 │ 10    │ 0        │ [:nominal_sys]                  │ Only :nominal_sol key       │
# │ T10 │ 10    │ 0        │ [:true_sys]                     │ Only :true_sol key          │
# │ T11 │ 10    │ 0        │ [:L1_sys]                   │ Only :L1_sol key            │
# │ T12 │ 10    │ 0        │ [:nominal_sys, :true_sys]           │ Two keys                    │
# │ T13 │ 10    │ 0        │ [:nominal_sys, :L1_sys]         │ Two keys                    │
# │ T14 │ 10    │ 0        │ [:true_sys, :L1_sys]            │ Two keys                    │
# ├─────┼───────┼──────────┼─────────────────────────────┼─────────────────────────────┤
# │ E01 │ 10    │ 0        │ []                          │ Error: empty systems        │
# │ E02 │ 10    │ 0        │ [:invalid_sys]                  │ Error: invalid system       │
# │ E03 │ 10    │ 0        │ [:L1_sys]                   │ Error: no L1params          │
# └─────┴───────┴──────────┴─────────────────────────────┴─────────────────────────────┘
####################################################################################

# Helper: Create minimal test setup
function create_test_setup(; Ntraj=10, include_L1params=true)
    # Simulation Parameters
    tspan = (0.0, 0.1)  # Short for fast tests
    Δₜ = 1e-3
    Δ_saveat = 1e-2
    simulation_parameters = sim_params(tspan, Δₜ, Ntraj, Δ_saveat)

    # System Dimensions
    n, m, d = 2, 1, 2
    system_dimensions = sys_dims(n, m, d)

    # Simple dynamics (no ControlSystemsBase dependency)
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

@testset "run_simulations - CPU" begin

    @testset "T01: Single trajectory, all systems" begin
        setup = create_test_setup(; Ntraj=1)
        solutions = run_simulations(setup; max_GPUs=0)

        @test solutions isa NamedTuple
        @test haskey(solutions, :nominal_sol)
        @test haskey(solutions, :true_sol)
        @test haskey(solutions, :L1_sol)
        @test length(solutions.nominal_sol) == 1  # CPU returns Vector of length 1
    end

    @testset "T03: Ensemble Ntraj=10, all systems" begin
        setup = create_test_setup(; Ntraj=10)
        solutions = run_simulations(setup; max_GPUs=0)

        @test solutions isa NamedTuple
        @test haskey(solutions, :nominal_sol)
        @test haskey(solutions, :true_sol)
        @test haskey(solutions, :L1_sol)
        @test length(solutions.nominal_sol) == 1
        # Check trajectory count
        @test length(solutions.nominal_sol[1]) == 10
    end

    @testset "T04: Ensemble Ntraj=100, all systems" begin
        setup = create_test_setup(; Ntraj=100)
        solutions = run_simulations(setup; max_GPUs=0)

        @test solutions isa NamedTuple
        @test length(solutions.nominal_sol[1]) == 100
        @test length(solutions.true_sol[1]) == 100
        @test length(solutions.L1_sol[1]) == 100
    end

    @testset "T09: Only nominal system" begin
        setup = create_test_setup(; Ntraj=10)
        solutions = run_simulations(setup; max_GPUs=0, systems=[:nominal_sys])

        @test !isnothing(solutions.nominal_sol)
        @test isnothing(solutions.true_sol)
        @test isnothing(solutions.L1_sol)
    end

    @testset "T10: Only true system" begin
        setup = create_test_setup(; Ntraj=10)
        solutions = run_simulations(setup; max_GPUs=0, systems=[:true_sys])

        @test isnothing(solutions.nominal_sol)
        @test !isnothing(solutions.true_sol)
        @test isnothing(solutions.L1_sol)
    end

    @testset "T11: Only L1DRAC system" begin
        setup = create_test_setup(; Ntraj=10)
        solutions = run_simulations(setup; max_GPUs=0, systems=[:L1_sys])

        @test isnothing(solutions.nominal_sol)
        @test isnothing(solutions.true_sol)
        @test !isnothing(solutions.L1_sol)
    end

    @testset "T12: nominal + true" begin
        setup = create_test_setup(; Ntraj=10)
        solutions = run_simulations(setup; max_GPUs=0, systems=[:nominal_sys, :true_sys])

        @test !isnothing(solutions.nominal_sol)
        @test !isnothing(solutions.true_sol)
        @test isnothing(solutions.L1_sol)
    end

    @testset "T13: nominal + L1DRAC" begin
        setup = create_test_setup(; Ntraj=10)
        solutions = run_simulations(setup; max_GPUs=0, systems=[:nominal_sys, :L1_sys])

        @test !isnothing(solutions.nominal_sol)
        @test isnothing(solutions.true_sol)
        @test !isnothing(solutions.L1_sol)
    end

    @testset "T14: true + L1DRAC" begin
        setup = create_test_setup(; Ntraj=10)
        solutions = run_simulations(setup; max_GPUs=0, systems=[:true_sys, :L1_sys])

        @test isnothing(solutions.nominal_sol)
        @test !isnothing(solutions.true_sol)
        @test !isnothing(solutions.L1_sol)
    end

    @testset "E01: Empty systems list" begin
        setup = create_test_setup(; Ntraj=10)
        @test_throws ErrorException run_simulations(setup; max_GPUs=0, systems=Symbol[])
    end

    @testset "E02: Invalid system name" begin
        setup = create_test_setup(; Ntraj=10)
        @test_throws ErrorException run_simulations(setup; max_GPUs=0, systems=[:invalid_sys])
    end

    @testset "E03: L1DRAC without L1params" begin
        setup = create_test_setup(; Ntraj=10, include_L1params=false)
        @test_throws ErrorException run_simulations(setup; max_GPUs=0, systems=[:L1_sys])
    end

end
