####################################################################################
# High-Level Simulation Runner
#
# Usage:
#   setup = setup_system(; Ntraj=10)
#   solutions = run_simulations(setup; max_GPUs=1)
#   solutions = run_simulations(setup; systems=[:nominal_sys, :true_sys, :L1_sys])
####################################################################################

function run_simulations(setup; max_GPUs::Int=1, systems::Vector{Symbol}=[:nominal_sys, :true_sys, :L1_sys])
    # Validate systems kwarg
    valid_systems = [:nominal_sys, :true_sys, :L1_sys]
    for s in systems
        s ∈ valid_systems || error("Invalid system: $s. Valid options: $valid_systems")
    end
    isempty(systems) && error("Must specify at least one system to simulate")

    # Check L1DRAC requirements
    if :L1_sys ∈ systems && !hasproperty(setup, :L1params)
        error("L1DRAC system requested but setup.L1params not provided")
    end

    # Setup backend
    numGPUs = get_numGPUs(max_GPUs)
    backend = get_backend(numGPUs)

    # Auto-detect ensemble vs single trajectory
    Ntraj = setup.simulation_parameters.Ntraj
    sim_kwargs = Ntraj > 1 ? (simtype=:ensemble,) : (;)

    # Run requested simulations
    results = Dict{Symbol, Any}()

    if :nominal_sys ∈ systems
        results[:nominal_sol] = system_simulation(
            setup.simulation_parameters,
            setup.nominal_system,
            backend;
            sim_kwargs...
        )
    end

    if :true_sys ∈ systems
        results[:true_sol] = system_simulation(
            setup.simulation_parameters,
            setup.true_system,
            backend;
            sim_kwargs...
        )
    end

    if :L1_sys ∈ systems
        results[:L1_sol] = system_simulation(
            setup.simulation_parameters,
            setup.true_system,
            setup.L1params,
            backend;
            sim_kwargs...
        )
    end

    # Cleanup
    cleanup_environment(backend)

    return NamedTuple(results)
end
