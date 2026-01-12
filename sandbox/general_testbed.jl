# General Testbed for L1DRAC simulations
  include("Setup_DoubleIntegrator1D.jl")
  setup = setup_double_integrator()
  test_params = sim_params(setup.tspan, setup.Δₜ, 10, setup.Δ_saveat)

function run_tests(type; kwargs...)
    backend = get(kwargs, :backend, :cpu)
    if backend == :cpu
        if type == 1 
            system_simulation(test_params, setup.nominal_system, CPU())
        elseif type == 2 
            system_simulation(test_params, setup.nominal_system, CPU(); simtype=:ensemble)
        elseif type == 3 
            system_simulation(test_params, setup.true_system, CPU())
        elseif type == 4 
            system_simulation(test_params, setup.true_system, CPU(); simtype=:ensemble)
        else 
            @error "Unknown test type: $type"
        end
    elseif backend == :gpu
        if type == 1
            system_simulation(test_params, setup.nominal_system, GPU())
        # elseif type == 3 
        #     system_simulation(test_params, setup.true_system, GPU())
        # elseif type == 4 
        #     system_simulation(test_params, setup.true_system, GPU(); simtype=:ensemble)
        else 
            @error "Unknown test type: $type"
        end
    else
        @error "Unknown backend: $backend"
    end

end
  
