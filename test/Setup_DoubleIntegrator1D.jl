# Setup for Double Integrator 1D benchmark


# Benchmark constants (fixed for this system)
const BENCHMARK_n = 2
const BENCHMARK_m = 1
const BENCHMARK_d = 2
const BENCHMARK_λ = 10.0
const BENCHMARK_σ_scale = 1.0

# Config struct for variable parameters
struct DoubleIntegratorConfig
    Δₜ::Float64
    tspan::Tuple{Float64, Float64}
end

# Module scope functions - required for GPU kernel compilation

function trck_traj_benchmark(t)
    return [5*sin(t) + 3*cos(2*t); 0.]
end

function stbl_cntrl_benchmark()
    A = [0 1.0; 0 0]
    B = [0; 1.0]
    C = I(2)
    D = 0.0
    sys = ss(A, B, C, D)
    DesiredPoles = -BENCHMARK_λ * ones(2)
    K = place(sys, DesiredPoles)
    return K, norm(A - B*K, 2), A - B*K
end

function f_benchmark(t, x)
    A = [0 1.0; 0 0]
    B = [0; 1.0]
    K = stbl_cntrl_benchmark()[1]
    return (A - B*K) * x + B * K * trck_traj_benchmark(t)
end

g_benchmark(t) = [0; 1]
g_perp_benchmark(t) = [1; 0]
g_bar_benchmark(t) = [g_benchmark(t) g_perp_benchmark(t)]
Θ_ad_benchmark(t) = [I(BENCHMARK_m) zeros(BENCHMARK_m, BENCHMARK_n - BENCHMARK_m)] * inv(g_bar_benchmark(t))

p_um_benchmark(t, x) = 2.0 * [0.01 0.1]
p_m_benchmark(t, x) = 1.0 * [0.0 0.8]
p_benchmark(t, x) = vcat(p_um_benchmark(t, x), p_m_benchmark(t, x))

Λμ_um_benchmark(t, x) = 1e-2 * (1 + sin(x[1]))
Λμ_m_benchmark(t, x) = 1.0 * (5 + 10*cos(x[2]) + 5*norm(x))
Λμ_benchmark(t, x) = vcat(Λμ_um_benchmark(t, x), Λμ_m_benchmark(t, x))

Λσ_um_benchmark(t, x) = 0.0 * [0.1 + cos(x[2]) 2]
Λσ_m_benchmark(t, x) = BENCHMARK_σ_scale * [0.0 5 + sin(x[2]) + 5.0*(norm(x) < 1 ? norm(x) : sqrt(norm(x)))]
Λσ_benchmark(t, x) = vcat(Λσ_um_benchmark(t, x), Λσ_m_benchmark(t, x))

function setup_double_integrator(config::DoubleIntegratorConfig)
    # Simulation Parameters
    Δ_saveat = 1e2 * config.Δₜ

    # System Dimensions
    system_dimensions = sys_dims(BENCHMARK_n, BENCHMARK_m, BENCHMARK_d)

    # Nominal Vector Fields (using module-scope functions)
    nominal_components = nominal_vector_fields(f_benchmark, g_benchmark, g_perp_benchmark, p_benchmark)

    # Uncertain Vector Fields (using module-scope functions)
    uncertain_components = uncertain_vector_fields(Λμ_benchmark, Λσ_benchmark)

    # Initial distributions
    nominal_ξ₀ = MvNormal(1e-2 * ones(BENCHMARK_n), 1 * I(BENCHMARK_n))
    true_ξ₀ = MvNormal(-1.0 * ones(BENCHMARK_n), 1e-1 * I(BENCHMARK_n))
    initial_distributions = init_dist(nominal_ξ₀, true_ξ₀)

    # Define the systems
    nominal_system = nom_sys(system_dimensions, nominal_components, initial_distributions)
    true_system = true_sys(system_dimensions, nominal_components, uncertain_components, initial_distributions)

    # L1 DRAC Parameters
    ω = 50.0
    Tₛ = 10 * config.Δₜ
    λₛ = 100.0
    L1params = drac_params(ω, Tₛ, λₛ)

    return (
        tspan = config.tspan,
        Δₜ = config.Δₜ,
        Δ_saveat = Δ_saveat,
        nominal_system = nominal_system,
        true_system = true_system,
        L1params = L1params
    )
end

# Convenience function with default config
function setup_double_integrator(; Δₜ=1e-4)
    config = DoubleIntegratorConfig(Δₜ, (0.0, 5.0))
    return setup_double_integrator(config)
end
