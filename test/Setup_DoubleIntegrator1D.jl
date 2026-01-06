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


# System constants
const n = 2
const m = 1
const d = 2
const λ = 10.0
const σ_scale = 1.0

# Config struct for variable parameters
struct DoubleIntegratorConfig
    Δₜ::Float64
    tspan::Tuple{Float64, Float64}
end

# Module scope functions - required for GPU kernel compilation

function trck_traj(t)
    return @SVector [5*sin(t) + 3*cos(2*t), 0]
end

const A = @SMatrix [0 1.0; 0 0]
const B = @SMatrix [0; 1.0]
const C = SMatrix{n,n}(1.0I)
const D = 0.0

function stbl_cntrl()
    sys = ss(A, B, C, D)
    DesiredPoles = -λ * ones(2)
    return SMatrix{m,n}(place(sys, DesiredPoles))
end
const K = stbl_cntrl()

const A_cl = A - B*K # Closed-loop Matrix 

function f(t, x)
    return A_cl*x + B*K*trck_traj(t)
end

g(t) = @SVector [0, 1]
g_perp(t) = @SVector [1, 0]
g_bar(t) = hcat([g(t) g_perp(t)])
Θ_ad(t) = hcat(SMatrix{m,m}(1.0I), SMatrix{m,n-m}(zeros(m,n-m))) * inv(g_bar(t))

p_um(t, x) = 2.0 * @SMatrix [0.01 0.1]
p_m(t, x) = 1.0 * @SMatrix [0.0 0.8]
p(t, x) = vcat(p_um(t, x), p_m(t, x))

Λμ_um(t, x) = 1e-2 * (1 + sin(x[1]))
Λμ_m(t, x) = 1.0 * (5 + 10*cos(x[2]) + 5*norm(x))
Λμ(t, x) = SVector{n}(vcat(Λμ_um(t, x), Λμ_m(t, x)))

Λσ_um(t, x) =  0.0 * SMatrix{(n-m),d}([0.1 + cos(x[2]) 2])
Λσ_m(t, x) = σ_scale * SMatrix{m,d}([0.0 5 + sin(x[2]) + 5.0*(norm(x) < 1 ? norm(x) : sqrt(norm(x)))])
Λσ(t, x) = vcat(Λσ_um(t, x), Λσ_m(t, x))

function setup_double_integrator(config::DoubleIntegratorConfig)
    # Simulation Parameters
    Δ_saveat = 1e2 * config.Δₜ

    # System Dimensions
    system_dimensions = sys_dims(n, m, d)

    # Nominal Vector Fields (using module-scope functions)
    nominal_components = nominal_vector_fields(f, g, g_perp, p)

    # Uncertain Vector Fields (using module-scope functions)
    uncertain_components = uncertain_vector_fields(Λμ, Λσ)

    # Initial distributions
    nominal_ξ₀ = MvNormal(1e-2 * ones(n), 1 * I(n))
    true_ξ₀ = MvNormal(-1.0 * ones(n), 1e-1 * I(n))
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
