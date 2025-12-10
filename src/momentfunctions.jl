using Distributions: MvNormal, Normal, mean, cov, var
using LinearAlgebra: tr, dot

"""
    Delta_star_Linear_Nominal(M, Σ, sim_params, initial_Gaussian, moment_order)

Compute Δ* = max_t ‖X*(t)‖_{Lₚ} for linear Gaussian SDE.
Uses Euler method for mean/covariance ODEs, analytical Gaussian moments.

# Arguments
- `M`: Drift matrix (n×n)
- `Σ`: Diffusion matrix (n×d)
- `sim_params`: SimParams with tspan and Δₜ
- `initial_Gaussian`: MvNormal or Normal distribution
- `moment_order`: Moment order p (e.g., 2 for E[‖X‖²], 4 for E[‖X‖⁴])

# Returns
- `Δ_star`: The bound max_t E[‖X*(t)‖^p]^{1/p}
"""
function Delta_star_Linear_Nominal(M, Σ, sim_params::SimParams, initial_Gaussian::MvNormal, moment_order::Int)
    @unpack tspan, Δₜ = sim_params

    # Initial conditions from distribution
    μ = mean(initial_Gaussian)
    P = Matrix(cov(initial_Gaussian))

    # Precompute Σ Σᵀ
    ΣΣᵀ = Σ * Σ'

    # Track maximum of ‖X*(t)‖_{Lₚ}
    Δ_star = 0.0

    t = tspan[1]
    while t ≤ tspan[2]
        # Compute E[‖X‖^p]^{1/p} analytically
        Lp_norm = gaussian_Lp_norm(μ, P, moment_order)
        Δ_star = max(Δ_star, Lp_norm)

        # Euler step for mean: dμ/dt = M μ
        μ = μ + Δₜ * (M * μ)

        # Euler step for covariance: dP/dt = M P + P Mᵀ + Σ Σᵀ
        P = P + Δₜ * (M * P + P * M' + ΣΣᵀ)

        t += Δₜ
    end

    return Δ_star
end

# Dispatch for 1D Normal
function Delta_star_Linear_Nominal(M::Real, Σ::Real, sim_params::SimParams, initial_Gaussian::Normal, moment_order::Int)
    @unpack tspan, Δₜ = sim_params

    μ = mean(initial_Gaussian)
    P = var(initial_Gaussian)

    ΣΣᵀ = Σ^2

    Δ_star = 0.0

    t = tspan[1]
    while t ≤ tspan[2]
        Lp_norm = gaussian_Lp_norm_1d(μ, P, moment_order)
        Δ_star = max(Δ_star, Lp_norm)

        μ = μ + Δₜ * M * μ
        P = P + Δₜ * (2 * M * P + ΣΣᵀ)

        t += Δₜ
    end

    return Δ_star
end

"""
    gaussian_Lp_norm(μ, P, p)

Compute E[‖X‖^p]^{1/p} for X ~ N(μ, P).
"""
function gaussian_Lp_norm(μ, P, p::Int)
    moment = gaussian_norm_moment(μ, P, p)
    return moment^(1/p)
end

function gaussian_Lp_norm_1d(μ::Real, P::Real, p::Int)
    moment = gaussian_norm_moment_1d(μ, P, p)
    return moment^(1/p)
end

"""
    gaussian_norm_moment(μ, P, p)

Compute E[‖X‖^p] for X ~ N(μ, P) analytically.
"""
function gaussian_norm_moment(μ, P, p::Int)
    if p == 2
        # E[‖X‖²] = tr(P) + ‖μ‖²
        return tr(P) + dot(μ, μ)
    elseif p == 4
        # E[‖X‖⁴] using Isserlis theorem
        trP = tr(P)
        trP2 = tr(P^2)
        μPμ = dot(μ, P * μ)
        μ2 = dot(μ, μ)
        return trP^2 + 2*trP2 + 2*trP*μ2 + 4*μPμ + μ2^2
    else
        error("Analytical formula for moment order p=$p not implemented (only p=2,4 supported)")
    end
end

function gaussian_norm_moment_1d(μ::Real, P::Real, p::Int)
    if p == 2
        # E[X²] = P + μ²
        return P + μ^2
    elseif p == 4
        # E[X⁴] = 3P² + 6Pμ² + μ⁴
        return 3*P^2 + 6*P*μ^2 + μ^4
    else
        error("Analytical formula for moment order p=$p not implemented (only p=2,4 supported)")
    end
end
