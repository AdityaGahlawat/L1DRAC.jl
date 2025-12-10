# ====================================================================
# Alpha Zero (α₀) Computation - Initial Condition Distance
# ====================================================================

# Note: Required imports (LinearAlgebra, Distributions, UnPack, OptimalTransport)
# come from L1DRAC.jl module level

"""
    alpha_zero(init_dist::InitialDistributions, order::Int; mode::Symbol=:Wasserstein)

Compute α₀ = α(X^r(0), X*(0))_{order} for the initial condition distance.

# Arguments
- `init_dist::InitialDistributions` - Initial distributions (nominal_ξ₀, true_ξ₀)
- `order::Int` - Distance order (pass 2*order_p as per manuscript notation)
- `mode::Symbol` - Either `:Wasserstein` for W_{order}(ξ₀, ξ₀*) or `:Lp_norm` for ||X^r(0)-X*(0)||_{L_{order}}

# Returns
- `Float64` - The computed distance α₀

# Example
```julia
α = alpha_zero(initial_distributions, 2 * constants.order_p; mode=:Wasserstein)
```
"""
function alpha_zero(init_dist::InitialDistributions, order::Int; mode::Symbol=:Wasserstein)
    @unpack nominal_ξ₀, true_ξ₀ = init_dist

    @info "Computing α₀" mode order nominal_type=typeof(nominal_ξ₀) true_type=typeof(true_ξ₀)

    if mode == :Lp_norm
        return _compute_Lp_norm_diff(nominal_ξ₀, true_ξ₀, order)
    elseif mode == :Wasserstein
        return _compute_wasserstein(nominal_ξ₀, true_ξ₀, order)
    else
        throw(ArgumentError("mode must be :Lp_norm or :Wasserstein, got $mode"))
    end
end


# ====================================================================
# Wasserstein Distance Computation
# ====================================================================

"""
    _compute_wasserstein(μ, ν, order)

Compute W_order(μ, ν) - the order-Wasserstein distance between distributions μ and ν.
Auto-detects distribution types and uses the most efficient method available.
"""
function _compute_wasserstein(μ, ν, order::Int)
    # Check for Gaussian special case with closed form (W₂ only)
    if _is_gaussian(μ) && _is_gaussian(ν) && order == 2
        @info "Using closed-form Bures formula for W₂ (both distributions Gaussian)"
        return _w2_gaussians_closed_form(μ, ν)
    else
        @info "Using OptimalTransport.jl numerical solver for W_$order"
        return _wp_numerical(μ, ν, order)
    end
end

"""
Check if distribution is Gaussian (Normal or MvNormal).
"""
_is_gaussian(d) = d isa Normal || d isa MvNormal

"""
    _w2_gaussians_closed_form(μ, ν)

Compute W₂ between two Gaussian distributions using the closed-form Bures formula.

For X₁ ~ N(m₁, Σ₁) and X₂ ~ N(m₂, Σ₂):
W₂(X₁, X₂)² = ||m₁ - m₂||² + tr(Σ₁) + tr(Σ₂) - 2tr(√(√Σ₂ Σ₁ √Σ₂))
"""
function _w2_gaussians_closed_form(μ::MvNormal, ν::MvNormal)
    m1, Σ1 = mean(μ), Matrix(cov(μ))
    m2, Σ2 = mean(ν), Matrix(cov(ν))

    # Mean difference term
    dm = m1 .- m2
    term1 = dot(dm, dm)

    # Bures metric term for covariances
    Σ2_sqrt = sqrt(Symmetric(Σ2))
    inner = Σ2_sqrt * Σ1 * Σ2_sqrt
    term2 = tr(Σ1) + tr(Σ2) - 2 * tr(sqrt(Symmetric(inner)))

    return sqrt(term1 + term2)
end

# Dispatch for univariate Normal
function _w2_gaussians_closed_form(μ::Normal, ν::Normal)
    m1, σ1² = mean(μ), var(μ)
    m2, σ2² = mean(ν), var(ν)

    # W₂² = (m1 - m2)² + (σ1 - σ2)²
    term1 = (m1 - m2)^2
    term2 = (sqrt(σ1²) - sqrt(σ2²))^2

    return sqrt(term1 + term2)
end

"""
    _wp_numerical(μ, ν, order; n_samples=1000)

Compute W_order numerically using OptimalTransport.jl (emd solver).
Samples from distributions and solves discrete optimal transport.
Cost is ||x-y||₂^order (Euclidean norm raised to order).
"""
function _wp_numerical(μ, ν, order::Int; n_samples::Int=1000)
    # Sample from both distributions
    samples_μ = _sample_distribution(μ, n_samples)
    samples_ν = _sample_distribution(ν, n_samples)

    # Compute cost matrix: c(x,y) = ||x-y||₂^order
    n = size(samples_μ, 2)
    C = zeros(n, n)
    for i in 1:n, j in 1:n
        diff = samples_μ[:, i] .- samples_ν[:, j]
        euclidean_dist_sq = sum(diff.^2)           # ||x-y||₂²
        C[i, j] = euclidean_dist_sq^(order/2)      # ||x-y||₂^order
    end

    # Uniform weights (empirical distributions)
    a = ones(n) / n
    b = ones(n) / n

    # Solve optimal transport using emd from OptimalTransport.jl
    γ = emd(a, b, C)
    cost = dot(γ, C)
    return cost^(1/order)  # W_order = (optimal cost)^{1/order}
end

"""
Sample n points from a distribution, returning matrix of size (dim, n).
"""
function _sample_distribution(d::MvNormal, n::Int)
    return rand(d, n)  # Returns (dim, n) matrix
end

function _sample_distribution(d::Normal, n::Int)
    return reshape(rand(d, n), 1, n)  # Returns (1, n) matrix
end

# Generic fallback for other distribution types
function _sample_distribution(d, n::Int)
    if applicable(rand, d, n)
        samples = rand(d, n)
        if samples isa Vector
            return reshape(samples, 1, n)
        else
            return samples
        end
    else
        error("Don't know how to sample from distribution of type $(typeof(d))")
    end
end


# ====================================================================
# Lp Norm Difference Computation
# ====================================================================

"""
    _compute_Lp_norm_diff(μ, ν, order)

Compute ||X - Y||_{L_order} = E[||X-Y||₂^order]^{1/order} where X ~ μ, Y ~ ν.
For independent X, Y via analytical formulas (Gaussian) or Monte Carlo.
"""
function _compute_Lp_norm_diff(μ, ν, order::Int)
    if _is_gaussian(μ) && _is_gaussian(ν)
        @info "Using analytical Gaussian L_$order norm (X-Y ~ N(μ₁-μ₂, Σ₁+Σ₂))"
        return _Lp_norm_diff_gaussians(μ, ν, order)
    else
        @info "Using Monte Carlo estimation for L_$order norm"
        return _Lp_norm_diff_monte_carlo(μ, ν, order)
    end
end

"""
Compute ||X - Y||_{L_order} for Gaussians analytically.
If X ~ N(m1, Σ1) and Y ~ N(m2, Σ2) independent, then X - Y ~ N(m1-m2, Σ1+Σ2).
"""
function _Lp_norm_diff_gaussians(μ::MvNormal, ν::MvNormal, order::Int)
    # X - Y ~ N(m1 - m2, Σ1 + Σ2)
    m_diff = mean(μ) .- mean(ν)
    Σ_sum = Matrix(cov(μ)) .+ Matrix(cov(ν))

    # Use existing gaussian_Lp_norm function from momentfunctions.jl
    return gaussian_Lp_norm(m_diff, Σ_sum, order)
end

function _Lp_norm_diff_gaussians(μ::Normal, ν::Normal, order::Int)
    m_diff = mean(μ) - mean(ν)
    σ²_sum = var(μ) + var(ν)

    return gaussian_Lp_norm_1d(m_diff, σ²_sum, order)
end

"""
Monte Carlo estimation of ||X - Y||_{L_order} = E[||X-Y||₂^order]^{1/order}.
"""
function _Lp_norm_diff_monte_carlo(μ, ν, order::Int; n_samples::Int=10000)
    samples_μ = _sample_distribution(μ, n_samples)
    samples_ν = _sample_distribution(ν, n_samples)

    # Compute ||X_i - Y_i||₂^order for each sample pair
    diffs = samples_μ .- samples_ν
    euclidean_norms_sq = sum(diffs.^2, dims=1)       # ||x-y||₂² for each column
    euclidean_norms_order = euclidean_norms_sq.^(order/2)  # ||x-y||₂^order

    # E[||X-Y||₂^order]^{1/order}
    return mean(euclidean_norms_order)^(1/order)
end


# ====================================================================
# Gamma Functions (Γ_r and Γ_a)
# ====================================================================

"""
    Gamma_r(rho_r, ω, constants, ic)

Compute Γ_r(ρ_r, p, ω) for reference process bound (Eq. 3.8 in manuscript).

# Arguments
- `rho_r` - Reference process bound ρ_r
- `ω` - Filter bandwidth
- `constants::AssumptionConstants` - Contains λ, Δ_star
- `ic::IntermediateConstants` - Contains reference process Delta constants
"""
function Gamma_r(rho_r, ω, constants::AssumptionConstants, ic::IntermediateConstants)
    @unpack λ, Δ_star = constants
    ref = ic.reference

    # Extract needed constants
    Δr_circ_1 = ref.circ.Delta_r_circ_1
    Δr_circ_4 = ref.circ.Delta_r_circ_4
    Δr_circledcirc_1 = ref.circledcirc.Delta_r_circledcirc_1
    Δr_circledcirc_4 = ref.circledcirc.Delta_r_circledcirc_4
    Δr_odot_1 = ref.odot.Delta_r_odot_1
    Δr_odot_2 = ref.odot.Delta_r_odot_2
    Δr_odot_3 = ref.odot.Delta_r_odot_3
    Δr_odot_8 = ref.odot.Delta_r_odot_8
    Δr_otimes_1 = ref.otimes.Delta_r_otimes_1
    Δr_circledast_1 = ref.circledast.Delta_r_circledast_1

    # Coefficients (from manuscript Eq. 3.8)
    # Note: ω > 2λ is required, so (ω - 2λ) > 0
    c_0 = Δr_circ_1 / (2λ) + (ω * Δr_circ_4) / (ω - 2λ)
    c_half = Δr_circledcirc_1 / (2λ) + (ω * Δr_circledcirc_4) / (ω - 2λ)
    c_1_a = Δr_odot_1 / (2λ) + (ω * Δr_odot_8) / (ω - 2λ)
    c_1_b = Δr_odot_2 / (2λ) + Δr_odot_3 / (2 * sqrt(λ)) + (Δr_circledast_1 / (2λ)) * Δ_star
    c_3_half = Δr_otimes_1 / (2 * sqrt(λ))

    # Γ_r formula
    return (c_0 +
            c_half * sqrt(rho_r + Δ_star) +
            c_1_a * (rho_r + Δ_star) +
            c_1_b * rho_r +
            c_3_half * (rho_r + Δ_star)^(3/2))
end

"""
    Gamma_a(rho_a, ω, constants, ic)

Compute Γ_a(ρ_a, p, ω) for adaptation process bound (Eq. 3.9 in manuscript).

# Arguments
- `rho_a` - Adaptation process bound ρ_a
- `ω` - Filter bandwidth
- `constants::AssumptionConstants` - Contains λ
- `ic::IntermediateConstants` - Contains true process Delta constants
"""
function Gamma_a(rho_a, ω, constants::AssumptionConstants, ic::IntermediateConstants)
    @unpack λ = constants
    tru = ic.true_process

    # Extract needed constants
    Δ_odot_1 = tru.odot.Delta_odot_1
    Δ_odot_4 = tru.odot.Delta_odot_4
    Δ_otimes_1 = tru.otimes.Delta_otimes_1

    # Coefficients (from manuscript Eq. 3.9)
    c_0 = Δ_odot_1 / (2λ) + (ω * Δ_odot_4) / (ω - 2λ)
    c_half = Δ_otimes_1 / (2 * sqrt(λ))

    # Γ_a formula
    return (c_0 + c_half * sqrt(rho_a)) * rho_a
end


# ====================================================================
# Theta Functions (Θ_r and Θ_a) - Bandwidth Selection
# ====================================================================

"""
    Theta_r(rho_r, ω, constants, ic)

Compute Θ_r(ρ_r, p, ω) for reference process bandwidth condition (Eq. 3.10 in manuscript).

# Arguments
- `rho_r` - Reference process bound ρ_r
- `ω` - Filter bandwidth
- `constants::AssumptionConstants` - Contains λ, Δ_star
- `ic::IntermediateConstants` - Contains reference process Delta constants
"""
function Theta_r(rho_r, ω, constants::AssumptionConstants, ic::IntermediateConstants)
    @unpack λ, Δ_star = constants
    ref = ic.reference

    # Extract needed constants
    Δr_circ_2 = ref.circ.Delta_r_circ_2
    Δr_circ_3 = ref.circ.Delta_r_circ_3
    Δr_circledcirc_2 = ref.circledcirc.Delta_r_circledcirc_2
    Δr_circledcirc_3 = ref.circledcirc.Delta_r_circledcirc_3
    Δr_odot_4 = ref.odot.Delta_r_odot_4
    Δr_odot_5 = ref.odot.Delta_r_odot_5
    Δr_odot_6 = ref.odot.Delta_r_odot_6
    Δr_odot_7 = ref.odot.Delta_r_odot_7
    Δr_otimes_2 = ref.otimes.Delta_r_otimes_2
    Δr_otimes_3 = ref.otimes.Delta_r_otimes_3
    Δr_otimes_4 = ref.otimes.Delta_r_otimes_4
    Δr_otimes_5 = ref.otimes.Delta_r_otimes_5
    Δr_circledast_2 = ref.circledast.Delta_r_circledast_2
    Δr_circledast_3 = ref.circledast.Delta_r_circledast_3

    # Terms (from manuscript Eq. 3.10)
    term1 = Δr_circ_2 + sqrt(ω) * Δr_circ_3
    term2 = (Δr_circledcirc_2 + sqrt(ω) * Δr_circledcirc_3) * sqrt(rho_r + Δ_star)
    term3 = (Δr_odot_4 + sqrt(ω) * Δr_odot_6) * (rho_r + Δ_star)
    term4 = (Δr_odot_5 + sqrt(ω) * Δr_odot_7) * rho_r
    term5 = ((Δr_otimes_2 + sqrt(ω) * Δr_otimes_4) * (rho_r + Δ_star) +
             (Δr_otimes_3 + sqrt(ω) * Δr_otimes_5) * rho_r) * sqrt(rho_r + Δ_star)
    term6 = (Δr_circledast_2 * (rho_r + Δ_star) + Δr_circledast_3 * rho_r) * (rho_r + Δ_star)

    return (term1 + term2 + term3 + term4 + term5 + term6) / (ω - 2λ)
end

"""
    Theta_a(rho_a, rho_r, ω, constants, ic)

Compute Θ_a(ρ_a, ρ_r, p, ω) for adaptation process bandwidth condition (Eq. 3.11 in manuscript).

# Arguments
- `rho_a` - Adaptation process bound ρ_a
- `rho_r` - Reference process bound ρ_r
- `ω` - Filter bandwidth
- `constants::AssumptionConstants` - Contains λ, Δ_star
- `ic::IntermediateConstants` - Contains true process Delta constants
"""
function Theta_a(rho_a, rho_r, ω, constants::AssumptionConstants, ic::IntermediateConstants)
    @unpack λ, Δ_star = constants
    tru = ic.true_process

    # Extract needed constants
    Δ_circledcirc_1 = tru.circledcirc.Delta_circledcirc_1
    Δ_odot_2 = tru.odot.Delta_odot_2
    Δ_odot_3 = tru.odot.Delta_odot_3
    Δ_otimes_2 = tru.otimes.Delta_otimes_2
    Δ_otimes_3 = tru.otimes.Delta_otimes_3
    Δ_otimes_4 = tru.otimes.Delta_otimes_4
    Δ_circledast_2 = tru.circledast.Delta_circledast_2
    Δ_circledast_3 = tru.circledast.Delta_circledast_3

    # ρ'' = ρ_a + 2(ρ_r + Δ_★)
    ρ_double_prime = rho_a + 2 * (rho_r + Δ_star)

    # Terms (from manuscript Eq. 3.11)
    term1 = (sqrt(ω) * Δ_circledcirc_1 +
             sqrt(ω) * Δ_otimes_3 * ρ_double_prime +
             (Δ_otimes_2 + sqrt(ω) * Δ_otimes_4) * rho_a) * sqrt(rho_a)

    term2 = (Δ_odot_2 +
             sqrt(ω) * Δ_odot_3 +
             Δ_circledast_2 * ρ_double_prime +
             Δ_circledast_3 * rho_a) * rho_a

    return (term1 + term2) / (ω - 2λ)
end


# ====================================================================
# Constraint Functions for Optimization
# ====================================================================

"""
    rho_r_condition(rho_r, ω, α, constants, ic, ε_r)

Evaluate ρ_r feasibility condition. Returns ≥ 0 when satisfied.

Condition: (1 - Δg⊥·Δμ⊥/λ)·ρ_r² ≥ α² + Γ_r(ρ_r, ω) + ε_r
"""
function rho_r_condition(rho_r, ω, α, constants::AssumptionConstants, ic::IntermediateConstants, ε_r)
    @unpack λ, Δg_perp, Δμ_perp = constants

    lhs = (1 - (Δg_perp * Δμ_perp) / λ) * rho_r^2
    rhs = α^2 + Gamma_r(rho_r, ω, constants, ic) + ε_r

    return lhs - rhs
end

"""
    rho_a_condition(rho_a, ω, constants, ic, ε_a)

Evaluate ρ_a feasibility condition. Returns ≥ 0 when satisfied.

Condition: (1 - Δg⊥·L_μ⊥/λ)·ρ_a² ≥ Γ_a(ρ_a, ω) + ε_a
"""
function rho_a_condition(rho_a, ω, constants::AssumptionConstants, ic::IntermediateConstants, ε_a)
    @unpack λ, Δg_perp, L_μ_perp = constants

    lhs = (1 - (Δg_perp * L_μ_perp) / λ) * rho_a^2
    rhs = Gamma_a(rho_a, ω, constants, ic) + ε_a

    return lhs - rhs
end

"""
    bandwidth_condition_r(rho_r, ω, α, constants, ic)

Evaluate bandwidth condition for reference process. Returns ≥ 0 when satisfied.

Condition: Θ_r/(ω-2λ) < (1 - Δg⊥·Δμ⊥/λ)·ρ_r² - α² - Γ_r
"""
function bandwidth_condition_r(rho_r, ω, α, constants::AssumptionConstants, ic::IntermediateConstants)
    @unpack λ, Δg_perp, Δμ_perp = constants

    lhs = Theta_r(rho_r, ω, constants, ic) / (ω - 2λ)
    rhs = (1 - (Δg_perp * Δμ_perp) / λ) * rho_r^2 - α^2 - Gamma_r(rho_r, ω, constants, ic)

    return rhs - lhs
end

"""
    bandwidth_condition_a(rho_a, rho_r, ω, constants, ic)

Evaluate bandwidth condition for adaptation process. Returns ≥ 0 when satisfied.

Condition: Θ_a/(ω-2λ) < (1 - Δg⊥·L_μ⊥/λ)·ρ_a² - Γ_a
"""
function bandwidth_condition_a(rho_a, rho_r, ω, constants::AssumptionConstants, ic::IntermediateConstants)
    @unpack λ, Δg_perp, L_μ_perp = constants

    lhs = Theta_a(rho_a, rho_r, ω, constants, ic) / (ω - 2λ)
    rhs = (1 - (Δg_perp * L_μ_perp) / λ) * rho_a^2 - Gamma_a(rho_a, ω, constants, ic)

    return rhs - lhs
end


# ====================================================================
# Main Optimization Function
# ====================================================================

"""
    optimal_bounds(constants, ic, init_dist; α_mode=:Wasserstein, ε_r=1e-3, ε_a=1e-3)

Compute optimal (ρ_r, ρ_a, ω) via nonlinear optimization using JuMP + Ipopt.

# Arguments
- `constants::AssumptionConstants` - System constants including λ, Δ_star, etc.
- `ic::IntermediateConstants` - Precomputed intermediate constants
- `init_dist::InitialDistributions` - Initial distributions for α₀ computation

# Keyword Arguments
- `α_mode::Symbol` - Mode for α₀ computation (:Wasserstein or :Lp_norm)
- `ε_r::Float64` - Tolerance for ρ_r condition (default 1e-3)
- `ε_a::Float64` - Tolerance for ρ_a condition (default 1e-3)

# Returns
Named tuple `(rho_r, rho_a, rho_total, omega, α, status)`
"""
function optimal_bounds(constants::AssumptionConstants,
                        ic::IntermediateConstants,
                        init_dist::InitialDistributions;
                        α_mode::Symbol = :Wasserstein,
                        ε_r::Float64 = 1e-3,
                        ε_a::Float64 = 1e-3)

    @unpack λ = constants
    order = 2 * constants.order_p

    # Compute α₀
    α = alpha_zero(init_dist, order; mode=α_mode)
    @info "Computed α₀" α

    # Create JuMP model with Ipopt
    model = Model(Ipopt.Optimizer)
    set_silent(model)

    # Variables with bounds
    # ω must be > 2λ for stability
    @variable(model, 1.0 <= rho_r <= 10000.0)
    @variable(model, 1.0 <= rho_a <= 10000.0)
    @variable(model, 2λ + 0.1 <= ω <= 10000.0)

    # Create closure functions for JuMP (capture constants, ic, α, ε)
    _rho_r_cond(rho_r, ω) = rho_r_condition(rho_r, ω, α, constants, ic, ε_r)
    _rho_a_cond(rho_a, ω) = rho_a_condition(rho_a, ω, constants, ic, ε_a)
    _bw_cond_r(rho_r, ω) = bandwidth_condition_r(rho_r, ω, α, constants, ic)
    _bw_cond_a(rho_a, rho_r, ω) = bandwidth_condition_a(rho_a, rho_r, ω, constants, ic)

    # Register nonlinear functions
    register(model, :rho_r_cond, 2, _rho_r_cond; autodiff=true)
    register(model, :rho_a_cond, 2, _rho_a_cond; autodiff=true)
    register(model, :bw_cond_r, 2, _bw_cond_r; autodiff=true)
    register(model, :bw_cond_a, 3, _bw_cond_a; autodiff=true)

    # Constraints (all must be ≥ 0)
    @NLconstraint(model, rho_r_cond(rho_r, ω) >= 0)
    @NLconstraint(model, rho_a_cond(rho_a, ω) >= 0)
    @NLconstraint(model, bw_cond_r(rho_r, ω) >= 0)
    @NLconstraint(model, bw_cond_a(rho_a, rho_r, ω) >= 0)

    # Objective: minimize total bound (ρ_r + ρ_a) and bandwidth
    @objective(model, Min, rho_r + rho_a + ω)

    # Solve
    @info "Starting optimization..."
    optimize!(model)

    status = termination_status(model)
    @info "Optimization finished" status

    if status == MOI.LOCALLY_SOLVED || status == MOI.OPTIMAL
        result = (
            rho_r = value(rho_r),
            rho_a = value(rho_a),
            rho_total = value(rho_r) + value(rho_a),
            omega = value(ω),
            α = α,
            status = status
        )
        @info "Optimal values found" result.rho_r result.rho_a result.rho_total result.omega
        return result
    else
        @warn "Optimization did not converge" status
        return (
            rho_r = NaN,
            rho_a = NaN,
            rho_total = NaN,
            omega = NaN,
            α = α,
            status = status
        )
    end
end
