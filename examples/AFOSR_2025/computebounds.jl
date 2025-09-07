
function alpha(initial_distributions)
    @unpack nominal_ξ₀, true_ξ₀ = initial_distributions

    if nominal_ξ₀ isa Normal && true_ξ₀ isa Normal
        μ1, σ1² = mean(nominal_ξ₀), var(nominal_ξ₀)
        μ2, σ2² = mean(true_ξ₀),    var(true_ξ₀)
        return w2_gaussians([μ1], reshape(σ1²,1,1), [μ2], reshape(σ2²,1,1))
    
    elseif nominal_ξ₀ isa MvNormal && true_ξ₀ isa MvNormal
        μ1, Σ1 = mean(nominal_ξ₀), cov(nominal_ξ₀)
        μ2, Σ2 = mean(true_ξ₀),    cov(true_ξ₀)
        return w2_gaussians(μ1, Σ1, μ2, Σ2)
    else
        throw(ArgumentError("alpha currently only supports Normal and MvNormal; got $(typeof(nominal_ξ₀)) and $(typeof(true_ξ₀))."))
    end
end

function Gamma_r(ρᵣ::Float64, assumption_constants::AssumptionConstants, ref_sys_constants::RefSystemConstants , L1params)

    @unpack p, λ, Δ_star   = assumption_constants
    @unpack Δr_circle_1, Δr_circle_4 = getfield(ref_sys_constants, :Δr_circle)
    @unpack Δr_circledcirc_1, Δr_circledcirc_4 = getfield(ref_sys_constants, :Δr_circledcirc)
    @unpack Δr_odot_1, Δr_odot_2, Δr_odot_3, Δr_odot_8  = getfield(ref_sys_constants, :Δr_odot)
    @unpack Δr_otimes_1 = getfield(ref_sys_constants, :Δr_otimes)
    @unpack ω = L1params

    c_0   = Δr_circle_1/(2λ) + (ω * Δr_circle_4) / abs(2λ - ω)
    c_half  = Δr_circledcirc_1/(2λ) + (ω * Δr_circledcirc_4 ) / abs(2λ - ω)
    c₁⁽¹⁾ = Δr_odot_1/(2λ) + (ω * Δr_odot_8 ) / abs(2λ - ω)
    c₁⁽²⁾ = Δr_odot_2/(2λ) + Δr_odot_3/(2√λ) + Δr_odot_1/(2λ)
    c3_half  = Δr_otimes_1/(2*sqrt(λ))

    return c_0 + c_half * (ρᵣ + Δ_star)^(1/2) + c₁⁽¹⁾ * (ρᵣ + Δ_star) +  c₁⁽²⁾ * ρᵣ +  c3_half  * (ρᵣ + Δ_star)^(3/2)
end     

function Gamma_a(ρₐ::Float64, assumption_constants::AssumptionConstants, true_sys_constants::TrueSystemConstants , L1params)

    @unpack λ = assumption_constants
    @unpack Δ_odot_1, Δ_odot_4 = getfield(true_sys_constants, :Δ_odot)
    @unpack Δ_otimes_1 = getfield(true_sys_constants, :Δ_otimes)
    @unpack ω = L1params

    c_0  = Δ_odot_1/(2*λ) + (ω * Δ_odot_4)/abs(2*λ - ω)
    c_half   = Δ_otimes_1/(2*sqrt(λ))

    return (c_0 + c_half *sqrt(ρₐ))* ρₐ
end

function ρᵣ_condition(ρᵣ::Float64; initial_distributions, assumption_constants::AssumptionConstants, ref_sys_constants::RefSystemConstants, L1params )
    

    @unpack λ, Δg_perp, Δμ_perp, ϵ_r   = assumption_constants

    lhs = (1 - (Δg_perp * Δμ_perp ) / λ) * (ρᵣ^2)
    rhs =  alpha(initial_distributions)^2 + Gamma_r(ρᵣ,assumption_constants, ref_sys_constants, L1params) + ϵ_r
    return lhs ≥ rhs
end

function ρₐ_condition(ρₐ::Float64; assumption_constants::AssumptionConstants, true_sys_constants::TrueSystemConstants, L1params )
    
    @unpack λ, Δg_perp, L_μ_perp, ϵ_a = assumption_constants

    lhs = (1 - (Δg_perp * L_μ_perp) / λ) * (ρₐ^2)
    rhs =  Gamma_a(ρₐ, assumption_constants, true_sys_constants, L1params ) + ϵ_a
    return lhs ≥ rhs
end

function binary_search(cond::Function; ρ_lb=0.0, ρ_ub=100.0, tol=0.05, maxiter=1000, kwargs...)
    lb, ub = ρ_lb, ρ_ub
    @assert lb < ub
    @assert cond(ub; kwargs...) "No feasible ρ in [ρ_lb, ρ_ub]"
    it = 0
    while (ub - lb) ≥ tol && it < maxiter
        mid = (lb + ub) / 2
        if cond(mid; kwargs...)
            ub = mid
        else
            lb = mid
        end
        it += 1
    end
    return ub
end

function find_rho(initial_distributions,assumption_constants::AssumptionConstants,
            true_sys_constants::TrueSystemConstants ,ref_sys_constants::RefSystemConstants, L1params)
    ρᵣ = binary_search(ρᵣ_condition; initial_distributions, assumption_constants, ref_sys_constants, L1params)
    ρₐ = binary_search(ρₐ_condition; assumption_constants, true_sys_constants, L1params)
    return ρᵣ, ρₐ, round(ρᵣ + ρₐ; digits=3)
end

function Theta_r(ρᵣ::Float64,
                  assumption_constants::AssumptionConstants,
                  ref_sys_constants::RefSystemConstants,
                  L1params)

    @unpack Δ_star = assumption_constants
    @unpack ω = L1params
    @unpack Δr_circle_2,  Δr_circle_3   = getfield(ref_sys_constants, :Δr_circle)
    @unpack Δr_circledcirc_2,  Δr_circledcirc_3   = getfield(ref_sys_constants, :Δr_circledcirc)
    @unpack Δr_odot_4,  Δr_odot_5,  Δr_odot_6,  Δr_odot_7 = getfield(ref_sys_constants, :Δr_odot)
    @unpack Δr_otimes_2, Δr_otimes_3, Δr_otimes_4, Δr_otimes_5 = getfield(ref_sys_constants, :Δr_otimes)
    @unpack Δr_ostar_2,  Δr_ostar_3 = getfield(ref_sys_constants, :Δr_ostar)

    term1 = (Δr_circle_2+ sqrt(ω) * Δr_circle_3)
    term2 = (Δr_circledcirc_2+ sqrt(ω) * Δr_circledcirc_3) * sqrt(ρᵣ + Δ_star)
    term3 = (Δr_odot_4 + sqrt(ω) * Δr_odot_6) * (ρᵣ + Δ_star)
    term4 = (Δr_odot_5 + sqrt(ω) * Δr_odot_7) * ρᵣ
    term5 = ((Δr_otimes_2 + sqrt(ω) * Δr_otimes_4) * (ρᵣ + Δ_star) +
             (Δr_otimes_3 + sqrt(ω) * Δr_otimes_5) * ρᵣ) * sqrt(ρᵣ + Δ_star)
    term6 = (Δr_ostar_2 * (ρᵣ + Δ_star) + Δr_ostar_3 * ρᵣ) * (ρᵣ + Δ_star)
    
    return (term1 + term2 + term3 + term4 + term5 + term6)/ abs(2λ - ω)
end

function Theta_a(ρₐ::Float64,
                  assumption_constants::AssumptionConstants,
                  true_sys_constants::TrueSystemConstants,
                  L1params,
                  ρᵣ::Float64)

    @unpack Δ_star = assumption_constants
    @unpack ω = L1params
    
    @unpack Δ_circledcirc_1 = getfield(true_sys_constants, :Δ_circledcirc)
    @unpack Δ_odot_2, Δ_odot_3, Δ_odot_4 = getfield(true_sys_constants, :Δ_odot)
    @unpack Δ_otimes_2, Δ_otimes_3, Δ_otimes_4  = getfield(true_sys_constants, :Δ_otimes)
    @unpack Δ_ostar_2, Δ_ostar_3  = getfield(true_sys_constants, :Δ_ostar)
    
    ϱ_double_prime = ρₐ + 2*(ρᵣ + Δ_star)

    term1 = (sqrt(ω) * Δ_circledcirc_1 +
             sqrt(ω) * Δ_otimes_3 * ϱ_double_prime +
             (Δ_otimes_2 + sqrt(ω) * Δ_otimes_4) * ρₐ) * sqrt(ρₐ)

    term2 = (Δ_odot_2 +
             sqrt(ω) * Δ_odot_3 +
             Δ_otimes_2 * ϱ_double_prime +
             Δ_otimes_3 * ρₐ) * ρₐ

    return (term1 + term2)/ abs(2λ - ω)
end

function filter_bandwidth_conditions(ρᵣ::Float64, ρₐ::Float64,
                      initial_distributions,
                      assumption_constants::AssumptionConstants,
                      ref_sys_constants::RefSystemConstants,
                      true_sys_constants::TrueSystemConstants,
                      L1params)

    @unpack λ, Δg_perp, Δμ_perp, L_μ_perp = assumption_constants
    @unpack ω = L1params

    lhs_r = Theta_r(ρᵣ, assumption_constants, ref_sys_constants, L1params) / abs(2*λ - ω)
    rhs_r = (1 - (Δg_perp * Δμ_perp) / λ) * (ρᵣ^2) - alpha(initial_distributions)^2 - Gamma_r(ρᵣ, assumption_constants, ref_sys_constants, L1params)
    
    lhs_a = Theta_a(ρₐ, assumption_constants, true_sys_constants, L1params, ρᵣ) / abs(2*λ - ω)
    rhs_a = (1 - (Δg_perp * L_μ_perp) / λ) * (ρₐ^2) - Gamma_a(ρₐ, assumption_constants, true_sys_constants, L1params)
    return (lhs_r < rhs_r) && (lhs_a < rhs_a)
end









