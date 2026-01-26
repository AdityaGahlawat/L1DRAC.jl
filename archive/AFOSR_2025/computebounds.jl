using JuMP, Ipopt
using Distributions

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

function Gamma_r(rho_r, ω, assumption_constants::AssumptionConstants, ref_sys_constants::RefSystemConstants)

    @unpack  λ, Δ_star   = assumption_constants
    @unpack Δr_circle_1, Δr_circle_4 = getfield(ref_sys_constants, :Δr_circle)
    @unpack Δr_circledcirc_1, Δr_circledcirc_4 = getfield(ref_sys_constants, :Δr_circledcirc)
    @unpack Δr_odot_1, Δr_odot_2, Δr_odot_3, Δr_odot_8  = getfield(ref_sys_constants, :Δr_odot)
    @unpack Δr_otimes_1 = getfield(ref_sys_constants, :Δr_otimes)

    c_0   = Δr_circle_1/(2λ) + (ω * Δr_circle_4) / (ω -2λ)
    c_half  = Δr_circledcirc_1/(2λ) + (ω * Δr_circledcirc_4 ) / (ω -2λ)
    c₁⁽¹⁾ = Δr_odot_1/(2λ) + (ω * Δr_odot_8 ) / (ω -2λ)
    c₁⁽²⁾ = Δr_odot_2/(2λ) + Δr_odot_3/(2√λ) + Δr_odot_1/(2λ)
    c3_half  = Δr_otimes_1/(2*sqrt(λ))

    return c_0 + c_half * (rho_r + Δ_star)^(1/2) + c₁⁽¹⁾ * (rho_r + Δ_star) +  c₁⁽²⁾ * rho_r +  c3_half  * (rho_r + Δ_star)^(3/2)
end   

function Gamma_a(rho_a, ω, assumption_constants::AssumptionConstants, true_sys_constants::TrueSystemConstants)

    @unpack λ = assumption_constants
    @unpack Δ_odot_1, Δ_odot_4 = getfield(true_sys_constants, :Δ_odot)
    @unpack Δ_otimes_1 = getfield(true_sys_constants, :Δ_otimes)

    c_0  = Δ_odot_1/(2*λ) + (ω * Δ_odot_4)/(ω-2*λ )
    c_half   = Δ_otimes_1/(2*sqrt(λ))

    return (c_0 + c_half *sqrt(rho_a))* rho_a
end

function Theta_r(rho_r, ω, assumption_constants,ref_sys_constants)

    @unpack λ, Δ_star = assumption_constants
    @unpack Δr_circle_2,  Δr_circle_3   = getfield(ref_sys_constants, :Δr_circle)
    @unpack Δr_circledcirc_2,  Δr_circledcirc_3   = getfield(ref_sys_constants, :Δr_circledcirc)
    @unpack Δr_odot_4,  Δr_odot_5,  Δr_odot_6,  Δr_odot_7 = getfield(ref_sys_constants, :Δr_odot)
    @unpack Δr_otimes_2, Δr_otimes_3, Δr_otimes_4, Δr_otimes_5 = getfield(ref_sys_constants, :Δr_otimes)
    @unpack Δr_ostar_2,  Δr_ostar_3 = getfield(ref_sys_constants, :Δr_ostar)

    term1 = (Δr_circle_2+ sqrt(ω) * Δr_circle_3)
    term2 = (Δr_circledcirc_2+ sqrt(ω) * Δr_circledcirc_3) * sqrt(rho_r + Δ_star)
    term3 = (Δr_odot_4 + sqrt(ω) * Δr_odot_6) * (rho_r + Δ_star)
    term4 = (Δr_odot_5 + sqrt(ω) * Δr_odot_7) * rho_r
    term5 = ((Δr_otimes_2 + sqrt(ω) * Δr_otimes_4) * (rho_r + Δ_star) +
             (Δr_otimes_3 + sqrt(ω) * Δr_otimes_5) * rho_r) * sqrt(rho_r + Δ_star)
    term6 = (Δr_ostar_2 * (rho_r + Δ_star) + Δr_ostar_3 * rho_r) * (rho_r + Δ_star)
    
    return (term1 + term2 + term3 + term4 + term5 + term6)/ (ω- 2λ)
end

function Theta_a(rho_a, rho_r, ω , assumption_constants, true_sys_constants)

    @unpack λ, Δ_star = assumption_constants
    
    @unpack Δ_circledcirc_1 = getfield(true_sys_constants, :Δ_circledcirc)
    @unpack Δ_odot_2, Δ_odot_3, Δ_odot_4 = getfield(true_sys_constants, :Δ_odot)
    @unpack Δ_otimes_2, Δ_otimes_3, Δ_otimes_4  = getfield(true_sys_constants, :Δ_otimes)
    @unpack Δ_ostar_2, Δ_ostar_3  = getfield(true_sys_constants, :Δ_ostar)
    
    ϱ_double_prime = rho_a + 2*(rho_r + Δ_star)

    term1 = (sqrt(ω) * Δ_circledcirc_1 +
             sqrt(ω) * Δ_otimes_3 * ϱ_double_prime +
             (Δ_otimes_2 + sqrt(ω) * Δ_otimes_4) * rho_a ) * sqrt(rho_a )

    term2 = (Δ_odot_2 +
             sqrt(ω) * Δ_odot_3 +
             Δ_otimes_2 * ϱ_double_prime +
             Δ_otimes_3 * rho_a ) * rho_a 

    return (term1 + term2)/ (ω - 2λ )
end

function filter_bandwidth_condition1(rho_r, ω)

    @unpack λ, Δg_perp, Δμ_perp, L_μ_perp = assumption_constants

    lhs_r = Theta_r(rho_r, ω, assumption_constants, ref_sys_constants) / (ω - 2*λ)
    rhs_r = (1 - (Δg_perp * Δμ_perp) / λ) * (rho_r^2) - α^2 - Gamma_r(rho_r,ω ,assumption_constants, ref_sys_constants)
    return rhs_r - lhs_r
end

function filter_bandwidth_condition2(rho_a, rho_r,  ω)

    @unpack λ, Δg_perp, Δμ_perp, L_μ_perp = assumption_constants

    lhs_a = Theta_a(rho_a, rho_r, ω, assumption_constants, true_sys_constants) / (ω - 2*λ)
    rhs_a = (1 - (Δg_perp * Δμ_perp) / λ) * (rho_a^2)  - Gamma_a(rho_a, ω ,assumption_constants, true_sys_constants)
    return rhs_a - lhs_a
end

function rho_r_condition(rho_r, ω)

    @unpack λ, Δg_perp, Δμ_perp, ϵ_r   = assumption_constants

    lhs = (1 - (Δg_perp * Δμ_perp ) / λ) * (rho_r^2)
    rhs =  α^2 + Gamma_r(rho_r,ω ,assumption_constants, ref_sys_constants) + ϵ_r

    return lhs - rhs
end

function rho_a_condition(rho_a, ω)

    @unpack λ, Δg_perp, L_μ_perp, ϵ_a = assumption_constants

    lhs = (1 - (Δg_perp * L_μ_perp) / λ) * (rho_a^2)
    rhs =  Gamma_a(rho_a, ω ,assumption_constants, true_sys_constants) + ϵ_a
    return lhs - rhs
end

function rho_and_filter_bandwidth_computation( α, assumption_constants::AssumptionConstants, ref_sys_constants::RefSystemConstants, true_sys_constants::TrueSystemConstants )
    model = Model(Ipopt.Optimizer)
    @unpack λ   = assumption_constants

    @variables model begin
        1.0   <= rho_a <= 1000.0
        1.0   <= rho_r <= 1000.0
        2*λ + 1e-2  <= ω  <= 1000.0  
    end

    register(model, :rho_r_condition, 2, rho_r_condition; autodiff = true)
    register(model, :rho_a_condition, 2, rho_a_condition; autodiff = true)
    register(model, :filter_bandwidth_condition1, 2, filter_bandwidth_condition1; autodiff = true)
    register(model, :filter_bandwidth_condition2, 3, filter_bandwidth_condition2; autodiff = true)

    @NLconstraint(model, rho_r_condition(rho_r, ω) >= 0)
    @NLconstraint(model, rho_a_condition(rho_a, ω) >= 0)
    @NLconstraint(model, filter_bandwidth_condition1(rho_r, ω) >= 0)
    @NLconstraint(model, filter_bandwidth_condition2(rho_a, rho_r, ω) >= 0)

    @objective(model, Min, rho_r + rho_a + ω)

    optimize!(model)

    rho = value(rho_r) + value(rho_a)

    return round(value(rho), digits=3), round(value(ω), digits=3)

end


