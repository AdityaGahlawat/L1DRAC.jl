using JuMP, Ipopt
using Distributions

function alpha_computation(initial_distributions::InitialDistributions,system_dimensions::SysDims, Ntraj::Int)
    @unpack nominal_ξ₀, true_ξ₀ = initial_distributions

    empirical_samples= EmpiricalSamples(rand(nominal_ξ₀,  Ntraj),rand(true_ξ₀,  Ntraj) )

    return empirical_wasserstein2(empirical_samples, system_dimensions) 
    
end

function Gamma_r(rho_r, ω, assumption_constants::AssumptionConstants, ref_system_constants::RefSystemConstants)

    @unpack  λ, Δ_star   = assumption_constants
    @unpack Δr_circle_1, Δr_circle_4 = getfield(ref_system_constants, :Δr_circle)
    @unpack Δr_circledcirc_1, Δr_circledcirc_4 = getfield(ref_system_constants, :Δr_circledcirc)
    @unpack Δr_odot_1, Δr_odot_2, Δr_odot_3, Δr_odot_8  = getfield(ref_system_constants, :Δr_odot)
    @unpack Δr_otimes_1 = getfield(ref_system_constants, :Δr_otimes)
    @unpack Δr_ostar_1= getfield(ref_system_constants, :Δr_ostar)

    c_0   = Δr_circle_1/(2λ) + (ω * Δr_circle_4) / (ω -2λ)
    c_half  = Δr_circledcirc_1/(2λ) + (ω * Δr_circledcirc_4 ) / (ω -2λ)
    c₁⁽¹⁾ = Δr_odot_1/(2λ) + (ω * Δr_odot_8 ) / (ω -2λ)
    c₁⁽²⁾ = Δr_odot_2/(2λ) + Δr_odot_3/(2*sqrt(λ)) + (Δr_ostar_1*Δ_star)/(2λ)
    c3_half  = Δr_otimes_1/(2*sqrt(λ))

    return c_0 + c_half * (rho_r + Δ_star)^(1/2) + c₁⁽¹⁾ * (rho_r + Δ_star) +  c₁⁽²⁾ * rho_r +  c3_half  * (rho_r + Δ_star)^(3/2)
end   

function Gamma_a(rho_a, ω, assumption_constants::AssumptionConstants, true_system_constants::TrueSystemConstants)

    @unpack λ = assumption_constants
    @unpack Δ_odot_1, Δ_odot_4 = getfield(true_system_constants, :Δ_odot)
    @unpack Δ_otimes_1 = getfield(true_system_constants, :Δ_otimes)

    c_0  = Δ_odot_1/(2*λ) + (ω * Δ_odot_4)/(ω-2*λ )
    c_half   = Δ_otimes_1/(2*sqrt(λ))

    return (c_0 + c_half *sqrt(rho_a))* rho_a
end

function Theta_r(rho_r, ω, assumption_constants,ref_system_constants)

    @unpack λ, Δ_star = assumption_constants
    @unpack Δr_circle_2,  Δr_circle_3   = getfield(ref_system_constants, :Δr_circle)
    @unpack Δr_circledcirc_2,  Δr_circledcirc_3   = getfield(ref_system_constants, :Δr_circledcirc)
    @unpack Δr_odot_4,  Δr_odot_5,  Δr_odot_6,  Δr_odot_7 = getfield(ref_system_constants, :Δr_odot)
    @unpack Δr_otimes_2, Δr_otimes_3, Δr_otimes_4, Δr_otimes_5 = getfield(ref_system_constants, :Δr_otimes)
    @unpack Δr_ostar_2,  Δr_ostar_3 = getfield(ref_system_constants, :Δr_ostar)

    term1 = (Δr_circle_2+ sqrt(ω) * Δr_circle_3)
    term2 = (Δr_circledcirc_2+ sqrt(ω) * Δr_circledcirc_3) * sqrt(rho_r + Δ_star)
    term3 = (Δr_odot_4 + sqrt(ω) * Δr_odot_6) * (rho_r + Δ_star)
    term4 = (Δr_odot_5 + sqrt(ω) * Δr_odot_7) * rho_r
    term5 = ((Δr_otimes_2 + sqrt(ω) * Δr_otimes_4) * (rho_r + Δ_star) +
             (Δr_otimes_3 + sqrt(ω) * Δr_otimes_5) * rho_r) * sqrt(rho_r + Δ_star)
    term6 = (Δr_ostar_2 * (rho_r + Δ_star) + Δr_ostar_3 * rho_r) * (rho_r + Δ_star)
    
    return (term1 + term2 + term3 + term4 + term5 + term6)
end

function Theta_a(rho_a, rho_r, ω , assumption_constants, true_system_constants)

    @unpack λ, Δ_star = assumption_constants
    
    @unpack Δ_circledcirc_1 = getfield(true_system_constants, :Δ_circledcirc)
    @unpack Δ_odot_2, Δ_odot_3, Δ_odot_4 = getfield(true_system_constants, :Δ_odot)
    @unpack Δ_otimes_2, Δ_otimes_3, Δ_otimes_4  = getfield(true_system_constants, :Δ_otimes)
    @unpack Δ_ostar_2, Δ_ostar_3  = getfield(true_system_constants, :Δ_ostar)
    
    ϱ_double_prime = rho_a + 2*(rho_r + Δ_star)

    term1 = (sqrt(ω) * Δ_circledcirc_1 +
             sqrt(ω) * Δ_otimes_3 * ϱ_double_prime +
             (Δ_otimes_2 + sqrt(ω) * Δ_otimes_4) * rho_a ) * sqrt(rho_a )

    term2 = (Δ_odot_2 +
             sqrt(ω) * Δ_odot_3 +
             Δ_ostar_2 * ϱ_double_prime +
             Δ_ostar_3 * rho_a ) * rho_a 

    return term1 + term2
end

function rhoHat(t_hat::Float64,rho_r::Float64 ,rho_a::Float64, 
                L2p_norm_x0_xr0::Float64, assumption_constants::AssumptionConstants, ref_system_constants::RefSystemConstants, L1params)
    @unpack  λ, Δg_perp, Δμ_perp = assumption_constants
    @unpack ω = L1params
    
    rhohat_r = sqrt( exp(-2*λ*t_hat) * L2p_norm_x0_xr0^2 +  (Δg_perp * Δμ_perp / λ) * (rho_r^2)
               +  Gamma_r(rho_r,ω ,assumption_constants, ref_system_constants) + (1 / abs(2*λ - ω)) * Theta_r(rho_r, ω, assumption_constants, ref_system_constants) )
    return rho_a + rhohat_r
end


function Upsilon_a1(ξ1::Float64, ξ2::Float64, Ts::Float64, assumption_constants::AssumptionConstants, L1params)
    @unpack λ = assumption_constants
    @unpack ω = L1params
    return UpsilonTildeMinus(ξ1, Ts, assumption_constants, L1params) * ξ2 +
           (ω / (λ * abs(2*λ - ω))) * UpsilonDot(ξ2, assumption_constants, L1params) * UpsilonMinus(ξ1, Ts, assumption_constants, L1params)
end

function Upsilon_a2(ξ1::Float64, ξ2::Float64, Ts::Float64, assumption_constants::AssumptionConstants, L1params)
    @unpack order_p, λ, Δg = assumption_constants
    @unpack ω = L1params
    return (1/λ) * (Δg*ξ2 + (sqrt(ω) / abs(2*λ - ω)) * UpsilonDot(ξ2, assumption_constants, L1params)) * 
            (UpsilonPrime_1(ξ1, Ts, assumption_constants, L1params) + sqrt(ω) * UpsilonPrime_2(ξ1, Ts, assumption_constants, L1params))
end

function Upsilon_a3(ξ1::Float64, ξ2::Float64, Ts::Float64, assumption_constants::AssumptionConstants, L1params)
    @unpack order_p, λ, Δg = assumption_constants
    @unpack ω = L1params
    return (1 / abs(2*λ - ω)) * (2*Δg*ξ2 + (sqrt(ω)/λ) * UpsilonDot(ξ2, assumption_constants, L1params)) * UpsilonPrime_3(ξ1, Ts, assumption_constants, L1params)
end


function filter_bandwidth_condition1(rho_r, ω)

    @unpack λ, Δg_perp, Δμ_perp, L_μ_perp = assumption_constants

    lhs = Theta_r(rho_r, ω, assumption_constants, ref_system_constants) / (ω - 2*λ)
    rhs = (1 - (Δg_perp * Δμ_perp) / λ) * (rho_r^2) - α^2 - Gamma_r(rho_r,ω ,assumption_constants, ref_system_constants)
    return rhs - lhs
end

function filter_bandwidth_condition2(rho_a, rho_r,  ω)

    @unpack λ, Δg_perp, Δμ_perp, L_μ_perp = assumption_constants

    lhs = Theta_a(rho_a, rho_r, ω, assumption_constants, true_system_constants) / (ω - 2*λ)
    rhs = (1 - (Δg_perp * L_μ_perp ) / λ) * (rho_a^2)  - Gamma_a(rho_a, ω ,assumption_constants, true_system_constants)
    return rhs - lhs
end

function sampling_period_condition(rho_a, rho_r,  ω, Ts)

    @unpack λ, Δg_perp, Δμ_perp, L_μ_perp, Δ_star   = assumption_constants

    rho_prime= rho_a + rho_r + Δ_star

    lhs = Upsilon_a1(rho_prime, rho_a, Ts, assumption_constants, L1params)
          +  Upsilon_a2(rho_prime, rho_a, Ts, assumption_constants, L1params)
            +  Upsilon_a3(rho_prime, rho_a, Ts, assumption_constants, L1params)
    rhs = (1 - (Δg_perp * Δμ_perp) / λ) * (rho_a^2)  - Gamma_a(rho_a, ω ,assumption_constants, true_system_constants) 
             - Theta_a(rho_a, rho_r, ω, assumption_constants, true_system_constants) / (ω - 2*λ)
    return rhs - lhs
end

function rho_r_condition(rho_r, ω)

    @unpack λ, Δg_perp, Δμ_perp, ϵ_r   = assumption_constants

    lhs = (1 - (Δg_perp * Δμ_perp ) / λ) * (rho_r^2)
    rhs =  α^2 + Gamma_r(rho_r,ω ,assumption_constants, ref_system_constants) + ϵ_r

    return lhs - rhs
end

function rho_a_condition(rho_a, ω)

    @unpack λ, Δg_perp, L_μ_perp, ϵ_a = assumption_constants

    lhs = (1 - (Δg_perp * L_μ_perp) / λ) * (rho_a^2)
    rhs =  Gamma_a(rho_a, ω ,assumption_constants, true_system_constants) + ϵ_a
    return lhs - rhs
end

function rho_and_filter_bandwidth_computation( α, assumption_constants::AssumptionConstants, ref_system_constants::RefSystemConstants, true_system_constants::TrueSystemConstants )
    model = Model(Ipopt.Optimizer)
    @unpack λ   = assumption_constants

    @variables model begin
        rho_a  >= 0.01 
        rho_r  >=  0.01
         ω >=  2*λ
    end

    register(model, :rho_r_condition, 2, rho_r_condition; autodiff = true)
    register(model, :rho_a_condition, 2, rho_a_condition; autodiff = true)
    register(model, :filter_bandwidth_condition1, 2, filter_bandwidth_condition1; autodiff = true)
    register(model, :filter_bandwidth_condition2, 3, filter_bandwidth_condition2; autodiff = true)

    @NLconstraint(model, rho_r_condition(rho_r, ω) >= 0)
    @NLconstraint(model, rho_a_condition(rho_a, ω) >= 0)
    @NLconstraint(model, filter_bandwidth_condition1(rho_r, ω) >= 0.01)
    @NLconstraint(model, filter_bandwidth_condition2(rho_a, rho_r, ω) >= 0.01)

    @objective(model, Min, rho_r + rho_a + ω)

    optimize!(model)

    rho = value(rho_r) + value(rho_a)

    return round(value(rho_r), digits=3), round(value(rho_a), digits=3) , round(value(rho), digits=3), round(value(ω), digits=3)

end

function sampling_period_computation(rho_a, rho_r, ω; Ts_min = 0.0, Ts_max = 1.0, tol::Float64=1e-5, max_iter::Int=10_000)

    itr = 0
    while (Ts_max - Ts_min ) > tol && itr < max_iter
        Ts_mid = (Ts_min + Ts_max) / 2
        
        if sampling_period_condition(rho_a, rho_r, ω, Ts_mid) > 0
            Ts_min = Ts_mid   
        else
            Ts_max = Ts_mid    
        end
        itr += 1
    end
    return Ts_min
end



