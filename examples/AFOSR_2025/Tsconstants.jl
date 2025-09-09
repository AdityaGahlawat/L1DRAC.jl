using UnPack

# γ-functions 
γ₁(Ts::Float64, λₛ::Float64) = sqrt(λₛ * (exp(λₛ*Ts) + 1) / (exp(λₛ*Ts) - 1))
γ₁_prime(ω::Float64, Ts::Float64, λ::Float64) = (1 - exp(-2*λ*Ts)) * (1 - exp(-ω*Ts))
γ₂(ω::Float64, Ts::Float64, λₛ::Float64) = max(exp((ω - λₛ)*Ts), 1.0)* (λₛ*exp(ω*Ts) - 1) / (ω*exp(λₛ*Ts) - 1)

γ₂_prime(ω::Float64, Ts::Float64, λₛ::Float64) = max(abs(1 - exp((ω - λₛ)*Ts) * ((λₛ*exp(ω*Ts) - 1) / (ω*exp(λₛ*Ts) - 1))),
                                                             abs(1 - ((λₛ*exp(ω*Ts) - 1) / (ω*exp(λₛ*Ts) - 1))))
γ_double_prime(p::Int, ω::Float64, Ts::Float64, λₛ::Float64) =  𝔭_prime(p) * γ₂_prime(ω, Ts, λₛ) + 𝔭_double_prime(p) * sqrt(1 - exp(-2*ω*Ts)) * (2 + γ₂(ω, Ts, λₛ))


function Gamma_mu(Ts::Float64, assumption_constants::AssumptionConstants, L1params)
    @unpack L_μ_parallel, L_μ, Δ_Theta = assumption_constants
    @unpack ω, λₛ = L1params
    return L_μ_parallel + L_μ * γ₂(ω, Ts, λₛ) * Δ_Theta
end

function GammaHat_mu(Ts::Float64, assumption_constants::AssumptionConstants, L1params)
    @unpack LHat_μ_parallel, LHat_μ, Δ_Theta = assumption_constants
    @unpack ω, λₛ = L1params
    return LHat_μ_parallel + LHat_μ * γ₂(ω, Ts, λₛ) * Δ_Theta
end

function Gamma_sigma(Ts::Float64, assumption_constants::AssumptionConstants, L1params)
    @unpack L_p_parallel, L_σ_parallel, L_p, L_σ, Δ_Theta = assumption_constants
    @unpack ω, λₛ = L1params
    return L_p_parallel + L_σ_parallel + γ₂(ω, Ts, λₛ) * Δ_Theta * (L_p + L_σ)
end

function GammaHat_sigma(Ts::Float64, assumption_constants::AssumptionConstants, L1params)
    @unpack LHat_p_parallel, L_p, LHat_σ_parallel, LHat_σ, Δ_Theta, λ = assumption_constants
    @unpack ω, λₛ = L1params
    return LHat_p_parallel + LHat_σ_parallel + γ₂(ω, Ts, λₛ) * Δ_Theta * (LHat_p + LHat_σ)
end

function DeltaBar_1(Ts::Float64, assumption_constants::AssumptionConstants, L1params)
    @unpack order_p, Δf, Δμ, Δg, Δ_Theta, Δp, Δσ = assumption_constants
    @unpack λₛ = L1params

    return 2*Ts*(Δf + (1 + Δg*Δ_Theta* exp(-λₛ*Ts))*Δμ +
             𝔭_prime(order_p) * Δg*Δ_Theta* exp(-λₛ*Ts) * γ₁(Ts, λₛ) * (Δp + Δσ)) + 2*sqrt(2)*sqrt(Ts)*𝔭_prime(order_p)*(Δp + Δσ)
end

function DeltaBar_2(Ts::Float64, assumption_constants::AssumptionConstants, L1params)
    @unpack order_p, Δσ, Δg, Δ_Theta = assumption_constants
    @unpack λₛ = L1params
    return 2 * 𝔭_prime(order_p) * Δσ *
           (sqrt(2)*sqrt(Ts) + Ts*Δg*Δ_Theta*exp(-λₛ*Ts)*γ₁(Ts, λₛ))
end

function DeltaBar_3(Ts::Float64, assumption_constants::AssumptionConstants, L1params)
    @unpack Δf, Δμ, Δg, Δ_Theta = assumption_constants
    @unpack λₛ = L1params
    return 2*Ts * (Δf + (1 + Δg*Δ_Theta*exp(-λₛ*Ts))*Δμ)
end

function Delta_mu_1(Ts::Float64, assumption_constants::AssumptionConstants, L1params)
    @unpack Δμ_parallel = assumption_constants
    @unpack λₛ = L1params
    return Ts*GammaHat_mu(Ts, assumption_constants, L1params) 
            + DeltaBar_1(Ts, assumption_constants, L1params)*Gamma_mu(Ts, assumption_constants, L1params) + Δμ_parallel*(1 - exp(-λₛ*Ts))
end

function Delta_mu_2(Ts::Float64, assumption_constants::AssumptionConstants, L1params)
   return DeltaBar_2(Ts, assumption_constants, L1params)* Gamma_mu(Ts, assumption_constants, L1params)
end

function Delta_mu_3(Ts::Float64, assumption_constants::AssumptionConstants, L1params)
    @unpack Δμ_parallel = assumption_constants
    @unpack λₛ = L1params
    Ts*Gamma_mu(Ts, assumption_constants, L1params)*DeltaBar_3(Ts, assumption_constants, L1params) + Δμ_parallel*(1 - exp(-λₛ*Ts))
end

function Delta_sigma_1(Ts::Float64, assumption_constants::AssumptionConstants, L1params)
    @unpack L_p_parallel, L_σ_parallel, LHat_p_parallel, LHat_σ_parallel = assumption_constants
    @unpack ω, λₛ = L1params
    return ((L_p_parallel + L_σ_parallel)*sqrt(DeltaBar_1(Ts, assumption_constants, L1params)) 
              + Ts*(LHat_p_parallel + LHat_σ_parallel))*(1 + γ₂(ω, Ts, λₛ)) + Gamma_sigma(Ts, assumption_constants, L1params)*sqrt(DeltaBar_1(Ts, assumption_constants, L1params)) 
                + Ts*GammaHat_sigma(Ts, assumption_constants, L1params)
end

function Delta_sigma_2(Ts::Float64, assumption_constants::AssumptionConstants, L1params)
    @unpack L_p_parallel, L_σ_parallel = assumption_constants
    @unpack ω, λₛ = L1params
    return sqrt(DeltaBar_2(Ts, assumption_constants, L1params)) * (Gamma_sigma(Ts, assumption_constants, L1params) + (1 + γ₂(ω, Ts, λₛ))*(L_p_parallel + L_σ_parallel))
end

function Delta_sigma_3(Ts::Float64, assumption_constants::AssumptionConstants, L1params)
    @unpack L_p_parallel, L_σ_parallel = assumption_constants
    @unpack ω, λₛ = L1params
    return sqrt(DeltaBar_3(Ts, assumption_constants, L1params)) * (Gamma_sigma(Ts, assumption_constants, L1params) + (1 + γ₂(ω, Ts, λₛ))*(L_p_parallel + L_σ_parallel))
end


function UpsilonPrime_1(ξ::Float64, Ts::Float64, assumption_constants::AssumptionConstants, L1params)
    @unpack order_p = assumption_constants
    @unpack ω = L1params
    return Delta_mu_1(Ts, assumption_constants, L1params) + sqrt(ω)*𝔭_prime(order_p)*Delta_sigma_1(Ts, assumption_constants, L1params)
            +  sqrt(ω)*𝔭_prime( order_p )* Delta_sigma_2(Ts, assumption_constants, L1params)*sqrt(sqrt(ξ)) 
             + (Delta_mu_2(Ts, assumption_constants, L1params) + sqrt(ω)*𝔭_prime(order_p)*Delta_sigma_3(Ts, assumption_constants, L1params))* sqrt(ξ)
               + Delta_mu_3(Ts, assumption_constants, L1params)*ξ
end

function UpsilonPrime_2(ξ::Float64, Ts::Float64, assumption_constants::AssumptionConstants, L1params)
    @unpack order_p, Δp_parallel, Δσ_parallel = assumption_constants
    @unpack ω, λₛ = L1params
    return γ_double_prime(order_p, ω, Ts, λₛ) * (Δp_parallel + Δσ_parallel*(1 + sqrt(ξ)))
end

function UpsilonPrime_3(ξ::Float64, Ts::Float64, assumption_constants::AssumptionConstants, L1params)
    @unpack Δμ_parallel = assumption_constants
    @unpack ω = L1params
    return Δμ_parallel * (exp(ω*Ts) - 1) * (1 + ξ)
end

function UpsilonTildeMinus(ξ::Float64, Ts::Float64, assumption_constants::AssumptionConstants, L1params)
    @unpack order_p, Δg, Δp_parallel, Δσ_parallel, Δμ_parallel, λ = assumption_constants
    @unpack ω = L1params
    return (1/λ) * (1 - exp(-2*λ*Ts)) * Δg * ( sqrt(ω) * 𝔭_prime(order_p) * sqrt(1 - exp(-2*ω*Ts)) * (Δp_parallel + Δσ_parallel*(1 + sqrt(ξ))) + Δμ_parallel * (1 + ξ) )
end

function UpsilonMinus(ξ::Float64, Ts::Float64, assumption_constants::AssumptionConstants, L1params)
    @unpack order_p, Δp_parallel, Δσ_parallel, Δμ_parallel, λ = assumption_constants
    @unpack ω = L1params
    return γ₁_prime(ω, Ts, λ) *( 2*𝔭_prime(order_p) * (Δp_parallel + Δσ_parallel*(1 + sqrt(ξ))) + (Δμ_parallel/sqrt(ω)) * (1 + ξ) )
end

function UpsilonDot(ξ::Float64, assumption_constants::AssumptionConstants, L1params)
    @unpack order_p, Δg, L_p_parallel, L_σ_parallel, L_μ = assumption_constants
    @unpack ω = L1params
    return Δg^2 * ( 𝔭_prime(order_p) * (L_p_parallel + L_σ_parallel) * sqrt(ξ) + (1/sqrt(ω)) * L_μ_parallel * ξ )
end

