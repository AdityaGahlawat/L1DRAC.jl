using UnPack
# ====================================================================
𝔭(p::Int)  = sqrt(p * (p - 1) / 2)
𝔭_prime(p::Int) = sqrt((2*p - 1) / 2)     
𝔭_double_prime(p::Int) = sqrt(p * (4*p - 1))      
I_Lip(f::Bool) = f ? 1 : 0
# ====================================================================

# ====================================================================
# Reference system constants definitions

function _DeltaRHat(assumption_constants::AssumptionConstants)

    @unpack order_p, Lipschitz_condn_holds,  Δg, Δg_dot, Δf, Δσ, Δp, Δμ, Δ_star, L_f, λ = assumption_constants
    ΔrHat_1 = Δg * ((1 / sqrt(λ)) * (Δf * (2 + Δ_star) * (1 - I_Lip(Lipschitz_condn_holds)) + Δμ) +
                 𝔭(order_p) * (2*Δp + Δσ))
    ΔrHat_2 = Δg * 𝔭(order_p) * Δσ
    ΔrHat_3 = (1 / sqrt(λ)) * Δg * (Δf * (1 - I_Lip(Lipschitz_condn_holds)) + Δμ)
    ΔrHat_4 = (1 / sqrt(λ)) * (Δg * L_f * I_Lip(Lipschitz_condn_holds) + Δg_dot)
    DeltaRHat(ΔrHat_1, ΔrHat_2, ΔrHat_3, ΔrHat_4)
end

function _DeltaR_circle(assumption_constants::AssumptionConstants, ΔrHat::DeltaRHat)
    @unpack order_p,  Δp, Δσ, Δμ_parallel, Δσ_parallel, Δp_parallel, Δg, λ, m = assumption_constants
    @unpack ΔrHat_1 = ΔrHat

    Δr_circle_1 = Δp^2 + (Δp + Δσ)^2
    Δr_circle_2 = (Δμ_parallel / sqrt(λ)) * (ΔrHat_1 + (Δg^2 * Δμ_parallel) / sqrt(λ))
    Δr_circle_3  = (𝔭_prime(order_p) / sqrt(λ)) * (Δp_parallel + Δσ_parallel) * (ΔrHat_1  + (2 * Δg^2 * Δμ_parallel) / sqrt(λ))
    Δr_circle_4 = (Δp_parallel + Δσ_parallel) * Δg / λ *((𝔭_prime(order_p)^2) * Δg * (Δp_parallel + Δσ_parallel)) + sqrt(m) * (2*Δp + Δσ)
    return DeltaR_circle(Δr_circle_1,Δr_circle_2, Δr_circle_3, Δr_circle_4 )
end

function _DeltaR_circledcirc(assumption_constants::AssumptionConstants, ΔrHat::DeltaRHat)
    @unpack order_p, Δg, Δσ, Δσ_parallel, Δp, Δμ_parallel, Δp_parallel, λ, m = assumption_constants
    @unpack ΔrHat_1, ΔrHat_2  = ΔrHat

    Δr_circledcirc_1 = 2 * Δσ * (Δp + Δσ)
    Δr_circledcirc_2 = (Δμ_parallel / sqrt(λ)) * ΔrHat_2
    Δr_circledcirc_3 = (𝔭_prime(order_p) / sqrt(λ)) *
                       ((Δp_parallel + Δσ_parallel) * ΔrHat_2 +
                        Δσ_parallel * (ΔrHat_1+ (2 * Δg^2 * Δμ_parallel) / sqrt(λ)))
    Δr_circledcirc_4 = (Δg / λ) *
                       ((Δp_parallel + Δσ_parallel) *
                        (2 * (𝔭_prime(order_p)^2) * Δg * Δσ_parallel + sqrt(m) * Δσ) +
                        sqrt(m) * Δσ_parallel * (2*Δp + Δσ))
    return DeltaR_circledcirc(Δr_circledcirc_1, Δr_circledcirc_2, Δr_circledcirc_3, Δr_circledcirc_4)
end

function _DeltaR_odot(assumption_constants::AssumptionConstants, ΔrHat::DeltaRHat)
    @unpack order_p,  Δg, Δg_perp, Δσ, Δσ_parallel, Δσ_perp, Δp, Δμ, Δμ_parallel, Δμ_perp, Δp_parallel,Δp_perp,  λ, m = assumption_constants
    @unpack ΔrHat_1, ΔrHat_2, ΔrHat_3, ΔrHat_4   = ΔrHat
    
    Δr_odot_1 = Δσ^2
    Δr_odot_2 = 2 * Δg_perp * Δμ_perp
    Δr_odot_3 = 2 * 𝔭(order_p) * (Δg_perp * (Δp_perp + Δσ_perp) + Δp)
    Δr_odot_4 = (Δμ_parallel / sqrt(λ)) * (ΔrHat_1 + ΔrHat_3 + (2 * Δg^2 * Δμ_parallel) / sqrt(λ))
    Δr_odot_5 = Δμ_parallel * (ΔrHat_4 / sqrt(λ) + 4 * Δg) +
                2 * sqrt(λ) * Δg * 𝔭(order_p) * (Δp_parallel + Δσ_parallel)
    Δr_odot_6 = (𝔭_prime(order_p) / sqrt(λ)) *
                ((Δp_parallel + Δσ_parallel) * (ΔrHat_3+ (2 * Δg^2 * Δμ_parallel) / sqrt(λ)) +
                Δσ_parallel * ΔrHat_2)
    Δr_odot_7 = 𝔭_prime(order_p) * (Δp_parallel + Δσ_parallel) * (ΔrHat_4 / sqrt(λ) + 2 * Δg)
    Δr_odot_8 = Δσ_parallel * (Δg / λ) *
                ((𝔭_prime(order_p)^2) * Δg * Δσ_parallel + sqrt(m) * Δσ)
    return DeltaR_odot(Δr_odot_1, Δr_odot_2, Δr_odot_3, Δr_odot_4,Δr_odot_5, Δr_odot_6, Δr_odot_7, Δr_odot_8)
end

function _DeltaR_otimes(assumption_constants::AssumptionConstants, ΔrHat::DeltaRHat)
    @unpack order_p,  Δg, Δg_perp, Δσ, Δσ_parallel,Δσ_perp, Δp, Δμ_parallel, λ = assumption_constants
    @unpack ΔrHat_2, ΔrHat_3, ΔrHat_4   = ΔrHat
    
    Δr_otimes_1 = 2 * 𝔭(order_p) * Δg_perp * Δσ_perp
    Δr_otimes_2 = Δμ_parallel * (ΔrHat_2 / sqrt(λ))
    Δr_otimes_3 = 2 * 𝔭(order_p) * sqrt(λ) * Δg * Δσ_parallel
    Δr_otimes_4 = 𝔭_prime(order_p) * Δσ_parallel *
                  ((ΔrHat_3 + (2 * Δg^2 * Δμ_parallel) / sqrt(λ)) / sqrt(λ))
    Δr_otimes_5 = 𝔭_prime(order_p) * Δσ_parallel * (ΔrHat_4 / sqrt(λ) + 2 * Δg)
    return DeltaR_otimes(Δr_otimes_1, Δr_otimes_2, Δr_otimes_3, Δr_otimes_4, Δr_otimes_5)
end

function _DeltaR_ostar(assumption_constants::AssumptionConstants, ΔrHat::DeltaRHat)
    @unpack Δg,Δg_perp, Δμ_parallel,Δμ_perp, λ = assumption_constants
    @unpack ΔrHat_3, ΔrHat_4   = ΔrHat

    Δr_ostar_1 = 2 * Δg_perp * Δμ_perp
    Δr_ostar_2 = Δμ_parallel * ((ΔrHat_3 + (Δg^2 * Δμ_parallel) / sqrt(λ)) / sqrt(λ))
    Δr_ostar_3 = Δμ_parallel * (ΔrHat_4  / sqrt(λ) + 4 * Δg)
    return DeltaR_ostar(Δr_ostar_1, Δr_ostar_2, Δr_ostar_3)
end


RefSystemConstants(assump_consts::AssumptionConstants) = begin
    DeltaRHat = _DeltaRHat(assump_consts)
    DeltaR_circle = _DeltaR_circle(assump_consts,  DeltaRHat)
    DeltaR_circledcirc= _DeltaR_circledcirc(assump_consts,  DeltaRHat)
    DeltaR_odot= _DeltaR_odot(assump_consts,  DeltaRHat)
    DeltaR_otimes = _DeltaR_otimes(assump_consts,  DeltaRHat)
    DeltaR_ostar = _DeltaR_ostar(assump_consts,  DeltaRHat)
    RefSystemConstants(DeltaRHat, DeltaR_circle, DeltaR_circledcirc , DeltaR_odot,DeltaR_otimes, DeltaR_ostar)
end


# # ====================================================================
# True system constants definitions


function _DeltaHat(assumption_constants::AssumptionConstants)
    @unpack order_p, Lipschitz_condn_holds, Δg, Δf, L_p, L_σ, L_μ, L_f, Δg_dot, λ, m  = assumption_constants
    
    ΔHat_1= (2 / sqrt(λ)) * Δg * Δf * (1 - I_Lip(Lipschitz_condn_holds))
    ΔHat_2 = Δg * 𝔭(order_p) * (L_p + L_σ)
    ΔHat_3 = (1 / sqrt(λ)) * Δg * Δf * (1 - I_Lip(Lipschitz_condn_holds))
    ΔHat_4 = (1 / sqrt(λ)) * (Δg * (L_μ + L_f * I_Lip(Lipschitz_condn_holds)) + Δg_dot)
    ΔHat_5 = sqrt(m) * Δg * (L_p + L_σ)
    return DeltaHat(ΔHat_1, ΔHat_2, ΔHat_3, ΔHat_4,ΔHat_5 )
end

function _Delta_circledcirc(assumption_constants::AssumptionConstants, ΔHat::DeltaHat)
    @unpack order_p, L_p_parallel, L_σ_parallel, λ  = assumption_constants
    @unpack ΔHat_1   = ΔHat
    Δ_circledcirc_1 = (1 / sqrt(λ)) * 𝔭_prime(order_p) * (L_p_parallel + L_σ_parallel) * ΔHat_1  
    return Delta_circledcirc(Δ_circledcirc_1)
end

function _Delta_odot(assumption_constants::AssumptionConstants, ΔHat::DeltaHat)
    @unpack order_p, L_p, L_σ, L_p_parallel, L_σ_parallel, L_μ_parallel, λ, Δg = assumption_constants
    @unpack ΔHat_1, ΔHat_2, ΔHat_5   = ΔHat
   
    Δ_odot_1 = (L_p + L_σ)^2
    Δ_odot_2 = (1 / sqrt(λ)) * L_μ_parallel * ΔHat_1
    Δ_odot_3 = (1 / sqrt(λ)) * 𝔭_prime(order_p) * (L_p_parallel + L_σ_parallel) * ΔHat_2
    Δ_odot_4 = (1 / λ) * (L_p_parallel + L_σ_parallel) * ( ΔHat_5  +
               Δg^2 * (𝔭_prime(order_p)^2) * (L_p_parallel + L_σ_parallel) )
    return Delta_odot(Δ_odot_1, Δ_odot_2, Δ_odot_3, Δ_odot_4)
end

function _Delta_otimes(assumption_constants::AssumptionConstants, ΔHat::DeltaHat)
    @unpack order_p, Δg, Δg_perp, L_p_parallel, L_σ_parallel, L_p_perp, L_σ_perp, L_μ_parallel, λ = assumption_constants
    @unpack ΔHat_2, ΔHat_3, ΔHat_4   = ΔHat

    Δ_otimes_1 = 2 * Δg_perp * 𝔭(order_p) * (L_p_perp + L_σ_perp)
    Δ_otimes_2 = 2 * sqrt(λ) * Δg * 𝔭(order_p) * (L_p_parallel + L_σ_parallel) +
                L_μ_parallel * ΔHat_2 / sqrt(λ)
    Δ_otimes_3 = (1 / sqrt(λ)) * 𝔭_prime(order_p) * (L_p_parallel + L_σ_parallel) * ΔHat_3
    Δ_otimes_4 = 𝔭_prime(order_p) * (L_p_parallel + L_σ_parallel) *
                (ΔHat_4 / sqrt(λ) + 2 * Δg * (1 + (Δg/λ) * L_μ_parallel))
    return Delta_otimes(Δ_otimes_1, Δ_otimes_2, Δ_otimes_3, Δ_otimes_4)
end

function _Delta_ostar(assumption_constants::AssumptionConstants, ΔHat::DeltaHat)
    @unpack order_p,  Δg, Δg_perp, L_μ_perp, L_μ_parallel, λ = assumption_constants
    @unpack ΔHat_3, ΔHat_4   = ΔHat
    
    Δ_ostar_1 = 2 * Δg_perp * L_μ_perp
    Δ_ostar_2 = (1 / sqrt(λ)) * L_μ_parallel * ΔHat_3
    Δ_ostar_3 = L_μ_parallel * (ΔHat_4 / sqrt(λ) + Δg * (4 + (Δg/λ) * L_μ_parallel))
    return Delta_ostar(Δ_ostar_1, Δ_ostar_2, Δ_ostar_3)
end

TrueSystemConstants(assump_consts::AssumptionConstants) = begin
    DeltaHat = _DeltaHat(assump_consts)
    Delta_circledcirc= _Delta_circledcirc(assump_consts, DeltaHat)
    Delta_odot= _Delta_odot(assump_consts, DeltaHat)
    Delta_otimes = _Delta_otimes(assump_consts, DeltaHat)
    Delta_ostar = _Delta_ostar(assump_consts, DeltaHat)
    TrueSystemConstants(DeltaHat, Delta_circledcirc , Delta_odot, Delta_otimes, Delta_ostar)
end