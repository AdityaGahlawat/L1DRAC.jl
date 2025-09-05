using StaticArrays
# ====================================================================
𝔭(p::Int)  = sqrt(p * (p - 1) / 2)
𝔭′(p::Int) = sqrt((2*p - 1) / 2)     
𝔭″(p::Int) = sqrt(p * (4*p - 1))      
I_Lip(f::Bool) = f ? 1 : 0
# ====================================================================
struct AssumptionConstants
    # Δ constants 
    Δg::Float64
    Δg_dot::Float64
    Δg_perp::Float64
    Δf::Float64
    Δ_star::Float64

    Δσ::Float64
    Δσ_parallel::Float64
    Δσ_perp::Float64

    Δp::Float64
    Δp_parallel::Float64
    Δp_perp::Float64

    Δμ::Float64
    Δμ_parallel::Float64
    Δμ_perp::Float64

    # L constants 
    L_p::Float64
    L_σ::Float64
    L_μ::Float64
    L_f::Float64

    L_p_parallel::Float64  
    L_σ_parallel::Float64   
    L_μ_parallel::Float64   

    L_p_perp::Float64      
    L_σ_perp::Float64     
    L_μ_perp::Float64     

    λ::Float64
    m::Float64
end

AssumptionConstants(; 
    Δg=0.0, Δg_dot=0.0, Δg_perp=0.0, Δf=0.0, Δ_star=0.0, Δσ=0.0, Δσ_parallel=0.0,Δσ_perp=0.0,
    Δp=0.0, Δp_parallel=0.0, Δp_perp=0.0, Δμ=0.0, Δμ_parallel=0.0, Δμ_perp=0.0,
    L_p=0.0, L_σ=0.0, L_μ=0.0, L_f=0.0,
    L_p_parallel=0.0, L_σ_parallel=0.0, L_μ_parallel=0.0,
    L_p_perp=0.0, L_σ_perp=0.0, L_μ_perp=0.0,
    λ=1.0, m=1.0
) = AssumptionConstants(Δg, Δg_dot, Δg_perp, Δf, Δ_star, Δσ, Δσ_parallel, Δσ_perp, Δp, Δp_parallel, Δp_perp, Δμ, Δμ_parallel, Δμ_perp, 
                        L_p, L_σ, L_μ, L_f,
                        L_p_parallel, L_σ_parallel, L_μ_parallel,
                        L_p_perp, L_σ_perp, L_μ_perp,
                        λ, m)
# ====================================================================
# Reference system constants 

# Δ̂ᵣ (DeltaRHat) ======================================

function DeltaRHat(assumption_constants::AssumptionConstants, p::Int, lip::Bool)

    (; Δg, Δg_dot, Δf, Δσ, Δp, Δμ, Δ_star, L_f, λ) = assumption_constants
    Δ̂ᵣ₁ = Δg * ((1 / sqrt(λ)) * (Δf * (2 + Δ_star) * (1 - I_Lip(lip)) + Δμ) +
                 𝔭(p) * (2*Δp + Δσ))
    Δ̂ᵣ₂ = Δg * 𝔭(p) * Δσ
    Δ̂ᵣ₃ = (1 / sqrt(λ)) * Δg * (Δf * (1 - I_Lip(lip)) + Δμ)
    Δ̂ᵣ₄ = (1 / sqrt(λ)) * (Δg * L_f * I_Lip(lip) + Δg_dot)
    return SVector{4,Float64}(Δ̂ᵣ₁, Δ̂ᵣ₂, Δ̂ᵣ₃, Δ̂ᵣ₄)
end

# Δᵣₒ (DeltaR_circle) ======================================

function DeltaR_circle(assumption_constants::AssumptionConstants, p::Int, drh::SVector{4,Float64})
    (; Δp, Δσ, Δμ_parallel, Δσ_parallel, Δp_parallel, Δg, λ, m) = assumption_constants
    Δ̂ᵣ₁ = drh[1]
    Δᵣₒ₁ = Δp^2 + (Δp + Δσ)^2
    Δᵣₒ₂ = (Δμ_parallel / sqrt(λ)) * (Δ̂ᵣ₁ + (Δg^2 * Δμ_parallel) / sqrt(λ))
    Δᵣₒ₃ = (𝔭′(p) / sqrt(λ)) * (Δp_parallel + Δσ_parallel) *
           (Δ̂ᵣ₁ + (2 * Δg^2 * Δμ_parallel) / sqrt(λ))
    Δᵣₒ₄ = (Δp_parallel + Δσ_parallel) * Δg / λ *
           ((𝔭′(p)^2) * Δg * (Δp_parallel + Δσ_parallel)) +
           sqrt(m) * (2*Δp + Δσ)
    return SVector{4,Float64}(Δᵣₒ₁, Δᵣₒ₂, Δᵣₒ₃, Δᵣₒ₄)
end
# Δᵣ⊚ (DeltaR_circledcirc) ======================================

function DeltaR_circledcirc(assumption_constants::AssumptionConstants, p::Int, drh::SVector{4,Float64})
    (; Δg, Δσ, Δσ_parallel, Δp, Δμ_parallel, Δp_parallel, λ, m) = assumption_constants
    Δ̂ᵣ₁, Δ̂ᵣ₂ = drh[1], drh[2]
    Δr_circledcirc_1 = 2 * Δσ * (Δp + Δσ)
    Δr_circledcirc_2 = (Δμ_parallel / sqrt(λ)) * Δ̂ᵣ₂
    Δr_circledcirc_3 = (𝔭′(p) / sqrt(λ)) *
                       ((Δp_parallel + Δσ_parallel) * Δ̂ᵣ₂ +
                        Δσ_parallel * (Δ̂ᵣ₁ + (2 * Δg^2 * Δμ_parallel) / sqrt(λ)))
    Δr_circledcirc_4 = (Δg / λ) *
                       ((Δp_parallel + Δσ_parallel) *
                        (2 * (𝔭′(p)^2) * Δg * Δσ_parallel + sqrt(m) * Δσ) +
                        sqrt(m) * Δσ_parallel * (2*Δp + Δσ))
    return SVector{4,Float64}(Δr_circledcirc_1, Δr_circledcirc_2, Δr_circledcirc_3, Δr_circledcirc_4)
end

# Δᵣ⊙ (DeltaR_odot) ======================================
function DeltaR_odot(assumption_constants::AssumptionConstants, p::Int, drh::SVector{4,Float64})
    (; Δg, Δg_perp, Δσ, Δσ_parallel, Δσ_perp, Δp, Δμ, Δμ_parallel, Δμ_perp,Δp_parallel,Δp_perp,  λ, m) = assumption_constants
    Δ̂ᵣ₁, Δ̂ᵣ₂, Δ̂ᵣ₃, Δ̂ᵣ₄ = drh[1], drh[2], drh[3], drh[4]
    Δr_odot_1 = Δσ^2
    Δr_odot_2 = 2 * Δg_perp * Δμ_perp
    Δr_odot_3 = 2 * 𝔭(p) * (Δg_perp * (Δp_perp + Δσ_perp) + Δp)
    Δr_odot_4 = (Δμ_parallel / sqrt(λ)) * (Δ̂ᵣ₁ + Δ̂ᵣ₃ + (2 * Δg^2 * Δμ_parallel) / sqrt(λ))
    Δr_odot_5 = Δμ_parallel * (Δ̂ᵣ₄ / sqrt(λ) + 4 * Δg) +
                2 * sqrt(λ) * Δg * 𝔭(p) * (Δp_parallel + Δσ_parallel)
    Δr_odot_6 = (𝔭′(p) / sqrt(λ)) *
                ((Δp_parallel + Δσ_parallel) * (Δ̂ᵣ₃ + (2 * Δg^2 * Δμ_parallel) / sqrt(λ)) +
                Δσ_parallel * Δ̂ᵣ₂)
    Δr_odot_7 = 𝔭′(p) * (Δp_parallel + Δσ_parallel) * (Δ̂ᵣ₄ / sqrt(λ) + 2 * Δg)
    Δr_odot_8 = Δσ_parallel * (Δg / λ) *
                ((𝔭′(p)^2) * Δg * Δσ_parallel + sqrt(m) * Δσ)
    return SVector{8,Float64}(Δr_odot_1, Δr_odot_2, Δr_odot_3, Δr_odot_4,
                       Δr_odot_5, Δr_odot_6, Δr_odot_7, Δr_odot_8)
end

# Δᵣ⊗ (DeltaR_otimes) ======================================
function DeltaR_otimes(assumption_constants::AssumptionConstants, p::Int, drh::SVector{4,Float64})
    (; Δg, Δg_perp, Δσ, Δσ_parallel,Δσ_perp, Δp, Δμ_parallel, λ) = assumption_constants
    Δ̂ᵣ₂, Δ̂ᵣ₃, Δ̂ᵣ₄ = drh[2], drh[3], drh[4]
    Δr_otimes_1 = 2 * 𝔭(p) * Δg_perp * Δσ_perp
    Δr_otimes_2 = Δμ_parallel * (Δ̂ᵣ₂ / sqrt(λ))
    Δr_otimes_3 = 2 * 𝔭(p) * sqrt(λ) * Δg * Δσ_parallel
    Δr_otimes_4 = 𝔭′(p) * Δσ_parallel *
                  ((Δ̂ᵣ₃ + (2 * Δg^2 * Δμ_parallel) / sqrt(λ)) / sqrt(λ))
    Δr_otimes_5 = 𝔭′(p) * Δσ_parallel * (Δ̂ᵣ₄ / sqrt(λ) + 2 * Δg)
    return SVector{5,Float64}(Δr_otimes_1, Δr_otimes_2, Δr_otimes_3, Δr_otimes_4, Δr_otimes_5)
end

# Δᵣ⊛ (DeltaR_ostar) ======================================
function DeltaR_ostar(assumption_constants::AssumptionConstants, drh::SVector{4,Float64})
    (; Δg,Δg_perp, Δμ_parallel,Δμ_perp, λ) = assumption_constants
    Δ̂ᵣ₃, Δ̂ᵣ₄ = drh[3], drh[4]
    Δr_ostar_1 = 2 * Δg_perp * Δμ_perp
    Δr_ostar_2 = Δμ_parallel * ((Δ̂ᵣ₃ + (Δg^2 * Δμ_parallel) / sqrt(λ)) / sqrt(λ))
    Δr_ostar_3 = Δμ_parallel * (Δ̂ᵣ₄ / sqrt(λ) + 4 * Δg)
    return SVector{3,Float64}(Δr_ostar_1, Δr_ostar_2, Δr_ostar_3)
end

# ====================================================================
# True system constants 

# Δ̂ (DeltaHat) ======================================

function DeltaHat(assumption_constants::AssumptionConstants, p::Int, lip::Bool)
    (; Δg, Δf, L_p, L_σ, L_μ, L_f, Δg_dot, λ, m) = assumption_constants
    
    Δ̂₁ = (2 / sqrt(λ)) * Δg * Δf * (1 - I_Lip(lip))
    Δ̂₂ = Δg * 𝔭(p) * (L_p + L_σ)
    Δ̂₃ = (1 / sqrt(λ)) * Δg * Δf * (1 - I_Lip(lip))
    Δ̂₄ = (1 / sqrt(λ)) * (Δg * (L_μ + L_f * I_Lip(lip)) + Δg_dot)
    Δ̂₅ = sqrt(m) * Δg * (L_p + L_σ)
    return SVector{5,Float64}(Δ̂₁, Δ̂₂, Δ̂₃, Δ̂₄, Δ̂₅)
end

# Δ⊚ (Delta_circledcirc) ======================================

function Delta_circledcirc(assumption_constants::AssumptionConstants, p::Int, dh::SVector{5,Float64})
    (; L_p_parallel, L_σ_parallel, λ) = assumption_constants
    Δ̂₁ = dh[1]
    
    Δcircledcirc_1 = (1 / sqrt(λ)) * 𝔭′(p) * (L_p_parallel + L_σ_parallel) * Δ̂₁
    return SVector{1,Float64}(Δcircledcirc_1)
end

# Δ⊙ (Delta_odot) ======================================

function Delta_odot(assumption_constants::AssumptionConstants, p::Int, dh::SVector{5,Float64})
    (; L_p, L_σ, L_p_parallel, L_σ_parallel, L_μ_parallel, λ, Δg) = assumption_constants
    Δ̂₁, Δ̂₂, Δ̂₅ = dh[1], dh[2], dh[5]
   
    Lsum_parallel = (L_p_parallel + L_σ_parallel)
    Δodot_1 = (L_p + L_σ)^2
    Δodot_2 = (1 / sqrt(λ)) * L_μ_parallel * Δ̂₁
    Δodot_3 = (1 / sqrt(λ)) * 𝔭′(p) * Lsum_parallel * Δ̂₂
    Δodot_4 = (1 / λ) * Lsum_parallel * ( Δ̂₅ +
               Δg^2 * (𝔭′(p)^2) * Lsum_parallel )
    return SVector{4,Float64}(Δodot_1, Δodot_2, Δodot_3, Δodot_4)
end

# Δ⊗ (Delta_otimes) ======================================

function Delta_otimes(assumption_constants::AssumptionConstants, p::Int, dh::SVector{5,Float64})
    (; Δg, Δg_perp, L_p_parallel, L_σ_parallel, L_p_perp, L_σ_perp, L_μ_parallel, λ) = assumption_constants
     Δ̂₂, Δ̂₃, Δ̂₄ = dh[2],dh[3], dh[4]

    Δotimes_1 = 2 * Δg_perp * 𝔭(p) * (L_p_perp + L_σ_perp)
    Δotimes_2 = 2 * sqrt(λ) * Δg * 𝔭(p) * (L_p_parallel + L_σ_parallel) +
                L_μ_parallel * Δ̂₂ / sqrt(λ)
    Δotimes_3 = (1 / sqrt(λ)) * 𝔭′(p) * (L_p_parallel + L_σ_parallel) * Δ̂₃
    Δotimes_4 = 𝔭′(p) * (L_p_parallel + L_σ_parallel) *
                (Δ̂₄ / sqrt(λ) + 2 * Δg * (1 + (Δg/λ) * L_μ_parallel))
    return SVector{4,Float64}(Δotimes_1, Δotimes_2, Δotimes_3, Δotimes_4)
end

# Δ⊛ (Delta_ostar) ======================================

function Delta_ostar(assumption_constants::AssumptionConstants, dh::SVector{5,Float64})
    (; Δg, Δg_perp, L_μ_perp, L_μ_parallel, λ) = assumption_constants
    Δ̂₃, Δ̂₄ = dh[3], dh[4]
    
    Δostar_1 = 2 * Δg_perp * L_μ_perp
    Δostar_2 = (1 / sqrt(λ)) * L_μ_parallel * Δ̂₃
    Δostar_3 = L_μ_parallel * (Δ̂₄ / sqrt(λ) + Δg * (4 + (Δg/λ) * L_μ_parallel))
    return SVector{3,Float64}(Δostar_1, Δostar_2, Δostar_3)
end