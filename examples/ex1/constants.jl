include("constants_types.jl")
# ====================================================================
# Reference system constants 

# Δ̂ᵣ (DeltaRHat) ======================================

function DeltaRHat(c::AssumptionConstants, p::Int, lip::Bool)
    (; Δg, Δf, Δσ, Δp, Δμ, Δ_star, L_f, λ) = c
    Δ̂ᵣ₁ = Δg * ((1 / sqrt(λ)) * (Δf * (2 + Δ_star) * (1 - I_Lip(lip)) + Δμ) +
                 𝔭(p) * (2*Δp + Δσ))
    Δ̂ᵣ₂ = Δg * 𝔭(p) * Δσ
    Δ̂ᵣ₃ = (1 / sqrt(λ)) * Δg * (Δf * (1 - I_Lip(lip)) + Δμ)
    Δ̂ᵣ₄ = (1 / sqrt(λ)) * (Δg * L_f * I_Lip(lip) + Δg)
    return DeltaRHat(Δ̂ᵣ₁, Δ̂ᵣ₂, Δ̂ᵣ₃, Δ̂ᵣ₄)
end

# Δᵣₒ (DeltaR_circle) ======================================

function DeltaR_circle(c::AssumptionConstants, p::Int, dh::DeltaRHat)
    (; Δp, Δσ, Δμ_parallel, Δσ_parallel, Δg, λ, m) = c
    Δ̂ᵣ₁ = dh.Δ̂ᵣ₁
    Δᵣₒ₁ = Δp^2 + (Δp + Δσ)^2
    Δᵣₒ₂ = (Δμ_parallel / sqrt(λ)) * (Δ̂ᵣ₁ + (Δg^2 * Δμ_parallel) / sqrt(λ))
    Δᵣₒ₃ = (𝔭′(p) / sqrt(λ)) * (Δp + Δσ_parallel) *
           (Δ̂ᵣ₁ + (2 * Δg^2 * Δμ_parallel) / sqrt(λ))
    Δᵣₒ₄ = (Δp + Δσ_parallel) * Δg / λ *
           ((𝔭′(p)^2) * Δg * (Δp + Δσ_parallel)) +
           sqrt(m) * (2*Δp + Δσ)
    return DeltaR_circle(Δᵣₒ₁, Δᵣₒ₂, Δᵣₒ₃, Δᵣₒ₄)
end
# Δᵣ⊚ (DeltaR_circledcirc) ======================================

function DeltaR_circledcirc(c::AssumptionConstants, p::Int, dh::DeltaRHat)
    (; Δg, Δσ, Δσ_parallel, Δp, Δμ_parallel, λ, m) = c
    Δ̂ᵣ₁, Δ̂ᵣ₂ = dh.Δ̂ᵣ₁, dh.Δ̂ᵣ₂
    Δr_circledcirc_1 = 2 * Δσ * (Δp + Δσ)
    Δr_circledcirc_2 = (Δμ_parallel / sqrt(λ)) * Δ̂ᵣ₂
    Δr_circledcirc_3 = (𝔭′(p) / sqrt(λ)) *
                       ((Δp + Δσ_parallel) * Δ̂ᵣ₂ +
                        Δσ_parallel * (Δ̂ᵣ₁ + (2 * Δg^2 * Δμ_parallel) / sqrt(λ)))
    Δr_circledcirc_4 = (Δg / λ) *
                       ((Δp + Δσ_parallel) *
                        (2 * (𝔭′(p)^2) * Δg * Δσ_parallel + sqrt(m) * Δσ) +
                        sqrt(m) * Δσ_parallel * (2*Δp + Δσ))
    return DeltaR_circledcirc(Δr_circledcirc_1, Δr_circledcirc_2, Δr_circledcirc_3, Δr_circledcirc_4)
end

# Δᵣ⊙ (DeltaR_odot) ======================================
function DeltaR_odot(c::AssumptionConstants, p::Int, dh::DeltaRHat)
    (; Δg, Δg_perp, Δσ, Δσ_parallel, Δp, Δμ, Δμ_parallel, λ, m) = c
    Δ̂ᵣ₁, Δ̂ᵣ₂, Δ̂ᵣ₃, Δ̂ᵣ₄ = dh.Δ̂ᵣ₁, dh.Δ̂ᵣ₂, dh.Δ̂ᵣ₃, dh.Δ̂ᵣ₄
    Δr_odot_1 = Δσ^2
    Δr_odot_2 = 2 * Δg_perp * Δμ
    Δr_odot_3 = 2 * 𝔭(p) * (Δg_perp * (Δp + Δσ) + Δp)
    Δr_odot_4 = (Δμ_parallel / sqrt(λ)) * (Δ̂ᵣ₁ + Δ̂ᵣ₃ + (2 * Δg^2 * Δμ_parallel) / sqrt(λ))
    Δr_odot_5 = Δμ_parallel * (Δ̂ᵣ₄ / sqrt(λ) + 4 * Δg) +
                2 * sqrt(λ) * Δg * 𝔭(p) * (Δp + Δσ_parallel)
    Δr_odot_6 = (𝔭′(p) / sqrt(λ)) *
                ((Δp + Δσ_parallel) * (Δ̂ᵣ₃ + (2 * Δg^2 * Δμ_parallel) / sqrt(λ))) +
                Δσ_parallel * Δ̂ᵣ₂
    Δr_odot_7 = 𝔭′(p) * (Δp + Δσ_parallel) * (Δ̂ᵣ₄ / sqrt(λ) + 2 * Δg)
    Δr_odot_8 = Δσ_parallel * (Δg / λ) *
                ((𝔭′(p)^2) * Δg * Δσ_parallel + sqrt(m) * Δσ)
    return DeltaR_odot(Δr_odot_1, Δr_odot_2, Δr_odot_3, Δr_odot_4,
                       Δr_odot_5, Δr_odot_6, Δr_odot_7, Δr_odot_8)
end

# Δᵣ⊗ (DeltaR_otimes) ======================================
function DeltaR_otimes(c::AssumptionConstants, p::Int, dh::DeltaRHat)
    (; Δg, Δg_perp, Δσ, Δσ_parallel, Δp, Δμ_parallel, λ) = c
    Δ̂ᵣ₂, Δ̂ᵣ₃, Δ̂ᵣ₄ = dh.Δ̂ᵣ₂, dh.Δ̂ᵣ₃, dh.Δ̂ᵣ₄
    Δr_otimes_1 = 2 * 𝔭(p) * Δg_perp * Δσ
    Δr_otimes_2 = Δμ_parallel * (Δ̂ᵣ₂ / sqrt(λ))
    Δr_otimes_3 = 2 * 𝔭(p) * sqrt(λ) * Δg * Δσ_parallel
    Δr_otimes_4 = 𝔭′(p) * Δσ_parallel *
                  ((Δ̂ᵣ₃ + (2 * Δg^2 * Δμ_parallel) / sqrt(λ)) / sqrt(λ))
    Δr_otimes_5 = 𝔭′(p) * Δσ_parallel * (Δ̂ᵣ₄ / sqrt(λ) + 2 * Δg)
    return DeltaR_otimes(Δr_otimes_1, Δr_otimes_2, Δr_otimes_3, Δr_otimes_4, Δr_otimes_5)
end

# Δᵣ⊛ (DeltaR_ostar) ======================================
function DeltaR_ostar(c::AssumptionConstants, dh::DeltaRHat)
    (; Δg, Δμ_parallel, λ) = c
    Δ̂ᵣ₃, Δ̂ᵣ₄ = dh.Δ̂ᵣ₃, dh.Δ̂ᵣ₄
    Δr_ostar_1 = 2 * Δg * Δμ_parallel
    Δr_ostar_2 = Δμ_parallel * ((Δ̂ᵣ₃ + (Δg^2 * Δμ_parallel) / sqrt(λ)) / sqrt(λ))
    Δr_ostar_3 = Δμ_parallel * (Δ̂ᵣ₄ / sqrt(λ) + 4 * Δg)
    return DeltaR_ostar(Δr_ostar_1, Δr_ostar_2, Δr_ostar_3)
end

# ====================================================================
# True system constants 

# Δ̂ᵣ (DeltaHat) ======================================

function DeltaHat(c::AssumptionConstants, p::Int, lip::Bool)
    (; Δg, Δf, L_p, L_σ, L_μ, L_f, Δġ, λ, m) = c
    Δ̂₁ = (2 / sqrt(λ)) * Δg * Δf * (1 - I_Lip(lip))
    Δ̂₂ = Δg * 𝔭(p) * (L_p + L_σ)
    Δ̂₃ = (1 / sqrt(λ)) * Δg * Δf * (1 - I_Lip(lip))
    Δ̂₄ = (1 / sqrt(λ)) * (Δg * (L_μ + L_f * I_Lip(lip)) + Δġ)
    Δ̂₅ = sqrt(m) * Δg * (L_p + L_σ)
    return DeltaHat(Δ̂₁, Δ̂₂, Δ̂₃, Δ̂₄, Δ̂₅)
end

# Δ⊚ (Delta_circledcirc) ======================================

function Delta_circledcirc(c::AssumptionConstants, p::Int, dh::DeltaHat)
    (; L_p_parallel, L_σ_parallel, λ) = c
    Δcircledcirc_1 = (1 / sqrt(λ)) * 𝔭′(p) * (L_p_parallel + L_σ_parallel) * dh.Δ̂₁
    return Delta_circledcirc(Δcircledcirc_1)
end

# Δ⊙ (Delta_odot) ======================================

function Delta_odot(c::AssumptionConstants, p::Int, dh::DeltaHat)
    (; L_p, L_σ, L_p_parallel, L_σ_parallel, L_μ_parallel, λ, Δg) = c
    Lsum_parallel = (L_p_parallel + L_σ_parallel)
    Δodot_1 = (L_p + L_σ)^2
    Δodot_2 = (1 / sqrt(λ)) * L_μ_parallel * dh.Δ̂₁
    Δodot_3 = (1 / sqrt(λ)) * 𝔭′(p) * Lsum_parallel * dh.Δ̂₂
    Δodot_4 = (1 / λ) * Lsum_parallel * ( sqrt(λ) * dh.Δ̂₄ +
               Δg^2 * (𝔭′(p)^2) * Lsum_parallel )
    return Delta_odot(Δodot_1, Δodot_2, Δodot_3, Δodot_4)
end

# Δ⊗ (Delta_otimes) ======================================

function Delta_otimes(c::AssumptionConstants, p::Int, dh::DeltaHat)
    (; Δg, Δg_perp, L_p_parallel, L_σ_parallel, L_p_perp, L_σ_perp, L_μ_parallel, λ) = c
    Δotimes_1 = 2 * Δg_perp * 𝔭(p) * (L_p_perp + L_σ_perp)
    Δotimes_2 = 2 * sqrt(λ) * Δg * 𝔭(p) * (L_p_parallel + L_σ_parallel) +
                L_μ_parallel * dh.Δ̂₂ / sqrt(λ)
    Δotimes_3 = (1 / sqrt(λ)) * 𝔭′(p) * (L_p_parallel + L_σ_parallel) * dh.Δ̂₃
    Δotimes_4 = 𝔭′(p) * (L_p_parallel + L_σ_parallel) *
                (dh.Δ̂₄ / sqrt(λ) + 2 * Δg * (1 + (Δg/λ) * L_μ_parallel))
    return Delta_otimes(Δotimes_1, Δotimes_2, Δotimes_3, Δotimes_4)
end

# Δ⊛ (Delta_ostar) ======================================

function Delta_ostar(c::AssumptionConstants, dh::DeltaHat)
    (; Δg, L_μ_perp, L_μ_parallel, λ) = c
    Δostar_1 = 2 * Δg * L_μ_perp
    Δostar_2 = (1 / sqrt(λ)) * L_μ_parallel * dh.Δ̂₃
    Δostar_3 = L_μ_parallel * (dh.Δ̂₄ / sqrt(λ) + Δg * (4 + (Δg/λ) * L_μ_parallel))
    return Delta_ostar(Δostar_1, Δostar_2, Δostar_3)
end