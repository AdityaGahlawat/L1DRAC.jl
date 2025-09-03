# ====================================================================
𝔭(p::Int)  = sqrt(p * (p - 1) / 2)
𝔭′(p::Int) = sqrt((2*p - 1) / 2)     
𝔭″(p::Int) = sqrt(p * (4*p - 1))      
I_Lip(f::Bool) = f ? 1 : 0
# ====================================================================
struct AssumptionConstants
    # Δ constants 
    Δg::Float64
    Δġ::Float64
    Δg_perp::Float64
    Δf::Float64
    Δσ::Float64
    Δσ_parallel::Float64
    Δp::Float64
    Δμ::Float64
    Δμ_parallel::Float64
    Δ_star::Float64

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
    Δg=0.0, Δġ=0.0, Δg_perp=0.0, Δf=0.0, Δσ=0.0, Δσ_parallel=0.0,
    Δp=0.0, Δμ=0.0, Δμ_parallel=0.0, Δ_star=0.0,
    L_p=0.0, L_σ=0.0, L_μ=0.0, L_f=0.0,
    L_p_parallel=0.0, L_σ_parallel=0.0, L_μ_parallel=0.0,
    L_p_perp=0.0, L_σ_perp=0.0, L_μ_perp=0.0,
    λ=1.0, m=1.0
) = AssumptionConstants(Δg, Δġ, Δg_perp, Δf, Δσ, Δσ_parallel, Δp, Δμ, Δμ_parallel, Δ_star,
                        L_p, L_σ, L_μ, L_f,
                        L_p_parallel, L_σ_parallel, L_μ_parallel,
                        L_p_perp, L_σ_perp, L_μ_perp,
                        λ, m)
# ====================================================================
# Reference system constants structure

# ===============================
# Δ̂ᵣ (DeltaRHat)
# ===============================
struct DeltaRHat
    Δ̂ᵣ₁::Float64
    Δ̂ᵣ₂::Float64
    Δ̂ᵣ₃::Float64
    Δ̂ᵣ₄::Float64
end

# ===============================
# Δᵣₒ (DeltaR_circle)
# ===============================
struct DeltaR_circle
    Δᵣₒ₁::Float64
    Δᵣₒ₂::Float64
    Δᵣₒ₃::Float64
    Δᵣₒ₄::Float64
end

# ===============================
# Δᵣ⊚ (DeltaR_circledcirc)
# ===============================
struct DeltaR_circledcirc
    Δr_circledcirc_1::Float64
    Δr_circledcirc_2::Float64
    Δr_circledcirc_3::Float64
    Δr_circledcirc_4::Float64
end

# ===============================
# Δᵣ⊙ (DeltaR_odot)
# ===============================
struct DeltaR_odot
    Δr_odot_1::Float64
    Δr_odot_2::Float64
    Δr_odot_3::Float64
    Δr_odot_4::Float64
    Δr_odot_5::Float64
    Δr_odot_6::Float64
    Δr_odot_7::Float64
    Δr_odot_8::Float64
end

# ===============================
# Δᵣ⊗ (DeltaR_otimes)
# ===============================
struct DeltaR_otimes
    Δr_otimes_1::Float64
    Δr_otimes_2::Float64
    Δr_otimes_3::Float64
    Δr_otimes_4::Float64
    Δr_otimes_5::Float64
end

# ===============================
# Δᵣ⊛ (DeltaR_ostar)
# ===============================
struct DeltaR_ostar
    Δr_ostar_1::Float64
    Δr_ostar_2::Float64
    Δr_ostar_3::Float64
end
# ====================================================================
# True system constants structure

# ===============================
# Δ̂ (DeltaHat)
# ===============================
struct DeltaHat
    Δ̂₁::Float64
    Δ̂₂::Float64
    Δ̂₃::Float64
    Δ̂₄::Float64
    Δ̂₅::Float64
end

# ===============================
# Δ⊚ (Delta_circledcirc)
# ===============================
struct Delta_circledcirc
    Δcircledcirc_1::Float64
end

# ===============================
# Δ⊙ (Delta_odot)
# ===============================
struct Delta_odot
    Δodot_1::Float64
    Δodot_2::Float64
    Δodot_3::Float64
    Δodot_4::Float64
end

# ===============================
# Δ⊗ (Delta_otimes)
# ===============================
struct Delta_otimes
    Δotimes_1::Float64
    Δotimes_2::Float64
    Δotimes_3::Float64
    Δotimes_4::Float64
end

# ===============================
# Δ⊛ (Delta_ostar)
# ===============================
struct Delta_ostar
    Δostar_1::Float64
    Δostar_2::Float64
    Δostar_3::Float64
end

