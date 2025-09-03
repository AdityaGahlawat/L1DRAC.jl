include("constants.jl")

c = AssumptionConstants(
    Δg=1.0, 
    Δġ=0.0, 
    Δg_perp=1.0,
    Δf=75,
    Δσ=0.3, 
    Δσ_parallel=0.25, 
    Δp=0.2, 
    Δμ=0.1,
    Δμ_parallel=0.08,
    Δ_star=0.05,
    L_p=0.3, 
    L_σ=0.25,
    L_μ=0.2, 
    L_f=2.0,
    L_p_parallel=0.22,
    L_σ_parallel=0.18,
    L_μ_parallel=0.4,
    L_p_perp=0.12, 
    L_σ_perp=0.11,
    L_μ_perp=0.09,
    λ=3.0, m=1.0
)

p = 2
lip_holds = true
println("Reference System Analysis constants summary:")
ΔrHat = DeltaRHat(c, p, lip_holds)
println("Δ̂ᵣ₁ = ", ΔrHat.Δ̂ᵣ₁)
println("Δ̂ᵣ₂ = ", ΔrHat.Δ̂ᵣ₂)
println("Δ̂ᵣ₃ = ", ΔrHat.Δ̂ᵣ₃)
println("Δ̂ᵣ₄ = ", ΔrHat.Δ̂ᵣ₄)

Δr_circle = DeltaR_circle(c, p, ΔrHat)
println("Δᵣ∘₁ = ", Δr_circle.Δᵣₒ₁)
println("Δᵣ∘₂ = ", Δr_circle.Δᵣₒ₂)
println("Δᵣ∘₃ = ", Δr_circle.Δᵣₒ₃)
println("Δᵣ∘₄ = ", Δr_circle.Δᵣₒ₄)

Δr_circledcirc = DeltaR_circledcirc(c, p, ΔrHat)
println("Δᵣ⊚₁ = ", Δr_circledcirc.Δr_circledcirc_1)
println("Δᵣ⊚₂ = ", Δr_circledcirc.Δr_circledcirc_2)
println("Δᵣ⊚₃ = ", Δr_circledcirc.Δr_circledcirc_3)
println("Δᵣ⊚₄ = ", Δr_circledcirc.Δr_circledcirc_4)

Δr_odot = DeltaR_odot(c, p, ΔrHat)
println("Δᵣ⊙₁ = ", Δr_odot.Δr_odot_1)
println("Δᵣ⊙₂ = ", Δr_odot.Δr_odot_2)
println("Δᵣ⊙₃ = ", Δr_odot.Δr_odot_3)
println("Δᵣ⊙₄ = ", Δr_odot.Δr_odot_4)
println("Δᵣ⊙₅ = ", Δr_odot.Δr_odot_5)
println("Δᵣ⊙₆ = ", Δr_odot.Δr_odot_6)
println("Δᵣ⊙₇ = ", Δr_odot.Δr_odot_7)
println("Δᵣ⊙₈ = ", Δr_odot.Δr_odot_8)

Δr_otimes = DeltaR_otimes(c, p, ΔrHat)
println("Δᵣ⊗₁ = ", Δr_otimes.Δr_otimes_1)
println("Δᵣ⊗₂ = ", Δr_otimes.Δr_otimes_2)
println("Δᵣ⊗₃ = ", Δr_otimes.Δr_otimes_3)
println("Δᵣ⊗₄ = ", Δr_otimes.Δr_otimes_4)
println("Δᵣ⊗₅ = ", Δr_otimes.Δr_otimes_5)

Δr_ostar = DeltaR_ostar(c, ΔrHat)
println("Δᵣ⊛₁ = ", Δr_ostar.Δr_ostar_1)
println("Δᵣ⊛₂ = ", Δr_ostar.Δr_ostar_2)
println("Δᵣ⊛₃ = ", Δr_ostar.Δr_ostar_3)


println("=======================================================================")
println("True System Analysis Constants Summary:")


ΔHat  = DeltaHat(c, p, lip_holds)
println("Δ̂₁ = ", ΔHat.Δ̂₁)
println("Δ̂₂ = ", ΔHat.Δ̂₂)
println("Δ̂₃ = ", ΔHat.Δ̂₃)
println("Δ̂₄ = ", ΔHat.Δ̂₄)
println("Δ̂₅ = ", ΔHat.Δ̂₅)

Δ_circledcirc = Delta_circledcirc(c, p, ΔHat)
println("Δ⊚₁ = ", Δ_circledcirc.Δcircledcirc_1)

Δ_odot = Delta_odot(c, p, ΔHat)
println("Δ⊙₁ = ", Δ_odot.Δodot_1)
println("Δ⊙₂ = ", Δ_odot.Δodot_2)
println("Δ⊙₃ = ", Δ_odot.Δodot_3)
println("Δ⊙₄ = ", Δ_odot.Δodot_4)

Δ_otimes = Delta_otimes(c, p, ΔHat)
println("Δ⊗₁ = ", Δ_otimes.Δotimes_1)
println("Δ⊗₂ = ", Δ_otimes.Δotimes_2)
println("Δ⊗₃ = ", Δ_otimes.Δotimes_3)
println("Δ⊗₄ = ", Δ_otimes.Δotimes_4)

Δ_ostar = Delta_ostar(c, ΔHat)
println("Δ⊛₁ = ", Δ_ostar.Δostar_1)
println("Δ⊛₂ = ", Δ_ostar.Δostar_2)
println("Δ⊛₃ = ", Δ_ostar.Δostar_3)

