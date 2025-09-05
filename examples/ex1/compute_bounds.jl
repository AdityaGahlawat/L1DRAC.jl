using Distributions

include("constants.jl")
include("utils.jl")


nominal_ξ₀ = MvNormal(5.0*ones(2), I(2))
true_ξ₀ = MvNormal(-2.0*ones(2), I(2))

assumption_constants = AssumptionConstants(
    Δg=1.0, 
    Δg_dot=0.0, 
    Δg_perp=1.0,
    Δf=75,
    Δ_star=10,
    Δσ=0.3, 
    Δσ_parallel=0.25, 
    Δσ_perp=0.2, 
    Δp=0.2, 
    Δp_parallel=0.2, 
    Δp_perp =0.2, 
    Δμ=0.1,
    Δμ_parallel=0.08,
    Δμ_perp=0.01,
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

# Change different values
p = 1
lip_holds = true  # Assumption 8 
ρᵣ=12.0 
ρₐ=0.9
ω=50.0
ϵ_a=0.3
ϵ_r=0.2


# println("=======================================================================")
# println("Reference System Analysis constants summary:")
ΔrHat = DeltaRHat(assumption_constants, p, lip_holds)
Δr_circle = DeltaR_circle(assumption_constants, p, ΔrHat)
Δr_circledcirc = DeltaR_circledcirc(assumption_constants, p, ΔrHat)
Δr_odot = DeltaR_odot(assumption_constants, p, ΔrHat)
Δr_otimes = DeltaR_otimes(assumption_constants, p, ΔrHat)
Δr_ostar = DeltaR_ostar(assumption_constants, ΔrHat)

# println("=======================================================================")
# println("True System Analysis Constants Summary:")

ΔHat  = DeltaHat(assumption_constants, p, lip_holds)
Δ_circledcirc = Delta_circledcirc(assumption_constants, p, ΔHat)
Δ_odot = Delta_odot(assumption_constants, p, ΔHat)
Δ_otimes = Delta_otimes(assumption_constants, p, ΔHat)
Δ_ostar = Delta_ostar(assumption_constants, ΔHat)
# println("=======================================================================")

function alpha(nominal_ξ₀::Any, true_ξ₀::Any)
    
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

function Γ_r(ρᵣ::Float64, p::Int, ω::Real, assumption_constants::AssumptionConstants)

    λ  = assumption_constants.λ
    Δ_star = assumption_constants.Δ_star

    c₀    = Δr_circle[1]/(2λ) + (ω * Δr_circle[4]) / abs(2λ - ω)
    c½    = Δr_circledcirc[1]/(2λ) + (ω * Δr_circledcirc[4]) / abs(2λ - ω)
    c₁⁽¹⁾ = Δr_odot[1]/(2λ) + (ω * Δr_odot[8]) / abs(2λ - ω)
    c₁⁽²⁾ = Δr_odot[2]/(2λ) + Δr_odot[3]/(2√λ) + Δr_ostar[1]/(2λ)
    c3½  = Δr_otimes[1]/(2*sqrt(λ))

    return c₀   +
           c½   * (ρᵣ + Δ_star)^(1/2) +
           c₁⁽¹⁾   * (ρᵣ + Δ_star) +
           c₁⁽²⁾ * ρᵣ +
           c3½  * (ρᵣ + Δ_star)^(3/2)
end     

function Γ_a(ρₐ::Float64, p::Int, ω::Real, assumption_constants::AssumptionConstants)

    λ = assumption_constants.λ
    c₀  = Δ_odot[1]/(2*λ) + (ω * Δ_odot[4])/abs(2*λ - ω)
    c½   = Δ_otimes[1]/(2*sqrt(λ))

    return (c₀ + c½ *sqrt(ρₐ))* ρₐ
end

function ρᵣ_condition(ρᵣ::Float64, p::Int, ω::Float64, assumption_constants::AssumptionConstants, ϵ_r::Float64, nominal_ξ₀::Any, true_ξ₀::Any )
    
    λ = assumption_constants.λ
    Δg_perp= assumption_constants.Δg_perp
    Δμ_perp = assumption_constants.Δμ_perp

    lhs = (1 - (Δg_perp * Δμ_perp ) / λ) * (ρᵣ^2)
    rhs =  alpha(nominal_ξ₀ , true_ξ₀)^2 + Γ_r(ρᵣ, p, ω, assumption_constants) + ϵ_r
    return lhs ≥ rhs
end

function ρₐ_condition(ρₐ::Float64, p::Int, ω::Float64, assumption_constants::AssumptionConstants, ϵ_a::Float64 )

    λ = assumption_constants.λ
    Δg_perp= assumption_constants.Δg_perp
    L_μ_perp = assumption_constants.L_μ_perp
    
    lhs = (1 - (Δg_perp * L_μ_perp) / λ) * (ρₐ^2)
    rhs =   Γ_a(ρₐ, p, ω, assumption_constants) + ϵ_a
    return lhs ≥ rhs
end

function check_ρ_conditions(ρᵣ, ρₐ, ω, p, ϵ_a, ϵ_r, assumption_constants, nominal_ξ₀, true_ξ₀)

    r_ok = ρᵣ_condition(ρᵣ, p, ω, assumption_constants, ϵ_r, nominal_ξ₀, true_ξ₀)
    a_ok = ρₐ_condition(ρₐ, p, ω, assumption_constants, ϵ_a)

    if r_ok && a_ok
        println("Both ρᵣ and ρₐ conditions satisfied for (ρᵣ=$ρᵣ, ρₐ=$ρₐ, ω=$ω, p=$p, ϵᵣ=$ϵ_r, ϵₐ=$ϵ_a).")
    else
        println("Condition check failed:")
        if !r_ok println("  - ρᵣ_condition FAILED (ρᵣ=$ρᵣ, p=$p, ω=$ω, ϵᵣ=$ϵ_r).") end
        if !a_ok println("  - ρₐ_condition FAILED (ρₐ=$ρₐ, p=$p, ω=$ω, ϵₐ=$ϵ_a).") end
    end
end


check_ρ_conditions(ρᵣ, ρₐ, ω, p, ϵ_a, ϵ_r, assumption_constants, nominal_ξ₀, true_ξ₀)





