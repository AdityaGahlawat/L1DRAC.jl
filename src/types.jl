## Simulation Parameters 
struct SimParams
    tspan::Tuple{Float64, Float64}
    Δₜ::Float64
    Ntraj::Int 
    Δ_saveat::Float64 
end
sim_params(tspan::Tuple{Float64, Float64}, Δₜ::Float64, Ntraj::Int, Δ_saveat::Float64) = SimParams(tspan, Δₜ, Ntraj, Δ_saveat)
## System Constants 
# struct SysConstants
# end
## System Dimensions
struct SysDims
    n::Int
    m::Int
    d::Int
end
sys_dims(n::Int, m::Int, d::Int) = SysDims(n, m, d)
## Nominal Vector Fields
struct NominalVectorFields
    f::Function
    g::Function
    g_perp::Function
    p::Function
end
nominal_vector_fields(f::Function, g::Function, g_perp::Function, p::Function) = NominalVectorFields(f, g, g_perp, p)
## Uncertain Vector Fields 
struct UncertainVectorFields 
    Λμ::Function  
    Λσ::Function
end
uncertain_vector_fields(Λμ::Function, Λσ::Function) = UncertainVectorFields(Λμ, Λσ)
## Initial distributions 
struct InitialDistributions
    nominal_ξ₀::Any
    true_ξ₀::Any
end
init_dist(nominal_ξ₀::Any, true_ξ₀::Any) = InitialDistributions(nominal_ξ₀, true_ξ₀)
# Nominal System
struct NominalSystem{SysDims, NominalVectorFields, InitialDistributions}
    sys_dims::SysDims
    nom_vec_fields::NominalVectorFields
    init_dists::InitialDistributions
end
nom_sys(sys_dims::SysDims, nom_vec_fields::NominalVectorFields, init_dists::InitialDistributions) = NominalSystem(sys_dims, nom_vec_fields, init_dists)
# True System 
struct TrueSystem{SysDims, NominalVectorFields, UncertainVectorFields, InitialDistributions}
    sys_dims::SysDims
    nom_vec_fields::NominalVectorFields
    unc_vec_fields::UncertainVectorFields
    init_dists::InitialDistributions
end
true_sys(sys_dims::SysDims, nom_vec_fields::NominalVectorFields, unc_vector_fields::UncertainVectorFields, init_dists::InitialDistributions) = TrueSystem(sys_dims, nom_vec_fields, unc_vector_fields, init_dists)

struct L1DRACParams
    ω::Float64
    Tₛ::Float64
    λₛ::Float64
end
drac_params(ω::Float64, Tₛ::Float64, λₛ::Float64) = L1DRACParams(ω, Tₛ, λₛ)


###################################################################
# Constant Types
###################################################################

struct AssumptionConstants
    #= Assumption 1: Nominal (Known) System =#
    # Known drift f bounds
    Δf::Float64                     # ‖f(t,a)‖² ≤ Δf²(1 + ‖a‖²)
    # Known diffusion p bounds
    Δp::Float64                     # ‖p(t,a)‖_F ≤ Δp
    Lhat_p::Float64                 # temporal Lipschitz for p
    L_p::Float64                    # spatial Hölder for p
    # Input operator g bounds
    Δg::Float64                     # ‖g(t)‖_F ≤ Δg
    Δg_dot::Float64                 # ‖ġ(t)‖_F ≤ Δg_dot
    Δg_perp::Float64                # ‖g⊥(t)‖_F ≤ Δg_perp
    Δ_Θ::Float64                    # ‖Θ_ad(t)‖_F ≤ Δ_Θ

    #= Assumption 2: Nominal System Stability =#
    λ::Float64                      # contraction rate: μ(∇_x F̄_μ) ≤ -λ

    #= Assumption 3: Bounded Moments =#
    order_p::Int                    # moment order p* ∈ ℕ≥1
    Δ_star::Float64                 # ‖X*(t)‖_{L_{2p*}} ≤ Δ*

    #= Assumption 4: True (Uncertain) System =#
    # Growth bounds
    Δμ::Float64                     # ‖Λ_μ(t,a)‖² ≤ Δμ²(1 + ‖a‖²)
    Δσ::Float64                     # ‖Λ_σ(t,a)‖_F² ≤ Δσ²(1 + ‖a‖²)^½
    # Decomposed uncertainty bounds
    Δμ_parallel::Float64            # matched drift uncertainty
    Δμ_perp::Float64                # unmatched drift uncertainty
    Δσ_parallel::Float64            # matched diffusion uncertainty
    Δσ_perp::Float64                # unmatched diffusion uncertainty

    #= Assumption 5: Decomposed Known Diffusion =#
    Δp_parallel::Float64            # ‖p∥(t,a)‖_F ≤ Δp∥
    Δp_perp::Float64                # ‖p⊥(t,a)‖_F ≤ Δp⊥
    Lhat_p_parallel::Float64        # temporal Lipschitz for p∥
    L_p_parallel::Float64           # spatial Hölder for p∥
    Lhat_p_perp::Float64            # temporal Lipschitz for p⊥
    L_p_perp::Float64               # spatial Hölder for p⊥

    #= Assumption 6: Lipschitz Continuity =#
    # Drift uncertainty Λ_μ
    L_μ::Float64                    # spatial Lipschitz for Λ_μ
    Lhat_μ::Float64                 # temporal Lipschitz for Λ_μ
    L_μ_parallel::Float64           # spatial Lipschitz for Λ_μ∥
    Lhat_μ_parallel::Float64        # temporal Lipschitz for Λ_μ∥
    L_μ_perp::Float64               # spatial Lipschitz for Λ_μ⊥
    Lhat_μ_perp::Float64            # temporal Lipschitz for Λ_μ⊥
    # Diffusion uncertainty Λ_σ
    L_σ::Float64                    # spatial Hölder for Λ_σ
    Lhat_σ::Float64                 # temporal Lipschitz for Λ_σ
    L_σ_parallel::Float64           # spatial Hölder for Λ_σ∥
    Lhat_σ_parallel::Float64        # temporal Lipschitz for Λ_σ∥
    L_σ_perp::Float64               # spatial Hölder for Λ_σ⊥
    Lhat_σ_perp::Float64            # temporal Lipschitz for Λ_σ⊥

    #= Assumption 7: Internal Stability - validated in validate() =#
    # No new constants - condition: λ > Δg_perp · max{Δμ_perp, L_μ_perp}

    #= Assumption 8: Stronger Regularity (optional) =#
    L_f::Float64                    # spatial Lipschitz for f
    Lhat_f::Float64                 # temporal Lipschitz for f

    # Safety: track which fields were explicitly set
    _set_fields::Set{Symbol}
end

function assumption_constants(; kwargs...)
    provided = Set(keys(kwargs))
    AssumptionConstants(
        # Assumption 1: Nominal System
        get(kwargs, :Δf, 0.0),
        get(kwargs, :Δp, 0.0),
        get(kwargs, :Lhat_p, 0.0),
        get(kwargs, :L_p, 0.0),
        get(kwargs, :Δg, 0.0),
        get(kwargs, :Δg_dot, 0.0),
        get(kwargs, :Δg_perp, 0.0),
        get(kwargs, :Δ_Θ, 0.0),
        # Assumption 2: Stability
        get(kwargs, :λ, 0.0),
        # Assumption 3: Bounded Moments
        get(kwargs, :order_p, 1),
        get(kwargs, :Δ_star, 0.0),
        # Assumption 4: True System
        get(kwargs, :Δμ, 0.0),
        get(kwargs, :Δσ, 0.0),
        get(kwargs, :Δμ_parallel, 0.0),
        get(kwargs, :Δμ_perp, 0.0),
        get(kwargs, :Δσ_parallel, 0.0),
        get(kwargs, :Δσ_perp, 0.0),
        # Assumption 5: Decomposed Diffusion
        get(kwargs, :Δp_parallel, 0.0),
        get(kwargs, :Δp_perp, 0.0),
        get(kwargs, :Lhat_p_parallel, 0.0),
        get(kwargs, :L_p_parallel, 0.0),
        get(kwargs, :Lhat_p_perp, 0.0),
        get(kwargs, :L_p_perp, 0.0),
        # Assumption 6: Lipschitz - drift uncertainty
        get(kwargs, :L_μ, 0.0),
        get(kwargs, :Lhat_μ, 0.0),
        get(kwargs, :L_μ_parallel, 0.0),
        get(kwargs, :Lhat_μ_parallel, 0.0),
        get(kwargs, :L_μ_perp, 0.0),
        get(kwargs, :Lhat_μ_perp, 0.0),
        # Assumption 6: Lipschitz - diffusion uncertainty
        get(kwargs, :L_σ, 0.0),
        get(kwargs, :Lhat_σ, 0.0),
        get(kwargs, :L_σ_parallel, 0.0),
        get(kwargs, :Lhat_σ_parallel, 0.0),
        get(kwargs, :L_σ_perp, 0.0),
        get(kwargs, :Lhat_σ_perp, 0.0),
        # Assumption 8: Stronger Regularity
        get(kwargs, :L_f, 0.0),
        get(kwargs, :Lhat_f, 0.0),
        provided
    )
end

# Mandatory constants that must always be set
const MANDATORY_CONSTANTS = [
    :Δf, :Δp, :Lhat_p, :L_p,           # Assumption 1: drift and diffusion
    :Δg, :Δg_dot, :Δg_perp, :Δ_Θ,      # Assumption 1: input operator
    :λ,                                 # Assumption 2: stability
    :order_p, :Δ_star                   # Assumption 3: bounded moments
]

# Contextual constants required when using TrueSystem (uncertainties)
const CONTEXTUAL_CONSTANTS_TRUE_SYSTEM = [
    :Δμ, :Δσ, :Δμ_parallel, :Δμ_perp, :Δσ_parallel, :Δσ_perp,  # Assumption 4
    :Δp_parallel, :Δp_perp, :Lhat_p_parallel, :L_p_parallel,    # Assumption 5
    :Lhat_p_perp, :L_p_perp,                                     # Assumption 5
    :L_μ, :Lhat_μ, :L_μ_parallel, :Lhat_μ_parallel,             # Assumption 6
    :L_μ_perp, :Lhat_μ_perp,                                     # Assumption 6
    :L_σ, :Lhat_σ, :L_σ_parallel, :Lhat_σ_parallel,             # Assumption 6
    :L_σ_perp, :Lhat_σ_perp                                      # Assumption 6
]

function validate(constants::AssumptionConstants)
    # Check mandatory constants are set
    for field in MANDATORY_CONSTANTS
        if field ∉ constants._set_fields
            error("Mandatory constant $field was not set")
        end
    end
    # Check order_p ≥ 1
    if constants.order_p < 1
        error("order_p must be ≥ 1, got $(constants.order_p)")
    end
    return true
end

function validate(constants::AssumptionConstants, sys::TrueSystem)
    # First check mandatory
    validate(constants)

    # Check contextual constants for true system
    for field in CONTEXTUAL_CONSTANTS_TRUE_SYSTEM
        if field ∉ constants._set_fields
            error("Constant $field is required when using TrueSystem but was not set")
        end
    end

    # Assumption 7: Internal stability condition
    lower_bound = constants.Δg_perp * max(constants.Δμ_perp, constants.L_μ_perp)
    if constants.λ <= lower_bound
        error("Assumption 7 violated: λ ($(constants.λ)) must be > Δg_perp · max{Δμ_perp, L_μ_perp} = $lower_bound")
    end

    return true
end

###################################################################
# Backend Types
###################################################################

struct CPU end
struct GPU end