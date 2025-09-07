# constants defined in the Assumptions (section 2.2)
struct AssumptionConstants
    p::Int 
    Lipschitz_condn_holds::Bool
    Δg::Float64
    Δg_dot::Float64
    Δg_perp::Float64
    Δf::Float64
    Δ_star::Float64

    Δp::Float64
    Δp_parallel::Float64
    Δp_perp::Float64

    Δμ::Float64
    Δμ_parallel::Float64
    Δμ_perp::Float64

    Δσ::Float64
    Δσ_parallel::Float64
    Δσ_perp::Float64

    L_p::Float64
    L_p_parallel::Float64  
    L_p_perp::Float64

    L_μ::Float64
    L_μ_parallel::Float64   
    L_μ_perp::Float64  

    L_σ::Float64
    L_σ_parallel::Float64 
    L_σ_perp::Float64     
   
    L_f::Float64

    λ::Float64
    m::Float64
    ϵ_r::Float64
    ϵ_a::Float64
end

assump_consts(; 
    p=1,
    Lipschitz_condn_holds=true,
    Δg=0.0, Δg_dot=0.0, Δg_perp=0.0, Δf=0.0, Δ_star=0.0,
    Δp=0.0, Δp_parallel=0.0, Δp_perp=0.0,
    Δμ=0.0, Δμ_parallel=0.0, Δμ_perp=0.0,
    Δσ=0.0, Δσ_parallel=0.0, Δσ_perp=0.0,
    L_p=0.0, L_p_parallel=0.0, L_p_perp=0.0,
    L_μ=0.0, L_μ_parallel=0.0, L_μ_perp=0.0,
    L_σ=0.0, L_σ_parallel=0.0, L_σ_perp=0.0,
    L_f=0.0,
    λ=1.0, m=1.0, ϵ_r=0.2 ,ϵ_a=0.2
) = AssumptionConstants(
    p, Lipschitz_condn_holds,
    Δg, Δg_dot, Δg_perp, Δf, Δ_star,
    Δp, Δp_parallel, Δp_perp,
    Δμ, Δμ_parallel, Δμ_perp,
    Δσ, Δσ_parallel, Δσ_perp,
    L_p, L_p_parallel, L_p_perp,
    L_μ, L_μ_parallel, L_μ_perp,
    L_σ, L_σ_parallel, L_σ_perp,
    L_f, λ, m, ϵ_r, ϵ_a
)

# Reference system constants 
# Common constants 
struct DeltaRHat
    ΔrHat_1::Float64
    ΔrHat_2::Float64
    ΔrHat_3::Float64
    ΔrHat_4::Float64
end

# Order 0
struct DeltaR_circle
    Δr_circle_1::Float64
    Δr_circle_2::Float64
    Δr_circle_3::Float64
    Δr_circle_4::Float64
end

# Order 1/2
struct DeltaR_circledcirc
    Δr_circledcirc_1::Float64
    Δr_circledcirc_2::Float64
    Δr_circledcirc_3::Float64
    Δr_circledcirc_4::Float64
end

# Order 1
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

# Order 3/2
struct DeltaR_otimes
    Δr_otimes_1::Float64
    Δr_otimes_2::Float64
    Δr_otimes_3::Float64
    Δr_otimes_4::Float64
    Δr_otimes_5::Float64
end

# Order 2
struct DeltaR_ostar
    Δr_ostar_1::Float64
    Δr_ostar_2::Float64
    Δr_ostar_3::Float64
end

struct RefSystemConstants
    ΔrHat::DeltaRHat
    Δr_circle::DeltaR_circle
    Δr_circledcirc::DeltaR_circledcirc
    Δr_odot::DeltaR_odot
    Δr_otimes::DeltaR_otimes
    Δr_ostar::DeltaR_ostar
end
##########################################################

# True system constants 
# Common constants 
struct DeltaHat
    ΔHat_1::Float64
    ΔHat_2::Float64
    ΔHat_3::Float64
    ΔHat_4::Float64
    ΔHat_5::Float64
end

# Order 1/2
struct Delta_circledcirc
    Δ_circledcirc_1::Float64
end

# Order 1
struct Delta_odot
    Δ_odot_1::Float64
    Δ_odot_2::Float64
    Δ_odot_3::Float64
    Δ_odot_4::Float64
end

# Order 3/2
struct Delta_otimes
    Δ_otimes_1::Float64
    Δ_otimes_2::Float64
    Δ_otimes_3::Float64
    Δ_otimes_4::Float64
end

# Order 2
struct Delta_ostar
    Δ_ostar_1::Float64
    Δ_ostar_2::Float64
    Δ_ostar_3::Float64
end

struct TrueSystemConstants
    ΔHat::DeltaHat
    Δ_circledcirc::Delta_circledcirc
    Δ_odot::Delta_odot
    Δ_otimes::Delta_otimes
    Δ_ostar::Delta_ostar
end


