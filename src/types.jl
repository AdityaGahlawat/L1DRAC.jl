## Simulation Parameters 
struct SimParams
    tspan::Tuple{Float64, Float64}
    Δₜ::Float64
    Ntraj::Int  
end
sim_params(tspan::Tuple{Float64, Float64}, Δₜ::Float64, Ntraj::Int) = SimParams(tspan, Δₜ, Ntraj)
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