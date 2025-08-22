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