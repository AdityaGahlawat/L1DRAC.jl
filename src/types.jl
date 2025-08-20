## Nominal Vector Fields
struct NominalVectorFields{fF, gF, pF} 
    f::fF
    g::gF
    p::pF
end
nominal_vector_fields(f::Function, g::Function, p::Function) = NominalVectorFields(f, g, p)

## Uncertain Vector Fields 


