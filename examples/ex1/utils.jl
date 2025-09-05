using LinearAlgebra

function w2_gaussians(μ1::AbstractVector, Σ1::AbstractMatrix,
                      μ2::AbstractVector, Σ2::AbstractMatrix)
    dμ = μ1 .- μ2
    term1 = dot(dμ, dμ)
    S2h   = sqrt(Symmetric(Σ2))                  
    inner = S2h * Σ1 * S2h
    term2 = tr(Σ1) + tr(Σ2) - 2tr(sqrt(Symmetric(inner)))
    return sqrt(term1 + term2)
end


