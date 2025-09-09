
using LinearAlgebra
using Distributions
using StatsBase
using OptimalTransport
using Distances
using Tulip

# closed-form Wasserstein distance of order 2 between Gaussians
function gaussian_wasserstein2(μ1::AbstractVector, Σ1::AbstractMatrix,
                      μ2::AbstractVector, Σ2::AbstractMatrix)
    dμ = μ1 .- μ2
    term1 = dot(dμ, dμ)
    inner_term = sqrt(Symmetric(Σ2)) * Σ1 * sqrt(Symmetric(Σ2))
    term2 = tr(Σ1) + tr(Σ2) - 2 * tr(sqrt(Symmetric(inner_term)))
    return sqrt(term1 + term2)
end

sample_gaussian(μ, Σ, N) = rand(MvNormal(μ, Symmetric(Σ)), N)

# Wasserstein distance of order 2 between empirical distributions 
function empirical_wasserstein2(μ_samples, ν_samples, bin_width)
    
    @assert size(μ_samples, 1) == 2 "μ_samples must be 2×N; got size $(size(μ_samples))"
    @assert size(ν_samples, 1) == 2 "ν_samples must be 2×M; got size $(size(ν_samples))"
    
    grid_min = floor.(min.(minimum(μ_samples, dims = 2), minimum(ν_samples, dims =2)))
    grid_max = ceil.(max.(maximum(μ_samples, dims =2), maximum(ν_samples, dims =2)))
    
    bins_x1 = grid_min[1]:bin_width:grid_max[1]
    bins_x2 = grid_min[2]:bin_width:grid_max[2]

    # Fitting Data and calculating Histogram Distribution 
    Hμ = fit(Histogram, (μ_samples[1,:], μ_samples[2,:]), (bins_x1,bins_x2), closed = :left);
    Hν = fit(Histogram, (ν_samples[1,:], ν_samples[2,:]), (bins_x1,bins_x2), closed = :left);

    histogram_centers = vec(collect(Iterators.product(midpoints(Hμ.edges[1]),midpoints(Hν.edges[2]))))

    # Defining the Cost Matrix between each pair of discretized state space
    cost_matrix = (pairwise(SqEuclidean(), histogram_centers, histogram_centers))

    @show size(cost_matrix)

    μ = normalize(Hμ, mode=:probability)
    ν = normalize(Hν, mode=:probability)

    dist_sq = emd2(μ.weights, ν.weights, cost_matrix, Tulip.Optimizer()) 

    return sqrt(dist_sq)

end

# Tests the difference between the closed form solution of Gaussian to the one computated from their empirical distributions
function test_wasserstein_computation()
    
    μ1 = [0.0, 0.0]
    Σ1 = [1.0 0.; 0. 1.]
    μ2 = [-1, -1]
    Σ2 = [1. 0.0; 0.0 1.]

    X = sample_gaussian(μ1, Σ1, 10000)  
    Y = sample_gaussian(μ2, Σ2, 10000) 

    w2_actual   = gaussian_wasserstein2(μ1, Σ1, μ2, Σ2)
    w2_empirical= empirical_wasserstein2(X,Y, 0.2 )

     @show w2_actual
     @show w2_empirical
end


#test_wasserstein_computation()

# struct L1DRACParams
#     ω::Float64
#     Tₛ::Float64
#     λₛ::Float64
# end

# drac_params(ω::Float64, Tₛ::Float64, λₛ::Float64) = L1DRACParams(ω, Tₛ, λₛ)

drac_params_builder = (ω, λₛ) -> (Tₛ -> drac_params(ω, Tₛ, λₛ))
#  ω_val = 1.0
#  λₛ_val = 2.2

#  partial_drac = drac_params_builder(ω_val, λₛ_val)
#  L1params= L1Params(0.001)