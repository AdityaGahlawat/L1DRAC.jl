using StatsBase
using OptimalTransport
using Distances
using Tulip
using JLD2


function nom_system_simulation(results_file, simulation_parameters, nominal_system; overwrite::Bool=false)
    if isfile(results_file)
        if overwrite
            rm(results_file; force=true)
            println("Overwriting with new simulation...")
            nom_sol = system_simulation(simulation_parameters, nominal_system)
            @time ens_nom_sol = system_simulation(simulation_parameters, nominal_system; simtype = :ensemble)
            jldsave(results_file; ens_nom_sol=ens_nom_sol)   # key "ens_nom_sol"
            return ens_nom_sol
        else
            println("Loading existing data...")
            return JLD2.load(results_file)["ens_nom_sol"]    # same key
        end
    else
        println("No existing file found. Running simulation...")
        nom_sol = system_simulation(simulation_parameters, nominal_system)
        @time ens_nom_sol = system_simulation(simulation_parameters, nominal_system; simtype = :ensemble)
        jldsave(results_file; ens_nom_sol=ens_nom_sol)
        return ens_nom_sol
    end
end

function Delta_star_computation(ens_nom_sol::EnsembleSolution, order_p::Int ,tspan::Tuple{Float64,Float64}, Δ_saveat:: Float64; dt= 0.1 )

    t_vals = collect(tspan[1]:dt:tspan[2])
    idx_num = Int(dt/Δ_saveat)
    t_idxs = [1; idx_num:idx_num:idx_num*(length(t_vals)-1)]

    L2p_norm(x, p) = (mean(sum(abs.(x).^(2*p); dims=1)))^(1/2*p)
    L2p_norm_val = [begin L2p_norm(ens_nom_sol[:,i,:], order_p) end for i in t_idxs]

    return maximum(L2p_norm_val)
end

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
function empirical_wasserstein2(empirical_samples::EmpiricalSamples, system_dimensions::SysDims; max_centers=1300)
    @unpack μ_samples, ν_samples= empirical_samples
    @unpack n = system_dimensions
    
    grid_min = min.(mapslices(minimum, μ_samples; dims=2), mapslices(minimum, ν_samples; dims=2))[:]
    grid_max = max.(mapslices(maximum, μ_samples; dims=2), mapslices(maximum, ν_samples; dims=2))[:]

    bin_width =0.5
    Hμ=nothing
    Hν=nothing
    while true
        edges = ntuple(d -> grid_min[d]:bin_width:grid_max[d], n)

        # fit n-D histograms 
        Hμ = fit(Histogram, tuple((μ_samples[d, :] for d in 1:n)...) , edges; closed=:left)
        Hν = fit(Histogram, tuple((ν_samples[d, :] for d in 1:n)...) , edges; closed=:left)

        mids = map(midpoints, Hμ.edges)           

        if prod(length.(mids))  <= max_centers
            break
        end
        bin_width += 0.25
    end

    histogram_centers = vec(collect(Iterators.product(midpoints(Hμ.edges[1]),midpoints(Hν.edges[2]))))

    # Defining the Cost Matrix between each pair of discretized state space
    cost_matrix = (pairwise(SqEuclidean(), histogram_centers, histogram_centers))

    # @show size(cost_matrix) 
    # @show bin_width

    μ = normalize(Hμ, mode=:probability)
    ν = normalize(Hν, mode=:probability)

    @time dist_sq = emd2(μ.weights, ν.weights, cost_matrix, Tulip.Optimizer()) 

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


