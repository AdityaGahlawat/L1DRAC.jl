# Setup for Double Integrator 1D benchmark

# using L1DRAC
# using LinearAlgebra
# using Distributions
# using ControlSystemsBase
# using Dates
# using DataFrames
# using CSV
# using CUDA
# using Distributed
# using StaticArrays
# using UnPack
# using StochasticDiffEq



function setup_double_integrator(; Ntraj = 10)
    # Simulation Parameters
    tspan = (0.0, 5.0)
    Δₜ = 1e-4
    Δ_saveat = 1e2 * Δₜ
    simulation_parameters = sim_params(tspan, Δₜ, Ntraj, Δ_saveat)

    # System Dimensions
    n, m, d = 2, 1, 2
    system_dimensions = sys_dims(n, m, d)

    # System constants
    λ = 10.0
    σ_scale = 1.0

    # System matrices
    A = @SMatrix [0.0 1.0; 0.0 0.0]
    B = @SMatrix [0.0; 1.0]
    C = SMatrix{2,2}(1.0I)
    D = 0.0

    sys = ss(A, B, C, D)
    DesiredPoles = -λ * ones(2)
    K = SMatrix{1,2}(place(sys, DesiredPoles))
    A_cl = A - B * K

    # All qunatities whose size cannot be determined at compile-time need to go inside the dynamics_params tuple for GPU compatibility
    # E.g., instead of declaring global consts, we wish to keep A_cl, B, and K as variables, hence will be collected into dynamics_params
    # Build params tuple for GPU
    

    # Dynamics functions

    trck_traj(t) = @SVector [5*sin(t) + 3*cos(2*t), 0.0]

    dynamics_params = (; A_cl, B, K) # Collects items for dynamics whose size cannot be determined at compile-time (for GPU)
    f(t, x, dynamics_params) = dynamics_params.A_cl * x + dynamics_params.B * dynamics_params.K * trck_traj(t)

    g(t, x, dynamics_params) = @SVector [0.0, 1.0]
    g_perp(t, x, dynamics_params) = @SVector [1.0, 0.0]

    p_um(t, x, dynamics_params) = 2.0 * @SMatrix [0.01 0.1]
    p_m(t, x, dynamics_params) = 1.0 * @SMatrix [0.0 0.8]
    p(t, x, dynamics_params) = vcat(p_um(t, x, dynamics_params), p_m(t, x, dynamics_params))

    Λμ_um(t, x, dynamics_params) = 1e-2 * (1 + sin(x[1]))
    Λμ_m(t, x, dynamics_params) = 1.0 * (5 + 10*cos(x[2]) + 5*norm(x))
    Λμ(t, x, dynamics_params) = @SVector [Λμ_um(t, x, dynamics_params), Λμ_m(t, x, dynamics_params)]

    Λσ_um(t, x, dynamics_params) = 0.0 * @SMatrix [0.1+cos(x[2]) 2.0]
    Λσ_m(t, x, dynamics_params) = σ_scale * @SMatrix [0.0 5+sin(x[2])+5.0*(norm(x) < 1 ? norm(x) : sqrt(norm(x)))]
    Λσ(t, x, dynamics_params) = vcat(Λσ_um(t, x, dynamics_params), Λσ_m(t, x, dynamics_params))

    # Nominal Vector Fields
    nominal_components = nominal_vector_fields(f, g, g_perp, p, dynamics_params)

    # Uncertain Vector Fields
    uncertain_components = uncertain_vector_fields(Λμ, Λσ)

    # Initial distributions
    nominal_ξ₀ = MvNormal(1e-2 * ones(2), 1.0 * I(2))
    true_ξ₀ = MvNormal(-1.0 * ones(2), 1e-1 * I(2))
    initial_distributions = init_dist(nominal_ξ₀, true_ξ₀)

    # Define the systems
    nominal_system = nom_sys(system_dimensions, nominal_components, initial_distributions)
    true_system = true_sys(system_dimensions, nominal_components, uncertain_components, initial_distributions)

    # L1 DRAC Parameters
    ω = 50.0
    Tₛ = 10 * Δₜ
    λₛ = 100.0
    L1params = drac_params(ω, Tₛ, λₛ)

    return (
        simulation_parameters = simulation_parameters,
        system_dimensions = system_dimensions,
        dynamics_params = dynamics_params,
        nominal_system = nominal_system,
        true_system = true_system,
        L1params = L1params
    )
end

function debug_isbits(setup)
    @unpack f, g, g_perp, p, dynamics_params = setup.true_system.nom_vec_fields
    @unpack Λμ, Λσ = setup.true_system.unc_vec_fields
    @unpack true_ξ₀ = setup.true_system.init_dists

    println("=== Individual Components ===")
    println("dynamics_params isbits: ", isbitstype(typeof(dynamics_params)))
    println("f isbits: ", isbitstype(typeof(f)))
    println("g isbits: ", isbitstype(typeof(g)))
    println("g_perp isbits: ", isbitstype(typeof(g_perp)))
    println("p isbits: ", isbitstype(typeof(p)))
    println("Λμ isbits: ", isbitstype(typeof(Λμ)))
    println("Λσ isbits: ", isbitstype(typeof(Λσ)))

    println("\n=== Distributions ===")
    println("true_ξ₀ isbits: ", isbitstype(typeof(true_ξ₀)))

    println("\n=== L1 Params Tuple ===")
    L1_params_tuple = (dynamics_params, Float32(50.0), Float32(0.001), Float32(100.0), Float32(0.0001))
    println("L1_params_tuple isbits: ", isbitstype(typeof(L1_params_tuple)))

    println("\n=== Test Closures (like Method 2) ===")
    drift_simple(X, dp, t) = f(t, X, dp) + Λμ(t, X, dp)
    println("drift_simple (captures f, Λμ) isbits: ", isbitstype(typeof(drift_simple)))

    println("\n=== Test Closures (like Method 3) ===")
    drift_L1_like(X, dp, t) = f(t, X, dp) + g(t, X, dp) + g_perp(t, X, dp) + Λμ(t, X, dp)
    println("drift_L1_like (captures f, g, g_perp, Λμ) isbits: ", isbitstype(typeof(drift_L1_like)))

    println("\n=== Test Closure with dimension capture ===")
    n_gpu = 2
    drift_with_dim(Z, dp, t) = Z[SOneTo(n_gpu)]
    println("drift_with_dim (captures n_gpu=2) isbits: ", isbitstype(typeof(drift_with_dim)))

    println("\n=== Test SDEProblem (like Method 2) ===")
    drift_m2(X, dp, t) = f(t, X, dp) + Λμ(t, X, dp)
    diff_m2(X, dp, t) = p(t, X, dp) + Λσ(t, X, dp)
    u0_m2 = SVector{2}(1.0f0, 1.0f0)
    prob_m2 = SDEProblem(drift_m2, diff_m2, u0_m2, (0.0f0, 1.0f0), dynamics_params,
                         noise_rate_prototype = SMatrix{2,2}(zeros(Float32, 2, 2)))
    println("SDEProblem Method 2 style isbits: ", isbitstype(typeof(prob_m2)))

    println("\n=== Test SDEProblem (like Method 3) ===")
    L1_params = (dynamics_params, 50.0f0, 0.001f0, 100.0f0, 0.0001f0)
    drift_m3(Z, prms, t) = vcat(f(t, Z[SOneTo(2)], prms[1]), Z[SOneTo(2)])
    diff_m3(Z, prms, t) = vcat(p(t, Z[SOneTo(2)], prms[1]) + Λσ(t, Z[SOneTo(2)], prms[1]),
                               @SMatrix zeros(Float32, 2, 2))
    u0_m3 = SVector{4}(1.0f0, 1.0f0, 0.0f0, 0.0f0)
    prob_m3 = SDEProblem(drift_m3, diff_m3, u0_m3, (0.0f0, 1.0f0), L1_params,
                         noise_rate_prototype = SMatrix{4,2}(zeros(Float32, 4, 2)))
    println("SDEProblem Method 3 style isbits: ", isbitstype(typeof(prob_m3)))

    println("\n=== Test with Val pattern (like actual Method 3) ===")
    test_prob = _test_val_pattern_outer(f, g, g_perp, p, Λμ, Λσ, dynamics_params)
    println("SDEProblem with Val pattern isbits: ", isbitstype(typeof(test_prob)))
end

# Mimics the actual Method 3 structure with Val pattern
function _test_val_pattern_outer(f, g, g_perp, p, Λμ, Λσ, dynamics_params)
    _test_val_pattern(f, g, g_perp, p, Λμ, Λσ, dynamics_params, Val(2), Val(1), Val(2))
end

function _test_val_pattern(f, g, g_perp, p, Λμ, Λσ, dynamics_params,
                           ::Val{n_gpu}, ::Val{m_gpu}, ::Val{d_gpu}) where {n_gpu, m_gpu, d_gpu}

    L1_params = (dynamics_params, 50.0f0, 0.001f0, 100.0f0, 0.0001f0)

    function drift_val(Z, params, t)
        dp, ω, Tₛ, λₛ, Δₜ = params
        X = Z[SOneTo(n_gpu)]
        Xhat = Z[StaticArrays.SUnitRange(n_gpu+1, 2n_gpu)]

        gbar_t = hcat(g(t, X, dp), g_perp(t, X, dp))
        dX = f(t, X, dp) + Λμ(t, X, dp)
        dXhat = f(t, X, dp)
        return vcat(dX, dXhat)
    end

    function diff_val(Z, params, t)
        dp = params[1]
        X = Z[SOneTo(n_gpu)]
        dX = p(t, X, dp) + Λσ(t, X, dp)
        return vcat(dX, @SMatrix zeros(Float32, n_gpu, d_gpu))
    end

    u0 = SVector{2n_gpu}(ones(Float32, 2n_gpu))
    SDEProblem(drift_val, diff_val, u0, (0.0f0, 1.0f0), L1_params,
               noise_rate_prototype = SMatrix{2n_gpu, d_gpu}(zeros(Float32, 2n_gpu, d_gpu)))
end

# Test: Method 2 vs Method 3 closure differences
function debug_closure_captures(setup)
    @unpack f, g, g_perp, p, dynamics_params = setup.true_system.nom_vec_fields
    @unpack Λμ, Λσ = setup.true_system.unc_vec_fields

    println("=== Method 2 style (no dimension capture) ===")
    drift_m2(X, dp, t) = f(t, X, dp) + Λμ(t, X, dp)
    println("drift_m2 isbits: ", isbitstype(typeof(drift_m2)))

    println("\n=== Method 3 style (captures local n_gpu) ===")
    n_gpu = 2
    drift_m3_local(Z, dp, t) = Z[SOneTo(n_gpu)] + f(t, Z[SOneTo(n_gpu)], dp)
    println("drift_m3_local (captures n_gpu as local Int) isbits: ", isbitstype(typeof(drift_m3_local)))

    println("\n=== Method 3 via Val pattern (functions as args) ===")
    prob = _test_val_pattern_outer(f, g, g_perp, p, Λμ, Λσ, dynamics_params)
    println("SDEProblem from _test_val_pattern isbits: ", isbitstype(typeof(prob)))

    println("\n=== Method 3 via Val pattern (functions from struct) ===")
    prob2 = _test_val_pattern_from_struct(setup.true_system)
    println("SDEProblem from _test_val_pattern_from_struct isbits: ", isbitstype(typeof(prob2)))
end

# This mimics how the actual L1simfunctionsGPU.jl extracts functions from struct
function _test_val_pattern_from_struct(true_system)
    _test_val_pattern_inner(true_system, Val(2), Val(1), Val(2))
end

function _test_val_pattern_inner(true_system,
                                  ::Val{n_gpu}, ::Val{m_gpu}, ::Val{d_gpu}) where {n_gpu, m_gpu, d_gpu}
    # Extract from struct (same as actual L1simfunctionsGPU.jl)
    @unpack f, g, g_perp, p, dynamics_params = getfield(true_system, :nom_vec_fields)
    @unpack Λμ, Λσ = getfield(true_system, :unc_vec_fields)

    L1_params = (dynamics_params, 50.0f0, 0.001f0, 100.0f0, 0.0001f0)

    function drift_val(Z, params, t)
        dp, ω, Tₛ, λₛ, Δₜ = params
        X = Z[SOneTo(n_gpu)]
        Xhat = Z[StaticArrays.SUnitRange(n_gpu+1, 2n_gpu)]

        gbar_t = hcat(g(t, X, dp), g_perp(t, X, dp))
        dX = f(t, X, dp) + Λμ(t, X, dp)
        dXhat = f(t, X, dp)
        return vcat(dX, dXhat)
    end

    function diff_val(Z, params, t)
        dp = params[1]
        X = Z[SOneTo(n_gpu)]
        dX = p(t, X, dp) + Λσ(t, X, dp)
        return vcat(dX, @SMatrix zeros(Float32, n_gpu, d_gpu))
    end

    println("  drift_val isbits: ", isbitstype(typeof(drift_val)))
    println("  diff_val isbits: ", isbitstype(typeof(diff_val)))

    u0 = SVector{2n_gpu}(ones(Float32, 2n_gpu))
    SDEProblem(drift_val, diff_val, u0, (0.0f0, 1.0f0), L1_params,
               noise_rate_prototype = SMatrix{2n_gpu, d_gpu}(zeros(Float32, 2n_gpu, d_gpu)))
end

# Test exact copy of actual L1simfunctionsGPU.jl drift/diffusion
function debug_exact_L1_copy(setup)
    println("=== Exact copy of L1simfunctionsGPU.jl structure ===")
    prob = _exact_L1_copy(setup.true_system, Val(2), Val(1), Val(2))
    println("SDEProblem isbits: ", isbitstype(typeof(prob)))
end

function _exact_L1_copy(true_system, ::Val{n_gpu}, ::Val{m_gpu}, ::Val{d_gpu}) where {n_gpu, m_gpu, d_gpu}
    # Exact same extraction as L1simfunctionsGPU.jl
    @unpack f, g, g_perp, p, dynamics_params = getfield(true_system, :nom_vec_fields)
    @unpack Λμ, Λσ = getfield(true_system, :unc_vec_fields)

    L1_params_tuple = (dynamics_params, Float32(50.0), Float32(0.001), Float32(100.0), Float32(0.0001))

    # Exact copy of drift_L1_gpu from L1simfunctionsGPU.jl
    function drift_L1_gpu(Z, params, t)
        dynamics_params, ω, Tₛ, λₛ, Δₜ = params

        X = Z[SOneTo(n_gpu)]
        Xhat = Z[StaticArrays.SUnitRange(n_gpu+1, 2n_gpu)]
        Xfilter = Z[StaticArrays.SUnitRange(2n_gpu+1, 2n_gpu+m_gpu)]
        Λhat = Z[StaticArrays.SUnitRange(2n_gpu+m_gpu+1, 3n_gpu+m_gpu)]

        gbar_t = hcat(g(t, X, dynamics_params), g_perp(t, X, dynamics_params))
        Θ_t = hcat(SMatrix{m_gpu, m_gpu}(I), @SMatrix(zeros(eltype(Z), m_gpu, n_gpu-m_gpu))) * inv(gbar_t)
        Λhat_m = Θ_t * Λhat
        uₐ = m_gpu == 1 ? -only(Xfilter) : -Xfilter

        dXfilter = -ω * Xfilter + ω * Λhat_m
        dXhat = -λₛ * (Xhat - X) + f(t, X, dynamics_params) + g(t, X, dynamics_params) * uₐ + Λhat
        dX = f(t, X, dynamics_params) + g(t, X, dynamics_params) * uₐ + Λμ(t, X, dynamics_params)

        is_crossover = (floor(t / Tₛ) > floor((t - Δₜ) / Tₛ)) && (t >= Tₛ)
        if is_crossover
            Λhat_new = (λₛ / (1 - exp(λₛ * Tₛ))) * (Xhat - X)
            dΛhat = (Λhat_new - Λhat) / Δₜ
        else
            dΛhat = zero(Λhat)
        end

        return vcat(dX, dXhat, dXfilter, dΛhat)
    end

    # Exact copy of diffusion_L1_gpu from L1simfunctionsGPU.jl
    function diffusion_L1_gpu(Z, params, t)
        dynamics_params = params[1]
        X = Z[SOneTo(n_gpu)]

        dX_noise = SMatrix{n_gpu, d_gpu}(p(t, X, dynamics_params) + Λσ(t, X, dynamics_params))

        return vcat(dX_noise,
                    @SMatrix(zeros(eltype(Z), n_gpu, d_gpu)),
                    @SMatrix(zeros(eltype(Z), m_gpu, d_gpu)),
                    @SMatrix(zeros(eltype(Z), n_gpu, d_gpu)))
    end

    println("  drift_L1_gpu isbits: ", isbitstype(typeof(drift_L1_gpu)))
    println("  diffusion_L1_gpu isbits: ", isbitstype(typeof(diffusion_L1_gpu)))

    total_state_size = 3 * n_gpu + m_gpu
    u0 = SVector{total_state_size}(zeros(Float32, total_state_size))

    SDEProblem(drift_L1_gpu, diffusion_L1_gpu, u0, (0.0f0, 1.0f0), L1_params_tuple,
               noise_rate_prototype = SMatrix{total_state_size, d_gpu}(zeros(Float32, total_state_size, d_gpu)))
end

# Binary search: which operation breaks isbits?
function debug_isolate_isbits_failure(setup)
    @unpack f, g, g_perp, p, dynamics_params = setup.true_system.nom_vec_fields
    @unpack Λμ, Λσ = setup.true_system.unc_vec_fields

    n_gpu, m_gpu, d_gpu = 2, 1, 2

    println("=== Isolating isbits failure ===")

    # Test 1: Just m_gpu capture
    drift1(Z, t) = Z[StaticArrays.SUnitRange(2n_gpu+1, 2n_gpu+m_gpu)]
    println("1. captures m_gpu: ", isbitstype(typeof(drift1)))

    # Test 2: SMatrix with m_gpu
    drift2(Z, t) = SMatrix{m_gpu, m_gpu}(I)
    println("2. SMatrix{m_gpu,m_gpu}(I): ", isbitstype(typeof(drift2)))

    # Test 3: inv() call
    drift3(Z, t) = inv(SMatrix{2,2}(1.0I))
    println("3. inv() on SMatrix: ", isbitstype(typeof(drift3)))

    # Test 4: floor/exp
    drift4(Z, t) = floor(t / 0.001) > floor((t - 0.0001) / 0.001)
    println("4. floor() comparison: ", isbitstype(typeof(drift4)))

    # Test 5: if-else with zero()
    drift5(Z, t) = t > 0.5 ? Z : zero(Z)
    println("5. if-else with zero(): ", isbitstype(typeof(drift5)))

    # Test 6: only()
    drift6(Z, t) = only(Z[SOneTo(1)])
    println("6. only(): ", isbitstype(typeof(drift6)))

    # Test 7: All functions together (no control logic)
    drift7(Z, params, t) = f(t, Z[SOneTo(n_gpu)], params) + g(t, Z[SOneTo(n_gpu)], params) * Z[SOneTo(m_gpu)] + Λμ(t, Z[SOneTo(n_gpu)], params)
    println("7. f+g+Λμ combined: ", isbitstype(typeof(drift7)))

    # Test 8: hcat of g, g_perp
    drift8(Z, params, t) = hcat(g(t, Z[SOneTo(n_gpu)], params), g_perp(t, Z[SOneTo(n_gpu)], params))
    println("8. hcat(g, g_perp): ", isbitstype(typeof(drift8)))

    # Test 9: Full gbar + inv
    drift9(Z, params, t) = begin
        X = Z[SOneTo(n_gpu)]
        gbar = hcat(g(t, X, params), g_perp(t, X, params))
        inv(gbar)
    end
    println("9. hcat(g,g_perp) + inv: ", isbitstype(typeof(drift9)))

    # Test 10: Θ_t calculation
    drift10(Z, params, t) = begin
        X = Z[SOneTo(n_gpu)]
        gbar = hcat(g(t, X, params), g_perp(t, X, params))
        hcat(SMatrix{m_gpu, m_gpu}(I), @SMatrix(zeros(eltype(Z), m_gpu, n_gpu-m_gpu))) * inv(gbar)
    end
    println("10. Full Θ_t calc: ", isbitstype(typeof(drift10)))

    # Test 11: Same tests but inside Val pattern
    println("\n=== Now testing inside Val pattern ===")
    _test_inside_val_pattern(f, g, g_perp, Λμ, Val(2), Val(1), Val(2))
end

function _test_inside_val_pattern(f, g, g_perp, Λμ, ::Val{n_gpu}, ::Val{m_gpu}, ::Val{d_gpu}) where {n_gpu, m_gpu, d_gpu}
    # Test with where-clause type parameters
    drift_a(Z, t) = Z[StaticArrays.SUnitRange(2n_gpu+1, 2n_gpu+m_gpu)]
    println("11a. Val pattern m_gpu capture: ", isbitstype(typeof(drift_a)))

    drift_b(Z, params, t) = begin
        X = Z[SOneTo(n_gpu)]
        gbar = hcat(g(t, X, params), g_perp(t, X, params))
        hcat(SMatrix{m_gpu, m_gpu}(I), @SMatrix(zeros(eltype(Z), m_gpu, n_gpu-m_gpu))) * inv(gbar)
    end
    println("11b. Val pattern Θ_t calc: ", isbitstype(typeof(drift_b)))

    # Full drift with all operations
    drift_full(Z, params, t) = begin
        dynamics_params, ω, Tₛ, λₛ, Δₜ = params
        X = Z[SOneTo(n_gpu)]
        Xhat = Z[StaticArrays.SUnitRange(n_gpu+1, 2n_gpu)]
        Xfilter = Z[StaticArrays.SUnitRange(2n_gpu+1, 2n_gpu+m_gpu)]
        Λhat = Z[StaticArrays.SUnitRange(2n_gpu+m_gpu+1, 3n_gpu+m_gpu)]

        gbar_t = hcat(g(t, X, dynamics_params), g_perp(t, X, dynamics_params))
        Θ_t = hcat(SMatrix{m_gpu, m_gpu}(I), @SMatrix(zeros(eltype(Z), m_gpu, n_gpu-m_gpu))) * inv(gbar_t)
        Λhat_m = Θ_t * Λhat
        uₐ = m_gpu == 1 ? -only(Xfilter) : -Xfilter

        dXfilter = -ω * Xfilter + ω * Λhat_m
        dXhat = -λₛ * (Xhat - X) + f(t, X, dynamics_params) + g(t, X, dynamics_params) * uₐ + Λhat
        dX = f(t, X, dynamics_params) + g(t, X, dynamics_params) * uₐ + Λμ(t, X, dynamics_params)

        is_crossover = (floor(t / Tₛ) > floor((t - Δₜ) / Tₛ)) && (t >= Tₛ)
        if is_crossover
            Λhat_new = (λₛ / (1 - exp(λₛ * Tₛ))) * (Xhat - X)
            dΛhat = (Λhat_new - Λhat) / Δₜ
        else
            dΛhat = zero(Λhat)
        end

        return vcat(dX, dXhat, dXfilter, dΛhat)
    end
    println("11c. Val pattern full drift: ", isbitstype(typeof(drift_full)))
end

# Test 12: Extract from struct inside Val pattern (like actual failing code)
function debug_struct_extraction(setup)
    println("=== Testing struct extraction inside Val pattern ===")
    _test_struct_extraction(setup.true_system, Val(2), Val(1), Val(2))
end

function _test_struct_extraction(true_system, ::Val{n_gpu}, ::Val{m_gpu}, ::Val{d_gpu}) where {n_gpu, m_gpu, d_gpu}
    # Extract from struct (THIS is what the actual code does)
    @unpack f, g, g_perp, p, dynamics_params = getfield(true_system, :nom_vec_fields)
    @unpack Λμ, Λσ = getfield(true_system, :unc_vec_fields)

    # Simple closure capturing extracted functions
    drift_simple(Z, params, t) = f(t, Z[SOneTo(n_gpu)], params) + Λμ(t, Z[SOneTo(n_gpu)], params)
    println("12a. Simple (f+Λμ from struct): ", isbitstype(typeof(drift_simple)))

    # Add g, g_perp
    drift_with_g(Z, params, t) = begin
        X = Z[SOneTo(n_gpu)]
        f(t, X, params) + g(t, X, params) + g_perp(t, X, params) + Λμ(t, X, params)
    end
    println("12b. With g,g_perp from struct: ", isbitstype(typeof(drift_with_g)))

    # Full drift
    drift_full(Z, params, t) = begin
        dynamics_params, ω, Tₛ, λₛ, Δₜ = params
        X = Z[SOneTo(n_gpu)]
        Xhat = Z[StaticArrays.SUnitRange(n_gpu+1, 2n_gpu)]
        Xfilter = Z[StaticArrays.SUnitRange(2n_gpu+1, 2n_gpu+m_gpu)]
        Λhat = Z[StaticArrays.SUnitRange(2n_gpu+m_gpu+1, 3n_gpu+m_gpu)]

        gbar_t = hcat(g(t, X, dynamics_params), g_perp(t, X, dynamics_params))
        Θ_t = hcat(SMatrix{m_gpu, m_gpu}(I), @SMatrix(zeros(eltype(Z), m_gpu, n_gpu-m_gpu))) * inv(gbar_t)
        Λhat_m = Θ_t * Λhat
        uₐ = m_gpu == 1 ? -only(Xfilter) : -Xfilter

        dXfilter = -ω * Xfilter + ω * Λhat_m
        dXhat = -λₛ * (Xhat - X) + f(t, X, dynamics_params) + g(t, X, dynamics_params) * uₐ + Λhat
        dX = f(t, X, dynamics_params) + g(t, X, dynamics_params) * uₐ + Λμ(t, X, dynamics_params)

        is_crossover = (floor(t / Tₛ) > floor((t - Δₜ) / Tₛ)) && (t >= Tₛ)
        if is_crossover
            Λhat_new = (λₛ / (1 - exp(λₛ * Tₛ))) * (Xhat - X)
            dΛhat = (Λhat_new - Λhat) / Δₜ
        else
            dΛhat = zero(Λhat)
        end

        return vcat(dX, dXhat, dXfilter, dΛhat)
    end
    println("12c. Full drift from struct: ", isbitstype(typeof(drift_full)))

    # Narrowing down: what breaks between 12b and 12c?
    println("\n--- Narrowing down 12b → 12c ---")

    # 12d: Add Xfilter, Λhat extraction
    drift_12d(Z, params, t) = begin
        X = Z[SOneTo(n_gpu)]
        Xfilter = Z[StaticArrays.SUnitRange(2n_gpu+1, 2n_gpu+m_gpu)]
        Λhat = Z[StaticArrays.SUnitRange(2n_gpu+m_gpu+1, 3n_gpu+m_gpu)]
        f(t, X, params) + g(t, X, params) + Λμ(t, X, params)
    end
    println("12d. + Xfilter,Λhat extraction: ", isbitstype(typeof(drift_12d)))

    # 12e: Add Θ_t calculation
    drift_12e(Z, params, t) = begin
        X = Z[SOneTo(n_gpu)]
        gbar_t = hcat(g(t, X, params), g_perp(t, X, params))
        Θ_t = hcat(SMatrix{m_gpu, m_gpu}(I), @SMatrix(zeros(eltype(Z), m_gpu, n_gpu-m_gpu))) * inv(gbar_t)
        f(t, X, params) + Λμ(t, X, params)
    end
    println("12e. + Θ_t calculation: ", isbitstype(typeof(drift_12e)))

    # 12f: Add uₐ conditional
    drift_12f(Z, params, t) = begin
        X = Z[SOneTo(n_gpu)]
        Xfilter = Z[StaticArrays.SUnitRange(2n_gpu+1, 2n_gpu+m_gpu)]
        uₐ = m_gpu == 1 ? -only(Xfilter) : -Xfilter
        f(t, X, params) + g(t, X, params) * uₐ + Λμ(t, X, params)
    end
    println("12f. + uₐ conditional: ", isbitstype(typeof(drift_12f)))

    # 12g: Add sample-and-hold logic
    drift_12g(Z, params, t) = begin
        dynamics_params, ω, Tₛ, λₛ, Δₜ = params
        X = Z[SOneTo(n_gpu)]
        Xhat = Z[StaticArrays.SUnitRange(n_gpu+1, 2n_gpu)]
        is_crossover = (floor(t / Tₛ) > floor((t - Δₜ) / Tₛ)) && (t >= Tₛ)
        if is_crossover
            Λhat_new = (λₛ / (1 - exp(λₛ * Tₛ))) * (Xhat - X)
            dΛhat = Λhat_new / Δₜ
        else
            dΛhat = zero(X)
        end
        f(t, X, dynamics_params) + Λμ(t, X, dynamics_params) + dΛhat
    end
    println("12g. + sample-and-hold: ", isbitstype(typeof(drift_12g)))

    # 12h: Combine d,e,f (no sample-and-hold)
    drift_12h(Z, params, t) = begin
        X = Z[SOneTo(n_gpu)]
        Xfilter = Z[StaticArrays.SUnitRange(2n_gpu+1, 2n_gpu+m_gpu)]
        Λhat = Z[StaticArrays.SUnitRange(2n_gpu+m_gpu+1, 3n_gpu+m_gpu)]
        gbar_t = hcat(g(t, X, params), g_perp(t, X, params))
        Θ_t = hcat(SMatrix{m_gpu, m_gpu}(I), @SMatrix(zeros(eltype(Z), m_gpu, n_gpu-m_gpu))) * inv(gbar_t)
        Λhat_m = Θ_t * Λhat
        uₐ = m_gpu == 1 ? -only(Xfilter) : -Xfilter
        f(t, X, params) + g(t, X, params) * uₐ + Λμ(t, X, params)
    end
    println("12h. d+e+f combined (no sample-hold): ", isbitstype(typeof(drift_12h)))

    # Narrowing down sample-and-hold
    println("\n--- Isolating sample-and-hold failure ---")

    # 12i: Just floor
    drift_12i(Z, params, t) = begin
        dynamics_params, ω, Tₛ, λₛ, Δₜ = params
        is_crossover = floor(t / Tₛ) > floor((t - Δₜ) / Tₛ)
        Z[SOneTo(n_gpu)]
    end
    println("12i. floor() only: ", isbitstype(typeof(drift_12i)))

    # 12j: Just exp
    drift_12j(Z, params, t) = begin
        dynamics_params, ω, Tₛ, λₛ, Δₜ = params
        val = 1 - exp(λₛ * Tₛ)
        Z[SOneTo(n_gpu)] * val
    end
    println("12j. exp() only: ", isbitstype(typeof(drift_12j)))

    # 12k: if-else with zero
    drift_12k(Z, params, t) = begin
        X = Z[SOneTo(n_gpu)]
        if t > 0.5
            dΛhat = X
        else
            dΛhat = zero(X)
        end
        dΛhat
    end
    println("12k. if-else with zero(): ", isbitstype(typeof(drift_12k)))

    # 12l: Full sample-and-hold but simpler
    drift_12l(Z, params, t) = begin
        dynamics_params, ω, Tₛ, λₛ, Δₜ = params
        X = Z[SOneTo(n_gpu)]
        Xhat = Z[StaticArrays.SUnitRange(n_gpu+1, 2n_gpu)]
        is_crossover = (floor(t / Tₛ) > floor((t - Δₜ) / Tₛ))
        if is_crossover
            dΛhat = (Xhat - X) / Δₜ
        else
            dΛhat = zero(X)
        end
        dΛhat
    end
    println("12l. sample-hold (no exp): ", isbitstype(typeof(drift_12l)))

    # 12m: Add exp back
    drift_12m(Z, params, t) = begin
        dynamics_params, ω, Tₛ, λₛ, Δₜ = params
        X = Z[SOneTo(n_gpu)]
        Xhat = Z[StaticArrays.SUnitRange(n_gpu+1, 2n_gpu)]
        is_crossover = (floor(t / Tₛ) > floor((t - Δₜ) / Tₛ))
        if is_crossover
            Λhat_new = (λₛ / (1 - exp(λₛ * Tₛ))) * (Xhat - X)
            dΛhat = Λhat_new / Δₜ
        else
            dΛhat = zero(X)
        end
        dΛhat
    end
    println("12m. sample-hold (with exp): ", isbitstype(typeof(drift_12m)))

    # 12n: Add the && (t >= Tₛ) condition
    drift_12n(Z, params, t) = begin
        dynamics_params, ω, Tₛ, λₛ, Δₜ = params
        X = Z[SOneTo(n_gpu)]
        Xhat = Z[StaticArrays.SUnitRange(n_gpu+1, 2n_gpu)]
        is_crossover = (floor(t / Tₛ) > floor((t - Δₜ) / Tₛ)) && (t >= Tₛ)
        if is_crossover
            Λhat_new = (λₛ / (1 - exp(λₛ * Tₛ))) * (Xhat - X)
            dΛhat = Λhat_new / Δₜ
        else
            dΛhat = zero(X)
        end
        dΛhat
    end
    println("12n. sample-hold (full condition): ", isbitstype(typeof(drift_12n)))

    # 12o: floor without any function capture - just n_gpu
    println("\n--- Testing floor/exp isolation ---")
    drift_12o(Z, t) = begin
        is_crossover = floor(t / 0.001) > floor((t - 0.0001) / 0.001)
        Z[SOneTo(n_gpu)]
    end
    println("12o. floor (constants, only n_gpu): ", isbitstype(typeof(drift_12o)))

    # 12p: floor with params but no functions in scope reference
    drift_12p(Z, params, t) = begin
        dp, ω, Tₛ, λₛ, Δₜ = params
        is_crossover = floor(t / Tₛ) > floor((t - Δₜ) / Tₛ)
        Z[SOneTo(n_gpu)]
    end
    println("12p. floor (params, only n_gpu): ", isbitstype(typeof(drift_12p)))

    # 12q: Does just having f in scope (but not using it) break things?
    drift_12q(Z, t) = Z[SOneTo(n_gpu)]  # f, g, etc are in scope but NOT used
    println("12q. minimal (f,g in scope, unused): ", isbitstype(typeof(drift_12q)))

    # 12r: Use f, then add floor
    drift_12r(Z, params, t) = begin
        X = Z[SOneTo(n_gpu)]
        result = f(t, X, params)
        is_crossover = floor(t / 0.001) > 0
        result
    end
    println("12r. f + floor (constants): ", isbitstype(typeof(drift_12r)))

    # 12s: Use f, floor with params
    drift_12s(Z, params, t) = begin
        dp, ω, Tₛ, λₛ, Δₜ = params
        X = Z[SOneTo(n_gpu)]
        result = f(t, X, dp)
        is_crossover = floor(t / Tₛ) > 0
        result
    end
    println("12s. f + floor (from params): ", isbitstype(typeof(drift_12s)))

    # 12t: Test if dynamics_params name shadowing is the issue
    println("\n--- Testing variable name shadowing ---")
    drift_12t(Z, params, t) = begin
        dynamics_params, ω, Tₛ, λₛ, Δₜ = params  # shadows outer dynamics_params
        is_crossover = floor(t / Tₛ) > floor((t - Δₜ) / Tₛ)
        Z[SOneTo(n_gpu)]
    end
    println("12t. floor + shadow dynamics_params: ", isbitstype(typeof(drift_12t)))

    drift_12u(Z, params, t) = begin
        dp, ω, Tₛ, λₛ, Δₜ = params  # different name, no shadow
        is_crossover = floor(t / Tₛ) > floor((t - Δₜ) / Tₛ)
        Z[SOneTo(n_gpu)]
    end
    println("12u. floor + dp (no shadow): ", isbitstype(typeof(drift_12u)))

    # 12v: Same as 12g but with dp instead of dynamics_params
    drift_12v(Z, params, t) = begin
        dp, ω, Tₛ, λₛ, Δₜ = params
        X = Z[SOneTo(n_gpu)]
        Xhat = Z[StaticArrays.SUnitRange(n_gpu+1, 2n_gpu)]
        is_crossover = (floor(t / Tₛ) > floor((t - Δₜ) / Tₛ)) && (t >= Tₛ)
        if is_crossover
            Λhat_new = (λₛ / (1 - exp(λₛ * Tₛ))) * (Xhat - X)
            dΛhat = Λhat_new / Δₜ
        else
            dΛhat = zero(X)
        end
        f(t, X, dp) + Λμ(t, X, dp) + dΛhat
    end
    println("12v. full sample-hold with dp: ", isbitstype(typeof(drift_12v)))
end
