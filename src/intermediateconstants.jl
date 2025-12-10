# ====================================================================
# Helper Functions
# ====================================================================

"""
    frakp(p::Int) -> Float64

Compute the fraktur-p helper function: sqrt(p*(p-1)/2).
Used in moment bounds for stochastic integrals.
"""
frakp(p::Int) = sqrt(p * (p - 1) / 2)

"""
    frakp_prime(p::Int) -> Float64

Compute the fraktur-p' helper function: sqrt(p*(2p-1)/2).
"""
frakp_prime(p::Int) = sqrt(p * (2p - 1) / 2)

"""
    frakp_double_prime(p::Int) -> Float64

Compute the fraktur-p'' helper function: sqrt(p*(4p-1)).
"""
frakp_double_prime(p::Int) = sqrt(p * (4p - 1))

"""
    Lip_f(constants::AssumptionConstants) -> Int

Return 1 if Assumption 8 (stronger regularity) holds, 0 otherwise.
Checks if L_f was explicitly set and is positive.
"""
Lip_f(constants::AssumptionConstants) = (:L_f in constants._set_fields && constants.L_f > 0) ? 1 : 0


# ====================================================================
# Reference Process Constant Structs
# ====================================================================

"""
    DeltaHatRef

Hat constants for reference process analysis (Eq. A.2 in manuscript).

# Fields
- `Delta_hat_r_1`: Depends on p
- `Delta_hat_r_2`: Depends on p
- `Delta_hat_r_3`: Independent of p
- `Delta_hat_r_4`: Independent of p
"""
struct DeltaHatRef
    Delta_hat_r_1::Float64
    Delta_hat_r_2::Float64
    Delta_hat_r_3::Float64
    Delta_hat_r_4::Float64
end

"""
    DeltaCircRef

Circle constants (order 0) for reference process (Eq. A.3 in manuscript).
"""
struct DeltaCircRef
    Delta_r_circ_1::Float64
    Delta_r_circ_2::Float64
    Delta_r_circ_3::Float64
    Delta_r_circ_4::Float64
end

"""
    DeltaCircledCircRef

Circled-circle constants (order 1/2) for reference process (Eq. A.4 in manuscript).
"""
struct DeltaCircledCircRef
    Delta_r_circledcirc_1::Float64
    Delta_r_circledcirc_2::Float64
    Delta_r_circledcirc_3::Float64
    Delta_r_circledcirc_4::Float64
end

"""
    DeltaOdotRef

Odot constants (order 1) for reference process (Eq. A.5 in manuscript).
"""
struct DeltaOdotRef
    Delta_r_odot_1::Float64
    Delta_r_odot_2::Float64
    Delta_r_odot_3::Float64
    Delta_r_odot_4::Float64
    Delta_r_odot_5::Float64
    Delta_r_odot_6::Float64
    Delta_r_odot_7::Float64
    Delta_r_odot_8::Float64
end

"""
    DeltaOtimesRef

Otimes constants (order 3/2) for reference process (Eq. A.6 in manuscript).
"""
struct DeltaOtimesRef
    Delta_r_otimes_1::Float64
    Delta_r_otimes_2::Float64
    Delta_r_otimes_3::Float64
    Delta_r_otimes_4::Float64
    Delta_r_otimes_5::Float64
end

"""
    DeltaCircledAstRef

Circled-ast constants (order 2) for reference process (Eq. A.7 in manuscript).
"""
struct DeltaCircledAstRef
    Delta_r_circledast_1::Float64
    Delta_r_circledast_2::Float64
    Delta_r_circledast_3::Float64
end

"""
    ReferenceProcessConstants

Container for all reference process intermediate constants.

# Fields
- `hat::DeltaHatRef`
- `circ::DeltaCircRef`
- `circledcirc::DeltaCircledCircRef`
- `odot::DeltaOdotRef`
- `otimes::DeltaOtimesRef`
- `circledast::DeltaCircledAstRef`
"""
struct ReferenceProcessConstants
    hat::DeltaHatRef
    circ::DeltaCircRef
    circledcirc::DeltaCircledCircRef
    odot::DeltaOdotRef
    otimes::DeltaOtimesRef
    circledast::DeltaCircledAstRef
end


# ====================================================================
# True Process Constant Structs
# ====================================================================

"""
    DeltaHatTrue

Hat constants for true process analysis (Eq. A.8 in manuscript).

# Fields
- `Delta_hat_1`: Depends on p
- `Delta_hat_2`: Depends on p
- `Delta_hat_3`: Independent of p
- `Delta_hat_4`: Independent of p
- `Delta_hat_5`: Independent of p (uses m)
"""
struct DeltaHatTrue
    Delta_hat_1::Float64
    Delta_hat_2::Float64
    Delta_hat_3::Float64
    Delta_hat_4::Float64
    Delta_hat_5::Float64
end

"""
    DeltaCircledCircTrue

Circled-circle constants (order 1/2) for true process (Eq. A.9 in manuscript).
"""
struct DeltaCircledCircTrue
    Delta_circledcirc_1::Float64
end

"""
    DeltaOdotTrue

Odot constants (order 1) for true process (Eq. A.10 in manuscript).
"""
struct DeltaOdotTrue
    Delta_odot_1::Float64
    Delta_odot_2::Float64
    Delta_odot_3::Float64
    Delta_odot_4::Float64
end

"""
    DeltaOtimesTrue

Otimes constants (order 3/2) for true process (Eq. A.11 in manuscript).
"""
struct DeltaOtimesTrue
    Delta_otimes_1::Float64
    Delta_otimes_2::Float64
    Delta_otimes_3::Float64
    Delta_otimes_4::Float64
end

"""
    DeltaCircledAstTrue

Circled-ast constants (order 2) for true process (Eq. A.12 in manuscript).
"""
struct DeltaCircledAstTrue
    Delta_circledast_1::Float64
    Delta_circledast_2::Float64
    Delta_circledast_3::Float64
end

"""
    TrueProcessConstants

Container for all true process intermediate constants.

# Fields
- `hat::DeltaHatTrue`
- `circledcirc::DeltaCircledCircTrue`
- `odot::DeltaOdotTrue`
- `otimes::DeltaOtimesTrue`
- `circledast::DeltaCircledAstTrue`
"""
struct TrueProcessConstants
    hat::DeltaHatTrue
    circledcirc::DeltaCircledCircTrue
    odot::DeltaOdotTrue
    otimes::DeltaOtimesTrue
    circledast::DeltaCircledAstTrue
end


# ====================================================================
# Main Container
# ====================================================================

"""
    IntermediateConstants

Container for all intermediate constants used in L1-DRAC analysis.
Computed from AssumptionConstants for given system dimensions.

# Fields
- `reference::ReferenceProcessConstants` - Constants for reference process
- `true_process::TrueProcessConstants` - Constants for true (uncertain) process
- `p::Int` - Moment order (from constants.order_p)
- `m::Int` - Input dimension
- `d::Int` - Brownian motion dimension
- `Lip_f_indicator::Int` - Value of Lip{f} indicator (0 or 1)
"""
struct IntermediateConstants
    reference::ReferenceProcessConstants
    true_process::TrueProcessConstants
    p::Int
    m::Int
    d::Int
    Lip_f_indicator::Int
end


# ====================================================================
# Reference Process Computation Functions
# ====================================================================

"""
Compute hat constants for reference process (Eq. A.2).
"""
function _compute_hat_ref(c::AssumptionConstants, p::Int, lip_f::Int)
    @unpack Δg, Δg_dot, Δf, Δμ, Δσ, Δp, Δ_star, L_f, λ = c

    Delta_hat_r_1 = Δg * (
        (1 / sqrt(λ)) * (Δf * (2 + Δ_star) * (1 - lip_f) + Δμ) +
        frakp(p) * (2*Δp + Δσ)
    )
    Delta_hat_r_2 = Δg * frakp(p) * Δσ
    Delta_hat_r_3 = (1 / sqrt(λ)) * Δg * (Δf * (1 - lip_f) + Δμ)
    Delta_hat_r_4 = (1 / sqrt(λ)) * (Δg * L_f * lip_f + Δg_dot)

    return DeltaHatRef(Delta_hat_r_1, Delta_hat_r_2, Delta_hat_r_3, Delta_hat_r_4)
end

"""
Compute circle constants for reference process (Eq. A.3).
"""
function _compute_circ_ref(c::AssumptionConstants, hat::DeltaHatRef, p::Int, m::Int)
    @unpack Δp, Δσ, Δμ_parallel, Δσ_parallel, Δp_parallel, Δg, λ = c
    @unpack Delta_hat_r_1 = hat

    Delta_r_circ_1 = Δp^2 + (Δp + Δσ)^2
    Delta_r_circ_2 = (Δμ_parallel / sqrt(λ)) * (Delta_hat_r_1 + (Δg^2 * Δμ_parallel) / sqrt(λ))
    Delta_r_circ_3 = (frakp_prime(p) / sqrt(λ)) * (Δp_parallel + Δσ_parallel) *
                     (Delta_hat_r_1 + (2 * Δg^2 * Δμ_parallel) / sqrt(λ))
    Delta_r_circ_4 = (Δp_parallel + Δσ_parallel) * (Δg / λ) *
                     (frakp_prime(p)^2 * Δg * (Δp_parallel + Δσ_parallel) + sqrt(m) * (2*Δp + Δσ))

    return DeltaCircRef(Delta_r_circ_1, Delta_r_circ_2, Delta_r_circ_3, Delta_r_circ_4)
end

"""
Compute circled-circle constants for reference process (Eq. A.4).
"""
function _compute_circledcirc_ref(c::AssumptionConstants, hat::DeltaHatRef, p::Int, m::Int)
    @unpack Δg, Δσ, Δσ_parallel, Δp, Δμ_parallel, Δp_parallel, λ = c
    @unpack Delta_hat_r_1, Delta_hat_r_2 = hat

    Delta_r_circledcirc_1 = 2 * Δσ * (Δp + Δσ)
    Delta_r_circledcirc_2 = (Δμ_parallel / sqrt(λ)) * Delta_hat_r_2
    Delta_r_circledcirc_3 = (frakp_prime(p) / sqrt(λ)) * (
        (Δp_parallel + Δσ_parallel) * Delta_hat_r_2 +
        Δσ_parallel * (Delta_hat_r_1 + (2 * Δg^2 * Δμ_parallel) / sqrt(λ))
    )
    Delta_r_circledcirc_4 = (Δg / λ) * (
        (Δp_parallel + Δσ_parallel) * (2 * frakp_prime(p)^2 * Δg * Δσ_parallel + sqrt(m) * Δσ) +
        sqrt(m) * Δσ_parallel * (2*Δp + Δσ)
    )

    return DeltaCircledCircRef(Delta_r_circledcirc_1, Delta_r_circledcirc_2,
                                Delta_r_circledcirc_3, Delta_r_circledcirc_4)
end

"""
Compute odot constants for reference process (Eq. A.5).
"""
function _compute_odot_ref(c::AssumptionConstants, hat::DeltaHatRef, p::Int, m::Int)
    @unpack Δg, Δg_perp, Δσ, Δσ_parallel, Δσ_perp, Δp, Δμ_parallel, Δμ_perp,
            Δp_parallel, Δp_perp, λ = c
    @unpack Delta_hat_r_1, Delta_hat_r_2, Delta_hat_r_3, Delta_hat_r_4 = hat

    Delta_r_odot_1 = Δσ^2
    Delta_r_odot_2 = 2 * Δg_perp * Δμ_perp
    Delta_r_odot_3 = 2 * frakp(p) * (Δg_perp * (Δp_perp + Δσ_perp) + Δp)
    Delta_r_odot_4 = (Δμ_parallel / sqrt(λ)) * (Delta_hat_r_1 + Delta_hat_r_3 +
                     (2 * Δg^2 * Δμ_parallel) / sqrt(λ))
    Delta_r_odot_5 = Δμ_parallel * (Delta_hat_r_4 / sqrt(λ) + 4 * Δg) +
                     2 * sqrt(λ) * Δg * frakp(p) * (Δp_parallel + Δσ_parallel)
    Delta_r_odot_6 = (frakp_prime(p) / sqrt(λ)) * (
        (Δp_parallel + Δσ_parallel) * (Delta_hat_r_3 + (2 * Δg^2 * Δμ_parallel) / sqrt(λ)) +
        Δσ_parallel * Delta_hat_r_2
    )
    Delta_r_odot_7 = frakp_prime(p) * (Δp_parallel + Δσ_parallel) *
                     (Delta_hat_r_4 / sqrt(λ) + 2 * Δg)
    Delta_r_odot_8 = Δσ_parallel * (Δg / λ) * (frakp_prime(p)^2 * Δg * Δσ_parallel + sqrt(m) * Δσ)

    return DeltaOdotRef(Delta_r_odot_1, Delta_r_odot_2, Delta_r_odot_3, Delta_r_odot_4,
                        Delta_r_odot_5, Delta_r_odot_6, Delta_r_odot_7, Delta_r_odot_8)
end

"""
Compute otimes constants for reference process (Eq. A.6).
"""
function _compute_otimes_ref(c::AssumptionConstants, hat::DeltaHatRef, p::Int)
    @unpack Δg, Δg_perp, Δσ_parallel, Δσ_perp, Δμ_parallel, λ = c
    @unpack Delta_hat_r_2, Delta_hat_r_3, Delta_hat_r_4 = hat

    Delta_r_otimes_1 = 2 * frakp(p) * Δg_perp * Δσ_perp
    Delta_r_otimes_2 = Δμ_parallel * (Delta_hat_r_2 / sqrt(λ))
    Delta_r_otimes_3 = 2 * frakp(p) * sqrt(λ) * Δg * Δσ_parallel
    Delta_r_otimes_4 = frakp_prime(p) * Δσ_parallel *
                       ((Delta_hat_r_3 + (2 * Δg^2 * Δμ_parallel) / sqrt(λ)) / sqrt(λ))
    Delta_r_otimes_5 = frakp_prime(p) * Δσ_parallel * (Delta_hat_r_4 / sqrt(λ) + 2 * Δg)

    return DeltaOtimesRef(Delta_r_otimes_1, Delta_r_otimes_2, Delta_r_otimes_3,
                          Delta_r_otimes_4, Delta_r_otimes_5)
end

"""
Compute circled-ast constants for reference process (Eq. A.7).
"""
function _compute_circledast_ref(c::AssumptionConstants, hat::DeltaHatRef)
    @unpack Δg, Δg_perp, Δμ_parallel, Δμ_perp, λ = c
    @unpack Delta_hat_r_3, Delta_hat_r_4 = hat

    Delta_r_circledast_1 = 2 * Δg_perp * Δμ_perp
    Delta_r_circledast_2 = Δμ_parallel * ((Delta_hat_r_3 + (Δg^2 * Δμ_parallel) / sqrt(λ)) / sqrt(λ))
    Delta_r_circledast_3 = Δμ_parallel * (Delta_hat_r_4 / sqrt(λ) + 4 * Δg)

    return DeltaCircledAstRef(Delta_r_circledast_1, Delta_r_circledast_2, Delta_r_circledast_3)
end

"""
Compute all reference process constants.
"""
function _compute_reference_constants(c::AssumptionConstants, p::Int, m::Int, lip_f::Int)
    hat = _compute_hat_ref(c, p, lip_f)
    circ = _compute_circ_ref(c, hat, p, m)
    circledcirc = _compute_circledcirc_ref(c, hat, p, m)
    odot = _compute_odot_ref(c, hat, p, m)
    otimes = _compute_otimes_ref(c, hat, p)
    circledast = _compute_circledast_ref(c, hat)

    return ReferenceProcessConstants(hat, circ, circledcirc, odot, otimes, circledast)
end


# ====================================================================
# True Process Computation Functions
# ====================================================================

"""
Compute hat constants for true process (Eq. A.8).
"""
function _compute_hat_true(c::AssumptionConstants, p::Int, m::Int, lip_f::Int)
    @unpack Δg, Δg_dot, Δf, L_μ, L_p, L_σ, L_f, λ = c

    Delta_hat_1 = (2 / sqrt(λ)) * Δg * Δf * (1 - lip_f)
    Delta_hat_2 = Δg * frakp(p) * (L_p + L_σ)
    Delta_hat_3 = (1 / sqrt(λ)) * Δg * Δf * (1 - lip_f)
    Delta_hat_4 = (1 / sqrt(λ)) * (Δg * (L_μ + L_f * lip_f) + Δg_dot)
    Delta_hat_5 = sqrt(m) * Δg * (L_p + L_σ)

    return DeltaHatTrue(Delta_hat_1, Delta_hat_2, Delta_hat_3, Delta_hat_4, Delta_hat_5)
end

"""
Compute circled-circle constants for true process (Eq. A.9).
"""
function _compute_circledcirc_true(c::AssumptionConstants, hat::DeltaHatTrue, p::Int)
    @unpack L_p_parallel, L_σ_parallel, λ = c
    @unpack Delta_hat_1 = hat

    Delta_circledcirc_1 = (1 / sqrt(λ)) * frakp_prime(p) * (L_p_parallel + L_σ_parallel) * Delta_hat_1

    return DeltaCircledCircTrue(Delta_circledcirc_1)
end

"""
Compute odot constants for true process (Eq. A.10).

Note: The manuscript references Delta_hat_6 in Delta_odot_4, but only defines
Delta_hat_1-5. Following the example implementation, we use Delta_hat_5 here.
"""
function _compute_odot_true(c::AssumptionConstants, hat::DeltaHatTrue, p::Int)
    @unpack L_p, L_σ, L_μ_parallel, L_p_parallel, L_σ_parallel, Δg, λ = c
    @unpack Delta_hat_1, Delta_hat_2, Delta_hat_5 = hat

    Delta_odot_1 = (L_p + L_σ)^2
    Delta_odot_2 = (1 / sqrt(λ)) * L_μ_parallel * Delta_hat_1
    Delta_odot_3 = (1 / sqrt(λ)) * frakp_prime(p) * (L_p_parallel + L_σ_parallel) * Delta_hat_2
    # Note: Using Delta_hat_5 where manuscript says Delta_hat_6 (confirmed from example code)
    Delta_odot_4 = (1 / λ) * (L_p_parallel + L_σ_parallel) * (
        Delta_hat_5 + Δg^2 * frakp_prime(p)^2 * (L_p_parallel + L_σ_parallel)
    )

    return DeltaOdotTrue(Delta_odot_1, Delta_odot_2, Delta_odot_3, Delta_odot_4)
end

"""
Compute otimes constants for true process (Eq. A.11).
"""
function _compute_otimes_true(c::AssumptionConstants, hat::DeltaHatTrue, p::Int)
    @unpack Δg, Δg_perp, L_p_parallel, L_σ_parallel, L_p_perp, L_σ_perp, L_μ_parallel, λ = c
    @unpack Delta_hat_2, Delta_hat_3, Delta_hat_4 = hat

    Delta_otimes_1 = 2 * Δg_perp * frakp(p) * (L_p_perp + L_σ_perp)
    Delta_otimes_2 = 2 * sqrt(λ) * Δg * frakp(p) * (L_p_parallel + L_σ_parallel) +
                     L_μ_parallel * Delta_hat_2 / sqrt(λ)
    Delta_otimes_3 = (1 / sqrt(λ)) * frakp_prime(p) * (L_p_parallel + L_σ_parallel) * Delta_hat_3
    Delta_otimes_4 = frakp_prime(p) * (L_p_parallel + L_σ_parallel) *
                     (Delta_hat_4 / sqrt(λ) + 2 * Δg * (1 + (Δg / λ) * L_μ_parallel))

    return DeltaOtimesTrue(Delta_otimes_1, Delta_otimes_2, Delta_otimes_3, Delta_otimes_4)
end

"""
Compute circled-ast constants for true process (Eq. A.12).
"""
function _compute_circledast_true(c::AssumptionConstants, hat::DeltaHatTrue)
    @unpack Δg, Δg_perp, L_μ_parallel, L_μ_perp, λ = c
    @unpack Delta_hat_3, Delta_hat_4 = hat

    Delta_circledast_1 = 2 * Δg_perp * L_μ_perp
    Delta_circledast_2 = (1 / sqrt(λ)) * L_μ_parallel * Delta_hat_3
    Delta_circledast_3 = L_μ_parallel * (
        Delta_hat_4 / sqrt(λ) + Δg * (4 + (Δg / λ) * L_μ_parallel)
    )

    return DeltaCircledAstTrue(Delta_circledast_1, Delta_circledast_2, Delta_circledast_3)
end

"""
Compute all true process constants.
"""
function _compute_true_constants(c::AssumptionConstants, p::Int, m::Int, lip_f::Int)
    hat = _compute_hat_true(c, p, m, lip_f)
    circledcirc = _compute_circledcirc_true(c, hat, p)
    odot = _compute_odot_true(c, hat, p)
    otimes = _compute_otimes_true(c, hat, p)
    circledast = _compute_circledast_true(c, hat)

    return TrueProcessConstants(hat, circledcirc, odot, otimes, circledast)
end


# ====================================================================
# Factory Function
# ====================================================================

"""
    intermediate_constants(constants::AssumptionConstants, m::Int, d::Int)

Compute all intermediate constants from assumption constants.

# Arguments
- `constants::AssumptionConstants` - The validated assumption constants
- `m::Int` - Input dimension (system_dimensions.m)
- `d::Int` - Brownian motion dimension (system_dimensions.d)

# Returns
`IntermediateConstants` containing all computed reference and true process constants.

# Example
```julia
constants = assumption_constants(
    Δf = 5 + sqrt(34), Δg = 1, λ = 3.0, ...
)
validate(constants, true_system)

# m = input dimension, d = Brownian motion dimension
ic = intermediate_constants(constants, system_dimensions.m, system_dimensions.d)

# Access via nested structure
ic.reference.circ.Delta_r_circ_1
ic.reference.odot.Delta_r_odot_3
ic.true_process.odot.Delta_odot_1
ic.true_process.circledast.Delta_circledast_2
```

# Notes
- The moment order `p` is taken from `constants.order_p` internally
- The Lipschitz indicator is computed as: `Lip_f = 1` if `L_f > 0` was set, else `0`
- All intermediate constants are computed eagerly upon construction
"""
function intermediate_constants(constants::AssumptionConstants, sys_dims::SysDims)
    p = constants.order_p
    m, d = sys_dims.m, sys_dims.d
    lip_f = Lip_f(constants)

    reference = _compute_reference_constants(constants, p, m, lip_f)
    true_process = _compute_true_constants(constants, p, m, lip_f)

    return IntermediateConstants(reference, true_process, p, m, d, lip_f)
end
