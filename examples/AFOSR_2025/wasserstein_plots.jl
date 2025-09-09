using Plots
using LaTeXStrings
using JLD2
using Measures


ens_nom_sol = JLD2.load("ens_nom_solution.jld2")["ens_nom_sol"]
ens_L1_sol = JLD2.load("ens_L1_solution.jld2")["ens_L1_sol"]

L2_X0Xr_t0 = sqrt(mean(sum(abs2, ens_nom_sol[:,1,:]-ens_L1_sol[1:2,1,:]; dims=1)))
bin_width =0.9
Δt=0.1

colors = ["#A2CD5A", "#019CAA", "#F88379"]

function plot_wasserstein(
    tspan::Tuple{Float64,Float64},
    rho_r::Float64,
    rho_a::Float64,
    L2_X0Xr_t0::Float64,
    Δt::Float64,
    bin_width::Float64,

    savepath::AbstractString = "./Wasserstein_Plots.png"
)
    t_vals = collect(tspan[1]:Δt:tspan[2])

    # indices that correspond to those times (1, 50, …, 350)
    # ensure same length as t_vals:
    idx_num = Int(Δt*100)
    t_idxs = [1; idx_num:idx_num:idx_num*(length(t_vals)-1)]

    # W₂ at those indices
    w2_vals = [begin
        μ = @view ens_nom_sol[1:2, i, :]
        ν = @view ens_L1_sol[1:2, i, :]
        empirical_wasserstein2(μ, ν, bin_width)  # W2 (not squared)
    end for i in t_idxs]

    @assert length(t_vals) == length(w2_vals)

    Tₛ_rounded= round(Tₛ, digits=3)
    # Title — interpolate ω and Tₛ values
    p = plot(t_vals, w2_vals;
        linewidth = 4,
        color = colors[1],
        label = nothing,  # no legend entry for W2 curve
        xlabel = L"t",
        ylabel = L"\mathsf{W}_{2}\!\left(\mathbb{X}_{t},\,\mathbb{X}^\star_{t}\right)",
        title  = latexstring("\\omega = $(ω), T_s= $(Tₛ_rounded)"),
        xlim = (tspan[1], tspan[2]),
        ylim = (0, 40),
        titlefont = font(20, "Computer Modern"),
        yguidefontsize = 18, xguidefontsize = 18, legendfontsize = 18,
        tickfontsize = 16, gridlinewidth = 2.5, dpi = 300, margin = 5mm,
        legend = :right,
    )

    # ρ̂ curve at same times
    rhoHat_vals = [rhoHat(t, rho_r, rho_a, L2_X0Xr_t0,
                          assumption_constants, ref_system_constants, L1params)
                   for t in t_vals]
    plot!(p, t_vals, rhoHat_vals; linewidth=3, linestyle=:dash,
          color=:black, label=L"\hat{\rho}")

    # horizontal ρ line with numeric value in legend
    rho_line = rho_r + rho_a
    hline!(p, [rho_line]; linewidth=3, linestyle=:dash, color=colors[2],
           label=latexstring("ρ = $(ρ)"),yguidefontsize = 18, xguidefontsize = 18, legendfontsize = 18,
        tickfontsize = 16, gridlinewidth = 2.5, dpi = 300, margin = 5mm)

    savefig(p, savepath)
    return p
end


