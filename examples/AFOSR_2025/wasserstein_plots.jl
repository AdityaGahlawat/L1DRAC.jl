using Plots
using LaTeXStrings
using JLD2
using Measures


function load_simulation_data(exp_name)

        ens_nom_sol_file =results_directory  * exp_name  * "ens_nom_sol.jld2"
        ens_tru_sol_file =results_directory  * exp_name  * "ens_tru_sol.jld2"
        ens_L1_sol_file  =results_directory   * exp_name   * "ens_L1_sol.jld2"

        ens_nom_sol= JLD2.load(ens_nom_sol_file)["ens_nom_sol"]
        ens_tru_sol = JLD2.load(ens_tru_sol_file)["ens_tru_sol"]
        ens_L1_sol = JLD2.load(ens_L1_sol_file)["ens_L1_sol"]
        
        return  ens_nom_sol, ens_tru_sol , ens_L1_sol
end

function plot_wasserstein(exp_name, tspan::Tuple{Float64,Float64}, rho_r::Float64, rho_a::Float64; dt=0.1, order_p=1)
    
    t_vals = collect(tspan[1]:dt:tspan[2])

    L2p_norm(x, p) = (mean(sum(abs.(x).^(2*p); dims=1)))^(1/2*p)

    L2_initial_error = L2p_norm(ens_nom_sol[:,1,:]-ens_L1_sol[1:2,1,:], order_p)

    # indices that correspond to those times
    # ensure same length as t_vals:
    idx_num = Int(dt*100)
    t_idxs = [1; idx_num:idx_num:idx_num*(length(t_vals)-1)]

    # W₂ between true system with l1 on and nominal systtem at those indices
    # w2_l1_vals= [begin
    #     empirical_samples= EmpiricalSamples(ens_nom_sol[1:n, i, :],ens_L1_sol[1:n, i, :])
    #     empirical_wasserstein2(empirical_samples, system_dimensions) 
    # end for i in t_idxs]

    wassd_l1_on_filename = results_directory * exp_name * "wassd_l1_on.jld2"
    w2_l1_vals= JLD2.load(wassd_l1_on_filename)["wassd_l1_on"]

    wassd_l1_off_filename = results_directory * exp_name * "wassd_l1_off.jld2"
    w2_tru_vals= JLD2.load(wassd_l1_off_filename)["wassd_l1_off"]

    # wassd_l1_on_filename = results_directory * exp_name * "wassd_l1_on.jld2"
    # jldsave(wassd_l1_on_filename; wassd_l1_on=w2_l1_vals)  

    # W₂ between true system with l1 off and nominal systtem at those indices
    # w2_tru_vals= [begin
    #     empirical_samples= EmpiricalSamples(ens_nom_sol[1:n, i, :],ens_tru_sol[1:n, i, :])
    #     empirical_wasserstein2(empirical_samples, system_dimensions) 
    # end for i in t_idxs]

    # wassd_l1_off_filename = results_directory * exp_name * "wassd_l1_off.jld2"
    # jldsave(wassd_l1_off_filename; wassd_l1_off=w2_tru_vals)  


    ω_rounded= round(ω, digits=2)
    Tₛ_rounded= round(Tₛ, digits=5)
    # Title — interpolate ω and Tₛ values
    p = plot(t_vals, w2_l1_vals;
        linewidth = 2.5,
        color = 1,
        label = L"\mathcal{L}_1 - on", 
        xlabel = L"t",
        ylabel = L"\mathsf{W}_{2}\!\left(\mathbb{X}_{t},\,\mathbb{X}^\star_{t}\right)",
        title  = latexstring("\\omega = $(ω), T_s= $(Tₛ_rounded)"),
        xlim = (tspan[1], tspan[2]),
        size = (550, 350)
    )

    plot!(t_vals, w2_tru_vals;
        linewidth = 2.5,
        color = 2,
        label = L"\mathcal{L}_1 - off ",  
        xlabel = L"t",
        ylabel = L"\mathsf{W}_{2}\!\left(\mathbb{X}_{t},\,\mathbb{X}^\star_{t}\right)",
        title  = latexstring("\\omega = $(ω_rounded), T_s= $(Tₛ_rounded)"),
        xlim = (tspan[1], tspan[2]),
        legend=:right
    )

    # ρ̂ curve at same times
    rhoHat_vals = [rhoHat(t, rho_r, rho_a, L2_initial_error,
                          assumption_constants, ref_system_constants, L1params)
                   for t in t_vals]
    plot!(p, t_vals, rhoHat_vals; linewidth = 2.5, linestyle=:dash, color=3, label=L"\hat{\rho}")

    # horizontal ρ line with numeric value in legend
    rho_line = rho_r + rho_a
    hline!(p, [rho_line]; linewidth = 2.5, linestyle=:dash, color=4,
           label=latexstring("ρ = $(ρ)"))

    savepath = wassd_filename

    savefig(p, savepath)
    return p
end


