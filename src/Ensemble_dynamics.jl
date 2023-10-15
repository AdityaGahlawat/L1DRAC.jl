if myid() == 1
    println(">>> Loading dynamics to be simulated: USER DEFINED")
    @warn "---> Only the dynamics added to the named tuple:dynamics_tuple at the bottom of Ensemble_dynamics.jl will be simulated. \n         ---> Any edits to Ensemble_dynamics.jl will require the launch of a new Julia session"
end

## Van der Pol
    # initial condition distribution
    begin
        local μ₀ = [1.0f0; 1.0f0]
        local Σ₀ = [1.0f0 0.0f0; 0.0f0 1.0f0]
        VdP_ν₀ = init_Gaussian(μ₀, Σ₀)
    end
    # dynamics functions
    function VdP_f(X, p, t)
        local μ = 2
        dX1 = X[2]
        dX2 = (μ * X[2]) * (1 - X[1]^2) - X[1]
        return SVector{2}(dX1, dX2)
    end
    function VdP_p(X, p, t)
        dX11 = 0.1*(sin(X[1]^2) + cos(X[2]^2)) 
        dX12 = 0.1
        dX21 = 0.2
        dX22 = 0.1*sin(X[1]*X[2]^2)*cos(X[1]^2*X[2])
        return @SMatrix [dX11 dX12; dX21 dX22]
    end
    VdP = dynamics(VdP_f, VdP_p, VdP_ν₀)
##    


######### ADD DYNAMICS BELOW #########

##----------------------------------------------------------------
# Dynamics Named Tuple 
# e.g dynamics_tuple = (sys1_name = sys_1::dynamics, sys2_name = sys_2::dynamics, ..., sysn_name = sys_n::dynamics)
##----------------------------------------------------------------
dynamics_tuple = (VanderPol = VdP, )

