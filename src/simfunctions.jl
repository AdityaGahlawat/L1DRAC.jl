
# function systems()
function _nominal_drift!(dX, X, params, t)
    dX[1:n] = f(t,X)[1:n]
end
function _nominal_diffusion!(dX, X, params, t)
    for i in 1:n
        for j in 1:d
            dX[i,j] = p(t,X)[i,j]
        end
    end
end

