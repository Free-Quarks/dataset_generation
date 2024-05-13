import numpy as np



def sidarthe_model(params, y0, t):
    def deriv(y, t, beta, epsilon, gamma1, gamma2, gamma3, delta, alpha, rho):
        S, I, D, A, R, T, H = y
        N = S + I + D + A + R + T + H
        dsdt = -beta * S * (I + epsilon * A) / N
        didt = beta * S * (I + epsilon * A) / N - (gamma1 + gamma2 + gamma3 + delta) * I
        dddt = gamma1 * I
        dadt = gamma2 * I
        drdt = gamma3 * I
        dtdt = delta * I
        dhdt = alpha * (I + epsilon * A)
        return dsdt, didt, dddt, dadt, drdt, dtdt, dhdt
    
    beta, epsilon, gamma1, gamma2, gamma3, delta, alpha, rho = params
    
    sol = odeint(deriv, y0, t, args=(beta, epsilon, gamma1, gamma2, gamma3, delta, alpha, rho))
    
    return sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4], sol[:, 5], sol[:, 6]
