import numpy as np
import matplotlib.pyplot as plt

def serid_rk2(beta, gamma, k, N, I0, E0, R0, D0, T):
    # Initial conditions
    S0 = N - I0 - E0 - R0 - D0
    y0 = [S0, E0, I0, R0, D0]
    
    # Parameters
    h = T[1] - T[0]
    
    # Empty arrays to store the simulation results
    S = np.zeros(len(T))
    E = np.zeros(len(T))
    I = np.zeros(len(T))
    R = np.zeros(len(T))
    D = np.zeros(len(T))
    
    # Assign initial conditions
    S[0] = y0[0]
    E[0] = y0[1]
    I[0] = y0[2]
    R[0] = y0[3]
    D[0] = y0[4]
    
    for i in range(1, len(T)):
        # Compute derivatives
        dSdt = -beta * S[i-1] * I[i-1] / N
        dEdt = beta * S[i-1] * I[i-1] / N - k * E[i-1]
        dIdt = k * E[i-1] - gamma * I[i-1]
        dRdt = gamma * I[i-1]
        dDdt = 0
        
        # Compute intermediate values
        S_half = S[i-1] + h/2 * dSdt
        E_half = E[i-1] + h/2 * dEdt
        I_half = I[i-1] + h/2 * dIdt
        R_half = R[i-1] + h/2 * dRdt
        D_half = D[i-1] + h/2 * dDdt
        
        # Compute next values
        S[i] = S[i-1] + h * dSdt
        E[i] = E[i-1] + h * dEdt
        I[i] = I[i-1] + h * dIdt
        R[i] = R[i-1] + h * dRdt
        D[i] = D[i-1] + h * dDdt

    # Return the simulation results
    return S, E, I, R, D
}

