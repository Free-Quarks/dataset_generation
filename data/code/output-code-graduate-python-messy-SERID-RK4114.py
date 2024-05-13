import numpy as np


def serid_model(beta, gamma, I0, N, T):
    # Step size
    dt = T[1] - T[0]
    
    # Empty arrays to store the results
    S = np.zeros(len(T))
    E = np.zeros(len(T))
    I = np.zeros(len(T))
    R = np.zeros(len(T))
    D = np.zeros(len(T))
    
    # Initial conditions
    S[0] = N - I0
    E[0] = 0
    I[0] = I0
    R[0] = 0
    
    # Runge-Kutta 4th order method
    for i in range(len(T) - 1):
        dSdt = -beta * S[i] * I[i] / N
        dEdt = beta * S[i] * I[i] / N - gamma * E[i]
        dIdt = gamma * E[i] - I[i]
        dRdt = gamma * I[i]
        dDdt = 0
        
        S[i+1] = S[i] + dt * (dSdt)
        E[i+1] = E[i] + dt * (dEdt)
        I[i+1] = I[i] + dt * (dIdt)
        R[i+1] = R[i] + dt * (dRdt)
        D[i+1] = D[i] + dt * (dDdt)
    
    return S, E, I, R, D
}

