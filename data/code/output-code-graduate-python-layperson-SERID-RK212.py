import numpy as np


def serid(beta, gamma, delta, N, I0, R0, D0, T):
    
    # Step size
    dt = T[1] - T[0]
    
    # Number of time steps
    num_steps = len(T)
    
    # Initialize arrays for S, E, R, I, and D
    S = np.zeros(num_steps)
    E = np.zeros(num_steps)
    R = np.zeros(num_steps)
    I = np.zeros(num_steps)
    D = np.zeros(num_steps)
    
    # Set initial values
    S[0] = N - I0 - R0 - D0
    E[0] = 0
    R[0] = R0
    I[0] = I0
    D[0] = D0
    
    # Iterate over time steps
    for t in range(num_steps - 1):
        
        # Compute the derivatives
        dSdt = -beta * S[t] * I[t] / N
        dEdt = beta * S[t] * I[t] / N - delta * E[t]
        dRdt = gamma * I[t]
        dIdt = delta * E[t] - gamma * I[t]
        dDdt = gamma * I[t]
        
        # Compute the next values using RK2 method
        S_half = S[t] + 0.5 * dt * dSdt
        E_half = E[t] + 0.5 * dt * dEdt
        R_half = R[t] + 0.5 * dt * dRdt
        I_half = I[t] + 0.5 * dt * dIdt
        D_half = D[t] + 0.5 * dt * dDdt
        
        dSdt_half = -beta * S_half * I_half / N
        dEdt_half = beta * S_half * I_half / N - delta * E_half
        dRdt_half = gamma * I_half
        dIdt_half = delta * E_half - gamma * I_half
        dDdt_half = gamma * I_half
        
        S[t+1] = S[t] + dt * dSdt_half
        E[t+1] = E[t] + dt * dEdt_half
        R[t+1] = R[t] + dt * dRdt_half
        I[t+1] = I[t] + dt * dIdt_half
        D[t+1] = D[t] + dt * dDdt_half
        
    return S, E, R, I, D


# Example usage:

# Parameters
beta = 0.2
gamma = 0.1
N = 1000
I0 = 10
R0 = 0
D0 = 0
T = np.linspace(0, 100, num=1000)

# Run the model
S, E, R, I, D = serid(beta, gamma, delta, N, I0, R0, D0, T)
