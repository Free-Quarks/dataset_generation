import numpy as np


def sir_model(beta, gamma, initial_conditions, timesteps):
    S_0, I_0, R_0 = initial_conditions
    N = S_0 + I_0 + R_0
    dt = 1
    S = np.zeros(timesteps)
    I = np.zeros(timesteps)
    R = np.zeros(timesteps)
    S[0] = S_0
    I[0] = I_0
    R[0] = R_0
    
    for t in range(1, timesteps):
        dS_dt = -beta * S[t-1] * I[t-1] / N
        dI_dt = (beta * S[t-1] * I[t-1] / N) - gamma * I[t-1]
        dR_dt = gamma * I[t-1]
        
        S[t] = S[t-1] + dt * dS_dt
        I[t] = I[t-1] + dt * dI_dt
        R[t] = R[t-1] + dt * dR_dt
    
    return S, I, R


# Example usage
beta = 0.2
gamma = 0.1
initial_conditions = (1000, 1, 0)
timesteps = 100

S, I, R = sir_model(beta, gamma, initial_conditions, timesteps)

print('Susceptible:', S)
print('Infected:', I)
print('Recovered:', R)
