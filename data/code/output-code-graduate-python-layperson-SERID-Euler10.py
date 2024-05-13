import numpy as np
import matplotlib.pyplot as plt

def SERID(initial_conditions, parameters, timesteps):
    # Unpack initial conditions
    S_0, E_0, R_0, I_0, D_0 = initial_conditions
    
    # Unpack parameters
    beta, sigma, gamma, mu = parameters
    
    # Initialize arrays to store results
    S = np.zeros(timesteps)
    E = np.zeros(timesteps)
    R = np.zeros(timesteps)
    I = np.zeros(timesteps)
    D = np.zeros(timesteps)
    
    # Assign initial values
    S[0] = S_0
    E[0] = E_0
    R[0] = R_0
    I[0] = I_0
    D[0] = D_0
    
    # Euler method to simulate the model
    for t in range(1, timesteps):
        dS = -beta * S[t-1] * I[t-1] / (S[t-1] + E[t-1] + I[t-1] + R[t-1])
        dE = beta * S[t-1] * I[t-1] / (S[t-1] + E[t-1] + I[t-1] + R[t-1]) - sigma * E[t-1]
        dR = gamma * I[t-1]
        dI = sigma * E[t-1] - gamma * I[t-1] - mu * I[t-1]
        dD = mu * I[t-1]
        
        S[t] = S[t-1] + dS
        E[t] = E[t-1] + dE
        R[t] = R[t-1] + dR
        I[t] = I[t-1] + dI
        D[t] = D[t-1] + dD
    
    # Return the results
    return S, E, R, I, D

# Example usage
initial_conditions = (1000, 10, 0, 1, 0)
parameters = (0.3, 0.1, 0.2, 0.05)
timesteps = 100

S, E, R, I, D = SERID(initial_conditions, parameters, timesteps)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(R, label='Recovered')
plt.plot(I, label='Infected')
plt.plot(D, label='Deaths')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
