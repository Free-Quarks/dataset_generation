import numpy as np
import matplotlib.pyplot as plt

def SIR_model(N, beta, gamma, I0, t_max):
    # Initialize arrays
    S = np.zeros(t_max)
    I = np.zeros(t_max)
    R = np.zeros(t_max)
    
    # Set initial conditions
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    
    # Euler method
    dt = 0.1
    for t in range(t_max-1):
        S[t+1] = S[t] - dt * beta * S[t] * I[t] / N
        I[t+1] = I[t] + dt * (beta * S[t] * I[t] / N - gamma * I[t])
        R[t+1] = R[t] + dt * gamma * I[t]
    
    # Return arrays
    return S, I, R

# Set parameters
N = 1000  # Total population
beta = 0.3  # Infection rate
gamma = 0.1  # Recovery rate
I0 = 1  # Initial infected individuals
t_max = 100  # Maximum time

# Run SIR model
S, I, R = SIR_model(N, beta, gamma, I0, t_max)

# Plot results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.legend()
plt.show()
