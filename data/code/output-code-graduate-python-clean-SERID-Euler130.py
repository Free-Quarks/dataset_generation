import numpy as np
import matplotlib.pyplot as plt

def SERID(beta, gamma, N, I0, R0, t_max):
    # Set up arrays to store the values
    t = np.linspace(0, t_max, t_max+1)
    S = np.zeros(t_max+1)
    E = np.zeros(t_max+1)
    I = np.zeros(t_max+1)
    R = np.zeros(t_max+1)
    D = np.zeros(t_max+1)
    
    # Initialize the values
    S[0] = N - I0 - R0
    E[0] = 0
    I[0] = I0
    R[0] = R0
    D[0] = 0
    
    # Euler method to solve the differential equations
    for i in range(t_max):
        dS = -beta * S[i] * I[i] / N
        dE = beta * S[i] * I[i] / N - gamma * E[i]
        dI = gamma * E[i] - gamma * I[i]
        dR = gamma * I[i]
        dD = 0
        
        S[i+1] = S[i] + dS
        E[i+1] = E[i] + dE
        I[i+1] = I[i] + dI
        R[i+1] = R[i] + dR
        D[i+1] = D[i] + dD
        
    return t, S, E, I, R, D


# Example usage
beta = 0.3
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
t_max = 100

t, S, E, I, R, D = SERID(beta, gamma, N, I0, R0, t_max)

# Plotting
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.plot(t, D, label='Deaths')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
