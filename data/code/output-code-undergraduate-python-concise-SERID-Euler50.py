import numpy as np
import matplotlib.pyplot as plt

def serid_euler(S0, E0, I0, R0, beta, gamma, delta, mu, N, T):
    # Step size
    dt = 1
    # Number of time steps
    num_steps = int(T/dt)
    
    # Initialize arrays to store the values of each compartment
    S = np.zeros(num_steps)
    E = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    
    # Set initial values
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    
    # Perform Euler's method
    for t in range(1, num_steps):
        # Calculate derivatives
        dS = -beta * S[t-1] * I[t-1] / N
        dE = beta * S[t-1] * I[t-1] / N - delta * E[t-1] - mu * E[t-1]
        dI = delta * E[t-1] - gamma * I[t-1] - mu * I[t-1]
        dR = gamma * I[t-1] - mu * R[t-1]
        
        # Update compartments
        S[t] = S[t-1] + dt * dS
        E[t] = E[t-1] + dt * dE
        I[t] = I[t-1] + dt * dI
        R[t] = R[t-1] + dt * dR
    
    # Return arrays
    return S, E, I, R


# Example usage
S0 = 990
E0 = 10
I0 = 0
R0 = 0
beta = 0.3
gamma = 0.1
delta = 0.2
mu = 0.01
N = S0 + E0 + I0 + R0
T = 100

S, E, I, R = serid_euler(S0, E0, I0, R0, beta, gamma, delta, mu, N, T)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
