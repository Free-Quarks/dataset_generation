import numpy as np
import matplotlib.pyplot as plt


def serid_euler(y0, beta, gamma, alpha, delta, N, num_days):
    # Initialize arrays to store results
    S = np.zeros(num_days)
    E = np.zeros(num_days)
    I = np.zeros(num_days)
    R = np.zeros(num_days)
    D = np.zeros(num_days)
    t = np.arange(num_days)
    
    # Set initial conditions
    S[0] = y0[0]
    E[0] = y0[1]
    I[0] = y0[2]
    R[0] = y0[3]
    D[0] = y0[4]
    
    # Euler's method loop
    for i in range(1, num_days):
        # Compute derivatives
        dSdt = -beta * S[i-1] * I[i-1] / N
        dEdt = beta * S[i-1] * I[i-1] / N - alpha * E[i-1]
        dIdt = alpha * E[i-1] - gamma * I[i-1] - delta * I[i-1]
        dRdt = gamma * I[i-1]
        dDdt = delta * I[i-1]
        
        # Update values using Euler's method
        S[i] = S[i-1] + dSdt
        E[i] = E[i-1] + dEdt
        I[i] = I[i-1] + dIdt
        R[i] = R[i-1] + dRdt
        D[i] = D[i-1] + dDdt
        
    # Return results
    return S, E, I, R, D


# Example usage

# Set the initial conditions
initial_conditions = [999, 1, 0, 0, 0]

# Set the parameters
beta = 0.5
gamma = 0.1
alpha = 0.2
delta = 0.01
N = 1000
num_days = 100

# Simulate the model
S, E, I, R, D = serid_euler(initial_conditions, beta, gamma, alpha, delta, N, num_days)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.plot(D, label='Deceased')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SERID Model Simulation')
plt.legend()
plt.show()
