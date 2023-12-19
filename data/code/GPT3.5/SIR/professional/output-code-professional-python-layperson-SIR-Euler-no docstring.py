import numpy as np
import matplotlib.pyplot as plt


# Function to implement SIR model
def simulate_SIR(beta, gamma, S0, I0, R0, t_max, dt):
    
    # Initialize arrays
    t = np.arange(0, t_max+dt, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    
    # Set initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Euler's method to solve the differential equations
    for i in range(1, len(t)):
        S[i] = S[i-1] - beta * S[i-1] * I[i-1] * dt
        I[i] = I[i-1] + (beta * S[i-1] * I[i-1] - gamma * I[i-1]) * dt
        R[i] = R[i-1] + gamma * I[i-1] * dt
    
    # Return results
    return S, I, R


# Set parameters and initial conditions
beta = 0.2
gamma = 0.1
S0 = 999
I0 = 1
R0 = 0
t_max = 100
dt = 0.1

# Simulate the SIR model
S, I, R = simulate_SIR(beta, gamma, S0, I0, R0, t_max, dt)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SIR Model Simulation')
plt.legend()
plt.show()
