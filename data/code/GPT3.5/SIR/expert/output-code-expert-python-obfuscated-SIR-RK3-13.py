import numpy as np
import matplotlib.pyplot as plt


# Implementing the SIR model
def sir_model(beta, gamma, S0, I0, R0, N, t):
    # Step size
    dt = t[1] - t[0]
    
    # Number of time steps
    num_steps = len(t)
    
    # Initialize arrays to store the compartments
    S = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    
    # Set initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Runge-Kutta 3rd order
    for i in range(1, num_steps):
        # Calculate intermediate values
        k1_s = -beta * S[i - 1] * I[i - 1] / N
        k1_i = beta * S[i - 1] * I[i - 1] / N - gamma * I[i - 1]
        k2_s = -beta * (S[i - 1] + 0.5 * dt * k1_s) * (I[i - 1] + 0.5 * dt * k1_i) / N
        k2_i = beta * (S[i - 1] + 0.5 * dt * k1_s) * (I[i - 1] + 0.5 * dt * k1_i) / N - gamma * (I[i - 1] + 0.5 * dt * k1_i)
        k3_s = -beta * (S[i - 1] - dt * k1_s + 2 * dt * k2_s) * (I[i - 1] - dt * k1_i + 2 * dt * k2_i) / N
        k3_i = beta * (S[i - 1] - dt * k1_s + 2 * dt * k2_s) * (I[i - 1] - dt * k1_i + 2 * dt * k2_i) / N - gamma * (I[i - 1] - dt * k1_i + 2 * dt * k2_i)
        
        # Update compartments
        S[i] = S[i - 1] + dt * (k1_s + 4 * k2_s + k3_s) / 6
        I[i] = I[i - 1] + dt * (k1_i + 4 * k2_i + k3_i) / 6
        R[i] = N - S[i] - I[i]
    
    return S, I, R


# Set parameters
beta = 0.3
gamma = 0.1
S0 = 900
I0 = 100
R0 = 0
N = S0 + I0 + R0

# Set time steps
t = np.linspace(0, 100, 1001)

# Run the SIR model
S, I, R = sir_model(beta, gamma, S0, I0, R0, N, t)

# Plot the results
plt.figure()
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
