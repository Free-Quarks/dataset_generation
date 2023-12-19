import numpy as np
import matplotlib.pyplot as plt

# Define the SIR model

def SIR_RK2(beta, gamma, S0, I0, R0, t_max, n_steps):
    
    # Create arrays to store the values of S, I, R
    S = np.zeros(n_steps)
    I = np.zeros(n_steps)
    R = np.zeros(n_steps)
    
    # Initialize the arrays with the initial values
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    dt = t_max / n_steps
    
    # Run the simulation
    for i in range(1, n_steps):
        
        # Calculate the derivatives
        dSdt = -beta * S[i-1] * I[i-1]
        dIdt = beta * S[i-1] * I[i-1] - gamma * I[i-1]
        dRdt = gamma * I[i-1]
        
        # Update the values of S, I, R using the RK2 method
        S_half = S[i-1] + 0.5 * dt * dSdt
        I_half = I[i-1] + 0.5 * dt * dIdt
        R_half = R[i-1] + 0.5 * dt * dRdt
        
        dSdt_half = -beta * S_half * I_half
        dIdt_half = beta * S_half * I_half - gamma * I_half
        dRdt_half = gamma * I_half
        
        S[i] = S[i-1] + dt * dSdt_half
        I[i] = I[i-1] + dt * dIdt_half
        R[i] = R[i-1] + dt * dRdt_half
    
    # Return the simulation results
    return S, I, R

# Set the simulation parameters
beta = 0.2
gamma = 0.1
S0 = 0.9
I0 = 0.1
R0 = 0.0
t_max = 100
n_steps = 1000

# Run the simulation
S, I, R = SIR_RK2(beta, gamma, S0, I0, R0, t_max, n_steps)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.legend()
plt.show()

