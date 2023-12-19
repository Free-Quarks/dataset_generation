import numpy as np
import matplotlib.pyplot as plt


# Define the SIR model

def sir_model(beta, gamma, N, I0, R0, t_end, dt):
    # Convert beta and gamma to per-day rates
    beta = beta / N
    gamma = gamma / N
    
    # Calculate the initial number of susceptibles and infected individuals
    S0 = N - I0 - R0
    
    # Create arrays to store the values of S, I, and R over time
    S = np.zeros(int(t_end / dt) + 1)
    I = np.zeros(int(t_end / dt) + 1)
    R = np.zeros(int(t_end / dt) + 1)
    
    # Set the initial values
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Run the simulation
    t = np.linspace(0, t_end, int(t_end / dt) + 1)
    for i in range(1, len(t)):
        # Compute the derivatives
        dSdt = -beta * S[i-1] * I[i-1]
        dIdt = beta * S[i-1] * I[i-1] - gamma * I[i-1]
        dRdt = gamma * I[i-1]
        
        # Update the values
        S[i] = S[i-1] + dt * dSdt
        I[i] = I[i-1] + dt * dIdt
        R[i] = R[i-1] + dt * dRdt
    
    # Return the simulation results
    return S, I, R


# Define the parameters
beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0

# Run the simulation
S, I, R = sir_model(beta, gamma, N, I0, R0, t_end=100, dt=0.1)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SIR Model Simulation')
plt.legend()
plt.show()
