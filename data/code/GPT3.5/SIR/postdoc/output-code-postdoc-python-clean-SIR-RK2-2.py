import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, N, I0, R0, t_end, dt):
    # Initialize arrays
    t = np.arange(0, t_end, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    
    # Set initial conditions
    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0
    
    # Perform RK2 integration
    for i in range(1, len(t)):
        # Calculate derivatives
        dSdt = -beta * S[i-1] * I[i-1] / N
        dIdt = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dRdt = gamma * I[i-1]
        
        # Calculate intermediate values
        S_half = S[i-1] + 0.5 * dt * dSdt
        I_half = I[i-1] + 0.5 * dt * dIdt
        R_half = R[i-1] + 0.5 * dt * dRdt
        
        dSdt_half = -beta * S_half * I_half / N
        dIdt_half = beta * S_half * I_half / N - gamma * I_half
        dRdt_half = gamma * I_half
        
        # Update values using RK2 method
        S[i] = S[i-1] + dt * dSdt_half
        I[i] = I[i-1] + dt * dIdt_half
        R[i] = R[i-1] + dt * dRdt_half
    
    return t, S, I, R

# Set parameters
beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
t_end = 50
dt = 0.1

# Run SIR model
t, S, I, R = SIR_model(beta, gamma, N, I0, R0, t_end, dt)

# Plot results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
