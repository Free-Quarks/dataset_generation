import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(S0, I0, R0, beta, gamma, t_end, n_steps):
    # Initialize arrays to store the results
    S = np.zeros(n_steps)
    I = np.zeros(n_steps)
    R = np.zeros(n_steps)
    t = np.linspace(0, t_end, n_steps)
    
    # Set initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Calculate the time step
    dt = t[1] - t[0]
    
    # Run the simulation
    for i in range(1, n_steps):
        # Calculate the derivatives
        dS = -beta * S[i-1] * I[i-1]
        dI = beta * S[i-1] * I[i-1] - gamma * I[i-1]
        dR = gamma * I[i-1]
        
        # Update the populations using RK2 method
        S[i] = S[i-1] + dt * dS
        I[i] = I[i-1] + dt * dI
        R[i] = R[i-1] + dt * dR
        
    return S, I, R

# Example usage
S0 = 999
I0 = 1
R0 = 0
beta = 0.3
gamma = 0.1
t_end = 50
n_steps = 500

S, I, R = SIR_RK2(S0, I0, R0, beta, gamma, t_end, n_steps)

# Plotting the results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK2')
plt.legend()
plt.show()
