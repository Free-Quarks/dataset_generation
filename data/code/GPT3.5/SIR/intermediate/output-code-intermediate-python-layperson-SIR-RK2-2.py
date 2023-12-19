import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(N, beta, gamma, S0, I0, R0, tmax, dt):
    # Initialize arrays to store the solution
    t = np.arange(0, tmax, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    
    # Set initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Define RK2 method
    for i in range(len(t) - 1):
        k1_S = -beta * S[i] * I[i] / N
        k1_I = beta * S[i] * I[i] / N - gamma * I[i]
        k2_S = -beta * (S[i] + dt * k1_S) * (I[i] + dt * k1_I) / N
        k2_I = beta * (S[i] + dt * k1_S) * (I[i] + dt * k1_I) / N - gamma * (I[i] + dt * k1_I)
        S[i+1] = S[i] + dt * (k1_S + k2_S) / 2
        I[i+1] = I[i] + dt * (k1_I + k2_I) / 2
        R[i+1] = R[i] + dt * gamma * (I[i] + dt * k1_I)
    
    return t, S, I, R

# Set parameters
N = 100000    # Total population
beta = 0.2   # Infection rate
gamma = 0.1  # Recovery rate
S0 = N - 1   # Initial susceptible
I0 = 1       # Initial infected
R0 = 0       # Initial recovered

# Simulation
t, S, I, R = SIR_RK2(N, beta, gamma, S0, I0, R0, 100, 0.1)

# Plotting
plt.figure()
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
