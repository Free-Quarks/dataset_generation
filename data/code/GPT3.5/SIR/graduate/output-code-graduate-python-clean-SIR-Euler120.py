import numpy as np
import matplotlib.pyplot as plt

def SIR_Euler(beta, gamma, S0, I0, R0, N, T):
    # Set up arrays to store values
    S = np.zeros(T+1)
    I = np.zeros(T+1)
    R = np.zeros(T+1)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    dt = 0.1
    # Euler method to solve the differential equations
    for t in range(T):
        dS_dt = -beta * S[t] * I[t] / N
        dI_dt = beta * S[t] * I[t] / N - gamma * I[t]
        dR_dt = gamma * I[t]
        S[t+1] = S[t] + dt * dS_dt
        I[t+1] = I[t] + dt * dI_dt
        R[t+1] = R[t] + dt * dR_dt
    return S, I, R

# Parameters
beta = 0.25
gamma = 0.05
S0 = 999
I0 = 1
R0 = 0
N = 1000
T = 100

# Run the simulation
S, I, R = SIR_Euler(beta, gamma, S0, I0, R0, N, T)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model Simulation')
plt.legend()
plt.show()
