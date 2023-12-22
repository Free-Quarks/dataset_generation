import numpy as np
import matplotlib.pyplot as plt
import json

# Define some constants
beta, gamma = 0.1, 0.05
N = 1000
I_0, R_0 = 1, 0
S_0 = N - I_0 - R_0
dt = 0.1
t = np.linspace(0, 160, int(160/dt) + 1)

# Define the SIR model
def sir_model(S, I, R, beta, gamma, dt):
    dS = -beta * S * I * dt
    dI = (beta * S * I - gamma * I) * dt
    dR = gamma * I * dt
    return dS, dI, dR

# Define a function to simulate the SIR model
def simulate_sir_model(S_0, I_0, R_0, beta, gamma, dt, t):
    S, I, R = S_0, I_0, R_0
    S_list, I_list, R_list = [S_0], [I_0], [R_0]
    for _ in t[1:]:
        dS, dI, dR = sir_model(S, I, R, beta, gamma, dt)
        S += dS
        I += dI
        R += dR
        S_list.append(S)
        I_list.append(I)
        R_list.append(R)
    return np.array(S_list), np.array(I_list), np.array(R_list)

# Simulate the SIR model
S, I, R = simulate_sir_model(S_0, I_0, R_0, beta, gamma, dt, t)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SIR model simulation using Euler\'s method')
plt.grid(True)
plt.show()
