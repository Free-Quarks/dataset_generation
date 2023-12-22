import numpy as np
import matplotlib.pyplot as plt
import json

def simulate_SIR(beta, gamma, N, I0, R0, T):
    """
    Simulate the SIR model using the Euler method
    Args:
        beta : Contact rate
        gamma : Recovery rate
        N : Total population
        I0 : initial number of infected individuals
        R0 : initial number of recovered individuals
        T : Time duration
    Returns:
        S : Susceptible individuals over time
        I : Infected individuals over time
        R : Recovered individuals over time
    """
    S0 = N - I0 - R0  # initial number of susceptible individuals
    dt = 0.01  # time step
    n = int(T/dt)  # number of steps

    # Initialize S, I, R arrays
    S, I, R = np.zeros(n), np.zeros(n), np.zeros(n)
    S[0], I[0], R[0] = S0, I0, R0

    for t in range(n-1):
        S[t + 1] = S[t] - dt * beta * S[t] * I[t] / N
        I[t + 1] = I[t] + dt * (beta * S[t] * I[t] / N - gamma * I[t])
        R[t + 1] = R[t] + dt * gamma * I[t]

    return S, I, R

# Parameters
beta, gamma = 0.2, 0.1
N, I0, R0 = 1000, 1, 0
T = 160

# Simulate the SIR model
S, I, R = simulate_SIR(beta, gamma, N, I0, R0, T)

# Plotting
times = np.arange(0, T, 0.01)
plt.plot(times, S, label='Susceptible')
plt.plot(times, I, label='Infected')
plt.plot(times, R, label='Recovered')
plt.legend()
plt.show()
