import numpy as np
import matplotlib.pyplot as plt

def seirid_model(N, I0, R0, beta, gamma, delta, sigma, t_max, dt):
    # Initialize arrays
    S = np.zeros(t_max)
    E = np.zeros(t_max)
    I = np.zeros(t_max)
    R = np.zeros(t_max)
    D = np.zeros(t_max)
    S[0] = N - I0
    E[0] = I0
    I[0] = I0
    R[0] = R0
    D[0] = 0

    # Euler's method
    for t in range(1, t_max):
        S[t] = S[t-1] - beta * S[t-1] * I[t-1] / N * dt
        E[t] = E[t-1] + (beta * S[t-1] * I[t-1] / N - sigma * E[t-1]) * dt
        I[t] = I[t-1] + (sigma * E[t-1] - gamma * I[t-1] - delta * I[t-1]) * dt
        R[t] = R[t-1] + gamma * I[t-1] * dt
        D[t] = D[t-1] + delta * I[t-1] * dt

    return S, E, I, R, D


# Example usage
N = 1000  # Total population
I0 = 1  # Initial number of infected individuals
R0 = 0  # Initial number of recovered individuals
beta = 0.2  # Contact rate
gamma = 0.1  # Recovery rate
delta = 0.01  # Death rate
sigma = 0.1  # Exposed rate
t_max = 100  # Number of time steps
dt = 0.1  # Time step size

S, E, I, R, D = seirid_model(N, I0, R0, beta, gamma, delta, sigma, t_max, dt)

# Plotting
plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.plot(D, label='Deaths')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.legend()
plt.title('SEIRID Model')
plt.show()
