import numpy as np
import matplotlib.pyplot as plt


def serid_model(beta, gamma, sigma, N, I0, E0, R0, D0, T):
    # Total population, N.
    # Initial number of infected and recovered individuals, I0 and R0.
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0
    # Initial number of exposed individuals, E0.
    # Initial number of deaths, D0.
    E0 = N - S0 - I0 - R0 - D0
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    # Rate at which an exposed individual becomes infectious, sigma (in 1/days)
    # A grid of time points (in days)
    t = np.linspace(0, T, num=T+1)
    # Everyone else, S0, is susceptible to infection initially.
    S = np.zeros_like(t)
    # Exposed individuals
    E = np.zeros_like(t)
    # Infected individuals
    I = np.zeros_like(t)
    # Recovered individuals
    R = np.zeros_like(t)
    # Deaths
    D = np.zeros_like(t)
    # Initial conditions
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    D[0] = D0
    # RK3 method to solve the differential equations
    for i in range(1, T+1):
        dt = t[i] - t[i-1]
        S[i] = S[i-1] - beta * S[i-1] * I[i-1] * dt
        E[i] = E[i-1] + (beta * S[i-1] * I[i-1] - sigma * E[i-1]) * dt
        I[i] = I[i-1] + (sigma * E[i-1] - gamma * I[i-1]) * dt
        R[i] = R[i-1] + gamma * I[i-1] * dt
        D[i] = D[i-1] + (gamma * I[i-1] * dt) * 0.01
    return S, E, I, R, D


# Parameters
beta = 0.2
gamma = 0.1
sigma = 0.05
N = 1000
I0 = 1
E0 = 0
R0 = 0
D0 = 0
T = 100


# Run the model
S, E, I, R, D = serid_model(beta, gamma, sigma, N, I0, E0, R0, D0, T)


# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.plot(D, label='Deaths')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SERID Model')
plt.legend()
plt.show()
