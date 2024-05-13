import numpy as np
import matplotlib.pyplot as plt


def simulate_serid(N, beta, gamma, delta, alpha, T):
    S = np.zeros(T)
    E = np.zeros(T)
    R = np.zeros(T)
    I = np.zeros(T)
    D = np.zeros(T)

    S[0] = N
    E[0] = 0
    R[0] = 0
    I[0] = 1
    D[0] = 0

    dt = 0.1

    for t in range(T-1):
        dS = -beta * S[t] * I[t] / N
        dE = beta * S[t] * I[t] / N - delta * E[t]
        dR = gamma * I[t]
        dI = delta * E[t] - (alpha + gamma) * I[t]
        dD = alpha * I[t]

        S[t+1] = S[t] + dS * dt
        E[t+1] = E[t] + dE * dt
        R[t+1] = R[t] + dR * dt
        I[t+1] = I[t] + dI * dt
        D[t+1] = D[t] + dD * dt

    return S, E, R, I, D


N = 1000000  # total population
beta = 0.4    # effective contact rate
gamma = 0.1   # recovery rate
alpha = 0.01  # mortality rate
T = 1000     # number of time steps

S, E, R, I, D = simulate_serid(N, beta, gamma, alpha, T)

plt.plot(range(T), S, label='Susceptible')
plt.plot(range(T), E, label='Exposed')
plt.plot(range(T), R, label='Recovered')
plt.plot(range(T), I, label='Infected')
plt.plot(range(T), D, label='Deaths')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SERID Model Simulation')
plt.show()
