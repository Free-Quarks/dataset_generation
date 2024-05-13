import numpy as np
import matplotlib.pyplot as plt


def serid_rk3(beta, gamma, N, I0, R0, t_max, dt):
    S0 = N - I0 - R0
    t = np.arange(0, t_max, dt)
    S = np.zeros_like(t)
    E = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        k1 = -beta * S[i-1] * I[i-1] / N
        k2 = -beta * (S[i-1] + 0.5 * dt * k1) * (I[i-1] + 0.5 * dt * k1) / N
        k3 = -beta * (S[i-1] - dt * k1 + 2 * dt * k2) * (I[i-1] - dt * k1 + 2 * dt * k2) / N
        S[i] = S[i-1] + dt * (k1 + 4 * k2 + k3) / 6

        k1 = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        k2 = beta * (S[i-1] + 0.5 * dt * k1) * (I[i-1] + 0.5 * dt * k1) / N - gamma * (I[i-1] + 0.5 * dt * k1)
        k3 = beta * (S[i-1] - dt * k1 + 2 * dt * k2) * (I[i-1] - dt * k1 + 2 * dt * k2) / N - gamma * (I[i-1] - dt * k1 + 2 * dt * k2)
        I[i] = I[i-1] + dt * (k1 + 4 * k2 + k3) / 6

        k1 = gamma * I[i-1]
        k2 = gamma * (I[i-1] + 0.5 * dt * k1)
        k3 = gamma * (I[i-1] - dt * k1 + 2 * dt * k2)
        R[i] = R[i-1] + dt * (k1 + 4 * k2 + k3) / 6

    return S, I, R


beta = 0.3
gamma = 0.1
N = 1000
I0 = 10
R0 = 0
t_max = 100
dt = 0.1

S, I, R = serid_rk3(beta, gamma, N, I0, R0, t_max, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time step')
plt.ylabel('Number of individuals')
plt.title('SERID Model Simulation')
plt.legend()
plt.show()
