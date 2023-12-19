import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(S0, I0, R0, beta, gamma, t_max, dt):
    N = S0 + I0 + R0
    t = np.arange(0, t_max, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        k1 = -beta * S[i-1] * I[i-1] / N
        k2 = -beta * (S[i-1] + k1 * dt/2) * (I[i-1] + k1 * dt/2) / N
        k3 = -beta * (S[i-1] - k1 * dt + 2 * k2 * dt) * (I[i-1] - k1 * dt + 2 * k2 * dt) / N
        S[i] = S[i-1] + (k1 + 4 * k2 + k3) * dt / 6

        k1 = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        k2 = beta * (S[i-1] + k1 * dt/2) * (I[i-1] + k1 * dt/2) / N - gamma * (I[i-1] + k1 * dt/2)
        k3 = beta * (S[i-1] - k1 * dt + 2 * k2 * dt) * (I[i-1] - k1 * dt + 2 * k2 * dt) / N - gamma * (I[i-1] - k1 * dt + 2 * k2 * dt)
        I[i] = I[i-1] + (k1 + 4 * k2 + k3) * dt / 6

        k1 = gamma * I[i-1]
        k2 = gamma * (I[i-1] + k1 * dt/2)
        k3 = gamma * (I[i-1] - k1 * dt + 2 * k2 * dt)
        R[i] = R[i-1] + (k1 + 4 * k2 + k3) * dt / 6

    return S, I, R


S0 = 990
I0 = 10
R0 = 0
beta = 0.3
gamma = 0.1
t_max = 100
dt = 0.1

S, I, R = SIR_RK3(S0, I0, R0, beta, gamma, t_max, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
