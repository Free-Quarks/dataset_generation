import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(N, beta, gamma, I0, R0, t_end, dt):
    def f(y, t):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    y0 = N - I0 - R0
    t = np.arange(0, t_end, dt)

    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)

    S[0] = y0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        k1 = f([S[i - 1], I[i - 1], R[i - 1]], t[i - 1])
        k2 = f([S[i - 1] + 0.5 * dt * k1[0], I[i - 1] + 0.5 * dt * k1[1], R[i - 1] + 0.5 * dt * k1[2]], t[i - 1] + 0.5 * dt)
        k3 = f([S[i - 1] - dt * k1[0] + 2 * dt * k2[0], I[i - 1] - dt * k1[1] + 2 * dt * k2[1], R[i - 1] - dt * k1[2] + 2 * dt * k2[2]], t[i - 1] + dt)
        S[i] = S[i - 1] + dt / 6 * (k1[0] + 4 * k2[0] + k3[0])
        I[i] = I[i - 1] + dt / 6 * (k1[1] + 4 * k2[1] + k3[1])
        R[i] = R[i - 1] + dt / 6 * (k1[2] + 4 * k2[2] + k3[2])

    return S, I, R


N = 1000
beta = 0.2
gamma = 0.1
I0 = 1
R0 = 0

t_end = 100
dt = 0.1

S, I, R = SIR_RK3(N, beta, gamma, I0, R0, t_end, dt)

t = np.arange(0, t_end, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model Using RK3')
plt.legend()
plt.show()
