import numpy as np


def sidarthe(y, t, R0, alpha, beta, gamma, delta, theta, sigma):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    dSdt = -alpha * R0 * S * (I + delta * A) / N
    dIdt = alpha * R0 * S * (I + delta * A) / N - (beta + gamma + theta) * I
    dDdt = theta * I
    dAdt = beta * I - (gamma + delta) * A
    dRdt = gamma * (I + delta * A)
    dTdt = sigma * theta * I
    dHdt = sigma * (beta * I + gamma * delta * A)
    dEdt = alpha * R0 * S * (I + delta * A) / N
    return dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt


# Test
y0 = [9999, 1, 0, 0, 0, 0, 0, 0]

R0 = 2.5
alpha = 0.2
beta = 0.5
gamma = 0.1
delta = 0.05
theta = 0.02
sigma = 0.01

T = np.arange(0, 100, 0.1)

from scipy.integrate import odeint

sol = odeint(sidarthe, y0, T, args=(R0, alpha, beta, gamma, delta, theta, sigma))

import matplotlib.pyplot as plt

plt.plot(T, sol[:, 0], label='S')
plt.plot(T, sol[:, 1], label='I')
plt.plot(T, sol[:, 2], label='D')
plt.plot(T, sol[:, 3], label='A')
plt.plot(T, sol[:, 4], label='R')
plt.plot(T, sol[:, 5], label='T')
plt.plot(T, sol[:, 6], label='H')
plt.plot(T, sol[:, 7], label='E')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIDARTHE Model')
plt.show()
