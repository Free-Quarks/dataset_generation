from matplotlib import pyplot as plt
import numpy as np
import json

def __sir__(y0, t, N, beta, gamma):
    S, I, R = y0
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return [dS, dI, dR]

def __rk3__(y0, t, dt, derivs, N, beta, gamma):
    k1 = derivs(y0, t, N, beta, gamma)
    k2 = derivs([y + ki * dt / 2 for y, ki in zip(y0, k1)], t + dt / 2, N, beta, gamma)
    k3 = derivs([y + ki * dt for y, ki in zip(y0, k2)], t + dt, N, beta, gamma)
    y = [y + (k1i + 4 * k2i + k3i) * dt / 6 for y, k1i, k2i, k3i in zip(y0, k1, k2, k3)]
    return y

N = 1000
beta = 0.2
gamma = 0.1
S0, I0, R0 = N-1, 1, 0
dt = 0.1
t = np.arange(0, 160, dt)

sir_sol = []
y0 = [S0, I0, R0]
for ti in t:
    y0 = __rk3__(y0, ti, dt, __sir__, N, beta, gamma)
    sir_sol.append(y0)

S, I, R = np.array(sir_sol).T

plt.figure(figsize=[6,4])
plt.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
plt.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
plt.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered')
plt.xlabel('Time /days')
plt.ylabel('Number')
plt.grid(True)
plt.legend()
plt.title('SIR model with RK3')
plt.show()
