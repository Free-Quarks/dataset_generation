import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def euler_method(func, y0, t, args):
    dt = t[1] - t[0]
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        y[i + 1] = y[i] + np.array(func(y[i], t, *args)) * dt
    return y

N = 1000
beta = 1.0
gamma = 0.1
S0, I0, R0 = N-1, 1, 0
t = np.linspace(0, 100, 1000)

solution = euler_method(sir_model, [S0, I0, R0], t, args=(beta, gamma))

plt.figure(figsize=(12, 8))
plt.plot(t, solution[:, 0], label='S(t)')
plt.plot(t, solution[:, 1], label='I(t)')
plt.plot(t, solution[:, 2], label='R(t)')
plt.legend()
plt.show()
