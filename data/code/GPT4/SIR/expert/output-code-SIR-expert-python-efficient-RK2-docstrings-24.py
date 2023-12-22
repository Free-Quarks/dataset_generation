import numpy as np
import matplotlib.pyplot as plt
import json

def rk2_SIR(y, t, dt, derivatives):
    k1 = dt * derivatives(y, t)
    k2 = dt * derivatives(y + 0.5 * k1, t + 0.5 * dt)
    y_next = y + k2
    return y_next

def SIR_derivatives(y, t):
    S, I, R = y
    dS_dt = -beta * S * I
    dI_dt = beta * S * I - gamma * I
    dR_dt = gamma * I
    return np.array([dS_dt, dI_dt, dR_dt])

beta, gamma = 0.2, 0.1
S0, I0, R0 = 1000, 1, 0
y0 = np.array([S0, I0, R0])
t_max = 365
dt = 0.1
ts = np.arange(0, t_max, dt)
ys = np.zeros((ts.size, y0.size))
ys[0, :] = y0
for i in range(1, ts.size):
    ys[i, :] = rk2_SIR(ys[i-1, :], ts[i-1], dt, SIR_derivatives)

plt.figure()
plt.plot(ts, ys)
plt.legend(['S', 'I', 'R'])
plt.show()
