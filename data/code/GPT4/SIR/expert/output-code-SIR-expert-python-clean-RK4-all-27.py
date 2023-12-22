import numpy as np
import matplotlib.pyplot as plt
import json

# SIR model dynamics
def sir_model(t, y, beta, gamma):
    S, I, R = y
    return [-beta*S*I, beta*S*I-gamma*I, gamma*I]

# RK4 method
def rk4(t, y, h, f, beta, gamma):
    k1 = h * np.array(f(t, y, beta, gamma))
    k2 = h * np.array(f(t + h / 2, y + k1 / 2, beta, gamma))
    k3 = h * np.array(f(t + h / 2, y + k2 / 2, beta, gamma))
    k4 = h * np.array(f(t + h, y + k3, beta, gamma))
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# parameters
beta = 0.5
gamma = 0.1
S0 = 0.99
I0 = 0.01
R0 = 0.0
T = 100
h = 0.1

# initial condition
y = [S0, I0, R0]

# time grid
t = np.arange(0, T+h, h)

# solution array
Y = np.zeros((len(t), len(y)))

Y[0, :] = y

# solve using RK4
for i in range(1, len(t)):
    Y[i, :] = rk4(t[i-1], Y[i-1, :], h, sir_model, beta, gamma)

# plot
plt.figure(figsize=(8, 5))
plt.plot(t, Y[:, 0], label='S(t)')
plt.plot(t, Y[:, 1], label='I(t)')
plt.plot(t, Y[:, 2], label='R(t)')
plt.legend()
plt.show()

