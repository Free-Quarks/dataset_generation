import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model_rk2(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return np.array([dSdt, dIdt, dRdt])

def rk2_step(y, t, dt, derivs, beta, gamma):
    k1 = dt * derivs(y, t, beta, gamma)
    k2 = dt * derivs(y + 0.5 * k1, t + 0.5 * dt, beta, gamma)
    y_new = y + k2
    return y_new

def simulate_sir_rk2(S0, I0, R0, beta, gamma, dt, T):
    N = int(T/dt)
    t = np.linspace(0, T, N)
    y = np.zeros((N, 3))
    y[0, :] = [S0, I0, R0]
    for i in range(N-1):
        y[i+1, :] = rk2_step(y[i, :], t[i], dt, sir_model_rk2, beta, gamma)
    plt.figure()
    plt.plot(t, y[:, 0], label='Susceptible')
    plt.plot(t, y[:, 1], label='Infected')
    plt.plot(t, y[:, 2], label='Recovered')
    plt.legend()
    plt.show()
