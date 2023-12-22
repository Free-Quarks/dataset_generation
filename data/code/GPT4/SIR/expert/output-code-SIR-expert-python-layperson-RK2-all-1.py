
import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def rk2_sir(y, t, dt, N, beta, gamma):
    k1 = np.asarray(sir_model(y, t, N, beta, gamma)) * dt
    k2 = np.asarray(sir_model(y + k1*0.5, t + dt*0.5, N, beta, gamma)) * dt
    y_new = y + k2
    return y_new

def simulate():
    N = 1000
    beta = 0.2  
    gamma = 0.1
    S0, I0, R0 = 999, 1, 0  

    t = np.linspace(0, 160, 160)
    dt = t[1] - t[0]
    Y = np.empty((3, len(t)))
    Y[:, 0] = S0, I0, R0

    for i in range(1, len(t)):
        Y[:, i] = rk2_sir(Y[:, i-1], t[i-1], dt, N, beta, gamma)

    S, I, R = Y

    plt.figure(figsize=[6,4])
    plt.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
    plt.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
    plt.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered')
    plt.xlabel('Time /days')
    plt.ylabel('Number')
    plt.legend()
    plt.show()

simulate()

