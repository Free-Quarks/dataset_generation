import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(beta, gamma, S_0, I_0, R_0, T, N, h):
    def f(t, y):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return np.array([dSdt, dIdt, dRdt])

    t = np.arange(0, T+1, h)
    y = np.zeros((len(t), 3))
    y[0] = [S_0, I_0, R_0]

    for i in range(len(t) - 1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h/2 * k1)
        y[i+1] = y[i] + h * k2

    plt.plot(t, y[:, 0], label='S')
    plt.plot(t, y[:, 1], label='I')
    plt.plot(t, y[:, 2], label='R')
    plt.xlabel('Time')
    plt.ylabel('Number of Individuals')
    plt.legend()
    plt.show()

SIR_RK2(0.3, 0.1, 999, 1, 0, 100, 1000, 0.1)
