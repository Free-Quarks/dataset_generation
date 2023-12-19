import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(beta, gamma, S0, I0, R0, T, N):
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    t = np.linspace(0, T, T)
    y0 = S0, I0, R0
    ret = np.empty((T, 3))
    ret[0] = y0
    for i in range(T-1):
        h = t[i+1] - t[i]
        k1 = deriv(ret[i], t[i], N, beta, gamma)
        k2 = deriv(ret[i] + 0.5 * h * k1, t[i] + 0.5 * h, N, beta, gamma)
        k3 = deriv(ret[i] + h * (2.0 * k2 - k1), t[i] + h, N, beta, gamma)
        ret[i+1] = ret[i] + (h / 6.0) * (k1 + 4 * k2 + k3)
    S, I, R = ret.T

    fig, ax = plt.subplots()
    ax.plot(t, S, label='Susceptible')
    ax.plot(t, I, label='Infected')
    ax.plot(t, R, label='Recovered')
    ax.set(xlabel='Time (days)', ylabel='Population', title='SIR Model using RK3')
    ax.legend()
    plt.show()


beta = 0.2
gamma = 0.1
S0 = 999
I0 = 1
R0 = 0
T = 100
N = S0 + I0 + R0

SIR_RK3(beta, gamma, S0, I0, R0, T, N)
