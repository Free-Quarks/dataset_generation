import numpy as np


def SEIR_model(t, y, beta, sigma, gamma):
    S, E, I, R = y
    N = S + E + I + R
    dSdt = -beta(t) * S * I / N
    dEdt = beta(t) * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]


def Euler_integration(t0, t_end, y0, beta, sigma, gamma, h):
    t = np.arange(t0, t_end + h, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i-1] + h * np.array(SEIR_model(t[i-1], y[i-1], beta, sigma, gamma))
    return t, y
