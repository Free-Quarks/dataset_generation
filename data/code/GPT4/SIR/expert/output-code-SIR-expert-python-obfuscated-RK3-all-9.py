import matplotlib.pyplot as plt
import numpy as np
import json

def _rk3_(x, y, h, f):
    k1 = h * f(x, y)
    k2 = h * f(x + h / 2, y + k1 / 2)
    k3 = h * f(x + h, y - k1 + 2 * k2)
    return y + (k1 + 4 * k2 + k3) / 6


def _sir_(X, Y, h, beta, gamma):
    S, I, R = Y
    dS = -beta * S * I
    dI = beta * S * I - gamma * I
    dR = gamma * I
    return np.array([_rk3_(X, S, h, lambda x, y: dS), _rk3_(X, I, h, lambda x, y: dI), _rk3_(X, R, h, lambda x, y: dR)])

def _sim_sir_(S0, I0, R0, h, T, beta, gamma):
    X, Y = [0], [[S0, I0, R0]]
    for _ in np.arange(h, T + h, h):
        X.append(_)
        Y.append(_sir_(X[-1], Y[-1], h, beta, gamma))
    return X, np.array(Y)

def _plot_sir_(T, Y):
    plt.figure()
    plt.plot(T, Y[:, 0], label='S')
    plt.plot(T, Y[:, 1], label='I')
    plt.plot(T, Y[:, 2], label='R')
    plt.legend()
    plt.show()


_sir_values = _sim_sir_(990, 10, 0, 0.1, 100, 0.001, 0.1)
_plot_sir_(_sir_values[0], _sir_values[1])
