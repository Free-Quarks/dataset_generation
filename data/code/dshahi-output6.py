import numpy as np


def serid_rk3(S0, E0, R0, I0, D0, beta, gamma, delta, sigma, tau, N, t_end, dt):

    def serid_model(t, y):
        S, E, R, I, D = y
        dS = -beta * S * I / N
        dE = beta * S * I / N - delta * E
        dR = gamma * I
        dI = delta * E - gamma * I - sigma * I
        dD = sigma * I
        return [dS, dE, dR, dI, dD]


    t = np.arange(0, t_end, dt)

    y = np.zeros((len(t), 5))
    y[0] = [S0, E0, R0, I0, D0]

    for i in range(len(t) - 1):
        k1 = np.array(serid_model(t[i], y[i]))
        k2 = np.array(serid_model(t[i] + dt / 2, y[i] + dt / 2 * k1))
        k3 = np.array(serid_model(t[i] + dt, y[i] - dt * k1 + 2 * dt * k2))
        y[i+1] = y[i] + dt / 6 * (k1 + 4 * k2 + k3)

    return y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4]

