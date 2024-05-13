import numpy as np


def seirid_rk4(N, beta, gamma, delta, alpha, rho, mu, sigma, t_max, dt):
    
    def seirid_model(t, y):
        S, E, I, R, D = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - delta * E
        dIdt = delta * E - (alpha + gamma + mu) * I
        dRdt = gamma * I + rho * D
        dDdt = alpha * I + sigma * I
        return np.array([dSdt, dEdt, dIdt, dRdt, dDdt])

    t = np.arange(0, t_max, dt)
    y0 = np.array([N - 1, 1, 0, 0, 0])
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    
    for i in range(len(t) - 1):
        k1 = seirid_model(t[i], y[i])
        k2 = seirid_model(t[i] + dt/2, y[i] + dt/2 * k1)
        k3 = seirid_model(t[i] + dt/2, y[i] + dt/2 * k2)
        k4 = seirid_model(t[i] + dt, y[i] + dt * k3)
        y[i+1] = y[i] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    return t, y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4]

