import numpy as np
from scipy.integrate import odeint


def serid_model(y, t, N, beta, gamma, mu):
    S, E, I, R, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - mu * E
    dIdt = mu * E - gamma * I
    dRdt = (1 - mu) * E
    dDdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt, dDdt


def run_serid_model(N, E0, I0, R0, D0, beta, gamma, mu, t_max):
    S0 = N - E0 - I0 - R0 - D0
    y0 = S0, E0, I0, R0, D0
    t = np.linspace(0, t_max, t_max+1)
    result = odeint(serid_model, y0, t, args=(N, beta, gamma, mu))
    return result[:, 0], result[:, 1], result[:, 2], result[:, 3], result[:, 4]
