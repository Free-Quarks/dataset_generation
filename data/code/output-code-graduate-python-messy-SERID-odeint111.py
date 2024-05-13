import numpy as np
from scipy.integrate import odeint


def serid_model(y, t, N, beta, gamma, delta, alpha):
    S, E, I, R, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - delta * E
    dIdt = delta * E - gamma * I - alpha * I
    dRdt = gamma * I
    dDdt = alpha * I
    return dSdt, dEdt, dIdt, dRdt, dDdt


def simulate_serid_model(N, E0, I0, R0, D0, beta, gamma, delta, alpha, t):
    S0 = N - E0 - I0 - R0 - D0
    y0 = S0, E0, I0, R0, D0
    t = np.linspace(0, t, t)
    result = odeint(serid_model, y0, t, args=(N, beta, gamma, delta, alpha))
    return result[:, 0], result[:, 1], result[:, 2], result[:, 3], result[:, 4]
