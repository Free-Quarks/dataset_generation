import numpy as np
from scipy.integrate import odeint


def serid_model(y, t, beta, gamma, mu):
    S, E, R, I, D = y
    N = S + E + R + I + D
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - gamma * E
    dRdt = gamma * E
    dIdt = (1 - mu) * gamma * E - mu * I
    dDdt = mu * I
    return [dSdt, dEdt, dRdt, dIdt, dDdt]


def solve_serid_model(S0, E0, R0, I0, D0, beta, gamma, mu, t):
    y0 = [S0, E0, R0, I0, D0]
    sol = odeint(serid_model, y0, t, args=(beta, gamma, mu))
    return sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]
