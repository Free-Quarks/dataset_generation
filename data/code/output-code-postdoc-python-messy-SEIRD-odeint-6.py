import numpy as np
from scipy.integrate import odeint


def seird_model(y, t, beta, gamma, delta, alpha):
    S, E, I, R, D = y
    N = S + E + I + R + D
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - delta * E
    dIdt = delta * E - (1 - alpha) * gamma * I - alpha * delta * I
    dRdt = (1 - alpha) * gamma * I
    dDdt = alpha * delta * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]


def simulate_seird_model(S0, E0, I0, R0, D0, beta, gamma, delta, alpha, t):
    y0 = [S0, E0, I0, R0, D0]
    sol = odeint(seird_model, y0, t, args=(beta, gamma, delta, alpha))
    S, E, I, R, D = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]
    return S, E, I, R, D
