import numpy as np
from scipy.integrate import odeint


def serid_model(y, t, beta, sigma, gamma, mu):
    S, E, I, R, D = y
    N = S + E + I + R + D
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - (1 - mu) * gamma * I - mu * gamma * I
    dRdt = (1 - mu) * gamma * I
    dDdt = mu * gamma * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]


def run_serid_model(S, E, I, R, D, beta, sigma, gamma, mu, t):
    y0 = [S, E, I, R, D]
    args = (beta, sigma, gamma, mu)
    output = odeint(serid_model, y0, t, args)
    return output[:, 0], output[:, 1], output[:, 2], output[:, 3], output[:, 4]
