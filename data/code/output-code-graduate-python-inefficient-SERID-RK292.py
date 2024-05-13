from scipy.integrate import odeint
import numpy as np


def SERID(y, t, beta, gamma, delta, alpha):
    S, E, R, I, D = y
    N = S + E + R + I + D
    dS_dt = -beta * S * I / N
    dE_dt = beta * S * I / N - delta * E
    dR_dt = gamma * I
    dI_dt = delta * E - alpha * I
    dD_dt = alpha * I
    return [dS_dt, dE_dt, dR_dt, dI_dt, dD_dt]


def simulate_SERID(S, E, R, I, D, beta, gamma, delta, alpha, t):
    y0 = [S, E, R, I, D]
    sol = odeint(SERID, y0, t, args=(beta, gamma, delta, alpha))
    return sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]
