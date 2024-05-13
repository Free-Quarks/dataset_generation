import numpy as np
from scipy.integrate import odeint


def serid_model(y, t, beta, gamma, sigma):
    S = y[0]
    E = y[1]
    I = y[2]
    R = y[3]
    D = y[4]
    N = S + E + I + R + D
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    dDdt = 0
    return [dSdt, dEdt, dIdt, dRdt, dDdt]


def simulate_serid_model(initial_conditions, parameters, t):
    y0 = initial_conditions
    beta, gamma, sigma = parameters
    solution = odeint(serid_model, y0, t, args=(beta, gamma, sigma))
    return solution

