import numpy as np
from scipy.integrate import odeint


def seir_model(y, t, beta, sigma, gamma):
    S, E, I, R = y
    N = S + E + I + R
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


def simulate_seir(beta, sigma, gamma, S0, E0, I0, R0, t): 
    y0 = S0, E0, I0, R0
    params = beta, sigma, gamma
    sol = odeint(seir_model, y0, t, args=params)
    return sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]
