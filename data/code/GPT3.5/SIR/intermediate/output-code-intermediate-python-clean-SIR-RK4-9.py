import numpy as np
from scipy.integrate import odeint


def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def simulate_SIR_model(S0, I0, R0, beta, gamma, t):
    y0 = [S0, I0, R0]
    result = odeint(SIR_model, y0, t, args=(beta, gamma))
    return result[:, 0], result[:, 1], result[:, 2]
