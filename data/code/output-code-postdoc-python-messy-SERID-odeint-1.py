import numpy as np
from scipy.integrate import odeint


def serid_model(y, t, N, beta, gamma):
    S, E, I, R, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - gamma * E
    dIdt = gamma * E - (1 - gamma) * I
    dRdt = (1 - gamma) * I
    dDdt = 0
    return dSdt, dEdt, dIdt, dRdt, dDdt



def run_serid_model(y0, N, beta, gamma, t_max):
    t = np.linspace(0, t_max, t_max + 1)
    result = odeint(serid_model, y0, t, args=(N, beta, gamma))
    return result

