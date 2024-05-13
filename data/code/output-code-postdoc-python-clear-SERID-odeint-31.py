import numpy as np
from scipy.integrate import odeint


def serid_model(y, t, beta, gamma): 
    S, E, I, R, D = y
    N = S + E + I + R + D
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - gamma * E
    dIdt = gamma * E - D * I
    dRdt = (1 - D) * gamma * I
    dDdt = D * gamma * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]


def simulate_serid_model(N, I0, R0, D0, beta, gamma, duration):
    E0 = I0
    S0 = N - I0 - R0 - D0
    y0 = [S0, E0, I0, R0, D0]
    t = np.linspace(0, duration, num=duration+1)
    result = odeint(serid_model, y0, t, args=(beta, gamma))
    return result[:, 0], result[:, 1], result[:, 2], result[:, 3], result[:, 4]
