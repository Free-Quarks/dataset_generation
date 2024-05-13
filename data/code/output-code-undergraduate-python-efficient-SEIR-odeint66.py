import numpy as np
from scipy.integrate import odeint


# The SEIR model differential equations.
def seir(y, t, N, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


# Function to run the SEIR model and return the results.
def run_SEIR_model(N, beta, gamma, sigma, E0, I0, R0, T):
    # Initial conditions vector
    y0 = N - E0 - I0 - R0
    # Time vector
    t = np.linspace(0, T, T)
    # Integrate the SEIR equations over the time grid.
    res = odeint(seir, (y0, E0, I0, R0), t, args=(N, beta, gamma, sigma))
    S, E, I, R = res.T
    return S, E, I, R

