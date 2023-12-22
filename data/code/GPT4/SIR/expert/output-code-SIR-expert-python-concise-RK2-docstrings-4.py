import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def rk2(y, t, dt, derivs):
    k0 = dt*np.asarray(derivs(y, t))
    k1 = dt*np.asarray(derivs(y + k0, t + dt))
    y_next = y + 0.5*(k0 + k1)
    return y_next
