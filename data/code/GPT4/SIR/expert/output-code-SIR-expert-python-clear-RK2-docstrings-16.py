import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def rk2(y, t, dt, model, beta, gamma):
    k1 = model(y, t, beta, gamma)
    k2 = model([y[i] + k1[i] * dt for i in range(3)], t + dt, beta, gamma)
    return [y[i] + (k1[i] + k2[i]) * dt / 2 for i in range(3)]
