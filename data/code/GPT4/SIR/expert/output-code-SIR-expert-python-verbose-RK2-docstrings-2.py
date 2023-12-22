import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def rk2_sir(y, t, dt, N, beta, gamma):
    S, I, R = y
    dSdt1, dIdt1, dRdt1 = sir_model(y, t, N, beta, gamma)
    half_step = [S + 0.5*dt*dSdt1, I + 0.5*dt*dIdt1, R + 0.5*dt*dRdt1]
    dSdt2, dIdt2, dRdt2 = sir_model(half_step, t + 0.5*dt, N, beta, gamma)
    next_S = S + dt*dSdt2
    next_I = I + dt*dIdt2
    next_R = R + dt*dRdt2
    return next_S, next_I, next_R

N = 1000
beta = 0.2
gamma = 0.1
S0, I0, R0 = 999, 1, 0
y0 = S0, I0, R0
t = np.linspace(0, 160, 160)
dt = t[1] - t[0]
ret = np.empty((len(t), len(y0)))
ret[0] = y0

for i in range(1, len(t)):
    ret[i] = rk2_sir(ret[i-1], t[i-1], dt, N, beta, gamma)
