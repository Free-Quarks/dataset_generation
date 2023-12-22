import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def run_sir_model(S0, I0, R0, beta, gamma, t):
    y0 = S0, I0, R0
    dt = t[1] - t[0]
    S, I, R = y0
    for _ in t[1:]:
        dSdt, dIdt, dRdt = sir_model((S, I, R), _, beta, gamma)
        S = S + dSdt * dt
        I = I + dIdt * dt
        R = R + dRdt * dt
    return S, I, R

S0, I0, R0 = 1000, 1, 0
beta, gamma = 0.2, 1./10
t = np.linspace(0, 160, 160)
S, I, R = run_sir_model(S0, I0, R0, beta, gamma, t)

plt.figure(figsize=(6,4))
plt.plot(t, S, 'b', label='Susceptible');
plt.plot(t, I, 'r', label='Infected');
plt.plot(t, R, 'g', label='Recovered/deceased');
plt.legend(loc='best')
plt.xlabel('Time /days')
plt.ylabel('Number')
plt.grid()
plt.show()
