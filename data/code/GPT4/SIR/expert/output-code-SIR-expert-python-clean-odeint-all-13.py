import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import json

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

N = 1000  # total population
I0, R0 = 1, 0  # initial number of infected and recovered individuals
S0 = N - I0 - R0  # everyone else is susceptible to infection initially
beta, gamma = 0.2, 1./10  # contact rate, mean recovery rate
t = np.linspace(0, 160, 160)  # grid of time points (in days)

# initial conditions vector
y0 = S0, I0, R0

# integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number (1000s)')
ax.legend()
plt.show()
