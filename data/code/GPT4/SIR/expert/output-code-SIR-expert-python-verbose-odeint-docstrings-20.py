from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

S0 = 0.9
I0 = 0.1
R0 = 0.0
beta = 2.2
gamma = 1.0

t = np.linspace(0, 20, 1000)

y0 = [S0, I0, R0]
ret = odeint(sir_model, y0, t, args=(beta, gamma))
S, I, R = ret.T

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number')
ax.legend()
plt.show()
