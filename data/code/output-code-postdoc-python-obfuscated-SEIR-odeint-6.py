import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that returns the derivative

def seir(y, t, N, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# Total population
N = 1000
# Initial conditions
E0, I0, R0 = 1, 0, 0
S0 = N - E0 - I0 - R0
# Parameters
beta, gamma, sigma = 0.2, 1.0, 1/5.2
# Time vector
t = np.linspace(0, 49, 50)
# Initial conditions vector
y0 = S0, E0, I0, R0
# Integrate the SEIR equations over the time grid
ret = odeint(seir, y0, t, args=(N, beta, gamma, sigma))
S, E, I, R = ret.T
# Plot the data
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, E/1000, 'y', alpha=0.5, lw=2, label='Exposed')
ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Population (thousands)')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
plt.show()
