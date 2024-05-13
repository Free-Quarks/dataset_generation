import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that returns the derivative of the system

def seir(y, t, N, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# Parameters
N = 1000
beta = 0.2
sigma = 1/5
gamma = 1/10

# Initial conditions
S0 = N - 1
E0 = 1
I0 = 0
R0 = 0

# Time vector
t = np.linspace(0, 160, 160)

# Initial conditions vector
y0 = S0, E0, I0, R0

# Integrate the SEIR equations over the time grid
result = odeint(seir, y0, t, args=(N, beta, sigma, gamma))
S, E, I, R = result.T

# Plot the data
fig, ax = plt.subplots()
ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
ax.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')
ax.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')
ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Population')
ax.set_ylim(0, N)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='gray', lw=0.5, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
plt.show()
