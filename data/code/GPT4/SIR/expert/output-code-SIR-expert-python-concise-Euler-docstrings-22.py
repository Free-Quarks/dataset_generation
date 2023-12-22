import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model_euler(y,t,beta,gamma):
    """
    The SIR model differential equations.

    Parameters:
    y : tuple
        A tuple containing S(t), I(t), R(t)
    t : float
        Time
    beta : float
        The parameter controlling how often a susceptible-infected contact results in a new infection.
    gamma : float
        The rate an infected recovers and moves into the resistant phase. 

    Returns:
    dydt : tuple
        A tuple containing the derivatives (dS/dt, dI/dt, dR/dt)
    """
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions
S0, I0, R0 = 999, 1, 0  # initial conditions: one infected, rest susceptible
beta, gamma = 0.2, 1./10  # beta and gamma parameters
t = np.linspace(0, 160, 160) # Grid of time points (in days)

# Integrate the SIR equations over the time grid, t, using Euler method
y = [S0, I0, R0]
dt = t[1] - t[0]
for time in t[1:]:
    yprime = sir_model_euler(y, time, beta, gamma)
    y_new = [y0 + dt*yprime0 for y0, yprime0 in zip(y, yprime)]
    y = y_new

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axisbelow=True)
ax.plot(t, y[0], 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, y[1], 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, y[2], 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
plt.show()
