import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that contains the model dynamics
def model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# number of individuals
N = 1000

# initial number of infected and recovered individuals
I0, R0 = 1, 0

# initial number of susceptible individuals
S0 = N - I0 - R0

# model parameters
beta, gamma = 0.2, 1./10

# initial conditions vector
y0 = S0, I0, R0

# time grid (in days)
t = np.linspace(0, 160, 160)

# solve ODE
result = odeint(model, y0, t, args=(N, beta, gamma))

# plot
S, I, R = result.T
fig = plt.figure(figsize=(6,4))
plt.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
plt.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
plt.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
plt.xlabel('Time /days')
plt.ylabel('Number (1000s)')
plt.legend()
plt.show()
