import numpy as np
import matplotlib.pyplot as plt
import json

# Parameters
N = 1000    # Total population
I0 = 1      # Initial number of infected individuals
R0 = 0      # Initial number of recovered individuals
S0 = N - I0 - R0   # Initial number of susceptible individuals
beta = 0.2  # Contact rate
gamma = 0.1 # Recovery rate

# Time grid (in days)
t = np.linspace(0, 160, 160)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0

# RK2 method
def rk2(y, t, dt, derivs):
    k1 = np.asarray(derivs(y, t, N, beta, gamma))
    k2 = np.asarray(derivs(y + 0.5*dt*k1, t + 0.5*dt, N, beta, gamma))
    y_next = y + dt*k2
    return y_next

# Time integration using RK2
y = np.zeros([len(t), 3])
y[0, :] = y0
dt = t[1] - t[0]
for i in range(len(t)-1):
    y[i+1, :] = rk2(y[i, :], t[i], dt, deriv)

S, I, R = y[:,0], y[:,1], y[:,2]

# Plotting the data
plt.figure(figsize=[6,4])
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of people')
plt.legend()
plt.show()
