import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the SEIR model

def seir_model(y, t, N, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# Set the parameter values

N = 1000  # Total population
beta = 0.2  # Contact rate
D = 10.0  # Infectious period
sigma = 1.0 / 5.0  # Incubation period
gamma = 1.0 / D

# Initial conditions

S0, E0, I0, R0 = N-1, 1, 0, 0  # One exposed individual

# Time vector

t = np.linspace(0, 49, 50)

# Integrate the SEIR equations over the time grid

y0 = S0, E0, I0, R0

ret = odeint(seir_model, y0, t, args=(N, beta, gamma, sigma))
S, E, I, R = ret.T

# Plot the data

fig, ax = plt.subplots()
ax.plot(t, S, 'b', label='Susceptible')
ax.plot(t, E, 'y', label='Exposed')
ax.plot(t, I, 'r', label='Infected')
ax.plot(t, R, 'g', label='Recovered')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Population')
ax.set_ylim(0, N)
ax.legend()
plt.show()

