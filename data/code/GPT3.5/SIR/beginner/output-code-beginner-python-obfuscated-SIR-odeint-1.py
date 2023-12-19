import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the SIR model


# Function that returns dS/dt, dI/dt, dR/dt

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# Initial conditions
N = 1000  # Total population
I0 = 1  # Initial number of infected individuals
R0 = 0  # Initial number of recovered individuals
S0 = N - I0 - R0  # Initial number of susceptible individuals

# Contact rate, beta, and mean recovery rate, gamma
beta = 0.2
gamma = 0.1

# A grid of time points (in days)
t = np.linspace(0, 160, 160)

# Initial conditions vector
y0 = S0, I0, R0

# Integrate the SIR equations over the time grid, t.
result = odeint(sir_model, y0, t, args=(beta, gamma))
S, I, R = result.T

# Plot the data
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SIR Model')
plt.legend()
plt.show()
