import numpy as np
from scipy.integrate import odeint

# Define the SEIR model

def seir_model(y, t, beta, sigma, gamma):
    S, E, I, R = y
    N = S + E + I + R
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

# Set the parameters

beta = 0.2
sigma = 0.1
gamma = 0.05

# Set the initial conditions

S0 = 990
E0 = 10
I0 = 0
R0 = 0

# Set the time grid

t = np.linspace(0, 100, 1000)

# Solve the ODE system

y0 = [S0, E0, I0, R0]
seir_solution = odeint(seir_model, y0, t, args=(beta, sigma, gamma))

# Plot the results

import matplotlib.pyplot as plt

plt.plot(t, seir_solution[:, 0], label='S')
plt.plot(t, seir_solution[:, 1], label='E')
plt.plot(t, seir_solution[:, 2], label='I')
plt.plot(t, seir_solution[:, 3], label='R')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
