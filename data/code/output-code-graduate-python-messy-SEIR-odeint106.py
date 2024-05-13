import numpy as np
from scipy.integrate import odeint


# Define the model

def seir_model(y, t, N, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


# Define the parameters

N = 1000
beta = 0.2
sigma = 1/5
gamma = 1/10


# Define the initial conditions

S0, E0, I0, R0 = N-1, 1, 0, 0


# Define the time grid

t = np.linspace(0, 49, 50)


# Solve the ODE

y0 = S0, E0, I0, R0


sol = odeint(seir_model, y0, t, args=(N, beta, gamma, sigma))


# Plot the results

import matplotlib.pyplot as plt


plt.plot(t, sol[:, 0], label='S')
plt.plot(t, sol[:, 1], label='E')
plt.plot(t, sol[:, 2], label='I')
plt.plot(t, sol[:, 3], label='R')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()
