import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Function implementing the SIR model

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


# Parameters

beta = 0.2
# transmission rate


gamma = 0.1
# recovery rate


S0 = 0.99
# initial susceptible population


I0 = 0.01
# initial infected population


R0 = 0
# initial recovered population


t = np.linspace(0, 100, 1000)
# time points


y0 = [S0, I0, R0]
# initial conditions


# Solve the differential equations

sol = odeint(sir_model, y0, t, args=(beta, gamma))


# Plot the results

plt.figure(figsize=(10, 6))
plt.plot(t, sol[:, 0], 'b', label='Susceptible')
plt.plot(t, sol[:, 1], 'r', label='Infected')
plt.plot(t, sol[:, 2], 'g', label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.grid()
plt.show()

