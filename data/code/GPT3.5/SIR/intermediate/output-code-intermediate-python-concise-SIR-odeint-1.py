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


# Set the initial conditions
N = 1000
I0 = 1
R0 = 0
S0 = N - I0 - R0

# Set the parameters
beta = 0.2
gamma = 0.1

# Set the time points to simulate
t = np.linspace(0, 100, 100)

# Solve the differential equations
sol = odeint(sir_model, [S0, I0, R0], t, args=(beta, gamma))

# Plot the results
plt.plot(t, sol[:, 0], label='S')
plt.plot(t, sol[:, 1], label='I')
plt.plot(t, sol[:, 2], label='R')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

