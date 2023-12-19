import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that contains the system of differential equations

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Model parameters
beta = 0.2
gamma = 0.1

# Initial conditions
S0 = 0.99
I0 = 0.01
R0 = 0.0
y0 = S0, I0, R0

# Time vector
t = np.linspace(0, 100, 1000)

# Solve the system of differential equations
sol = odeint(sir_model, y0, t, args=(beta, gamma))

# Plot the results
plt.plot(t, sol[:, 0], label='S(t)')
plt.plot(t, sol[:, 1], label='I(t)')
plt.plot(t, sol[:, 2], label='R(t)')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
