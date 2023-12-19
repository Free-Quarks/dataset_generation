import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Define the SIR model


# Function to define the system of differential equations


def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dS_dt = -beta * S * I
    dI_dt = beta * S * I - gamma * I
    dR_dt = gamma * I
    return [dS_dt, dI_dt, dR_dt]


# Initial conditions
S0 = 999
I0 = 1
R0 = 0
y0 = [S0, I0, R0]


# Parameters
beta = 0.3
gamma = 0.1


# Time vector
t = np.linspace(0, 49, 1000)


# Solve the system of differential equations
sol = odeint(SIR_model, y0, t, args=(beta, gamma))


# Plot the results
plt.plot(t, sol[:, 0], label='S')
plt.plot(t, sol[:, 1], label='I')
plt.plot(t, sol[:, 2], label='R')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()

