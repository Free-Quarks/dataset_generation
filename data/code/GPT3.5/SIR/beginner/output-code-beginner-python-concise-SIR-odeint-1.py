import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Function that defines the SIR model


def SIR_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# Parameters
N = 1000
beta = 0.2
D = 10
gamma = 1.0 / D

# Initial conditions
I0, R0 = 1, 0
S0 = N - I0 - R0

# Time vector
t = np.linspace(0, 100, 100)

# Initial conditions vector
y0 = S0, I0, R0

# Integrate the SIR equations over the time grid t
result = odeint(SIR_model, y0, t, args=(N, beta, gamma))
S, I, R = result.T

# Plot the data
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
