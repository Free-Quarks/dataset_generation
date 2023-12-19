import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that returns dy/dt

def model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions
N = 1000
I0 = 1
R0 = 0
S0 = N - I0 - R0

# Parameters
beta = 0.2
gamma = 0.1

# Time vector
t = np.linspace(0, 100, 100)

# Initial condition vector
y0 = S0, I0, R0

# Integrate the SIR equations over the time grid
result = odeint(model, y0, t, args=(N, beta, gamma))

# Plot
plt.plot(t, result[:, 0], label='Susceptible')
plt.plot(t, result[:, 1], label='Infected')
plt.plot(t, result[:, 2], label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
