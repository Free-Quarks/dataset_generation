import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# SIR model

# The differential equations describing the SIR model

def SIR(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Initial conditions
S0 = 999
I0 = 1
R0 = 0
y0 = [S0, I0, R0]

# Time vector
t = np.linspace(0, 100, 1000)

# Parameters
beta = 0.2
gamma = 0.1

# Solve the SIR model
solution = odeint(SIR, y0, t, args=(beta, gamma))

# Plotting the results
plt.plot(t, solution[:, 0], 'b', label='Susceptible')
plt.plot(t, solution[:, 1], 'r', label='Infected')
plt.plot(t, solution[:, 2], 'g', label='Recovered')
plt.title('SIR Model')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
