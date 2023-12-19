import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that returns dy/dt

def SIR(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Initial conditions
S0 = 0.99
I0 = 0.01
R0 = 0.0

# Parameters
beta = 0.2
gamma = 0.1

# Time vector
t = np.linspace(0, 100, 1000)

# Solve ODE
y = odeint(SIR, [S0, I0, R0], t, args=(beta, gamma))

# Plot results
plt.plot(t, y[:, 0], 'b', label='S(t)')
plt.plot(t, y[:, 1], 'r', label='I(t)')
plt.plot(t, y[:, 2], 'g', label='R(t)')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.title('SIR Model')
plt.legend()
plt.show()
