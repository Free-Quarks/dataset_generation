import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

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

# Solve the SIR model
sol = odeint(SIR_model, [S0, I0, R0], t, args=(beta, gamma))

# Plot the results
plt.plot(t, sol[:, 0], label='S(t)')
plt.plot(t, sol[:, 1], label='I(t)')
plt.plot(t, sol[:, 2], label='R(t)')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

