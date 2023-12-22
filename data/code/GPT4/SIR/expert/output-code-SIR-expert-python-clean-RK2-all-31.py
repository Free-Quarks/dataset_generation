import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import json

def model_SIR_RK2(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Initial conditions
S0 = 0.999
I0 = 0.001
R0 = 0.0
beta = 0.35
gamma = 0.1

# Time points
t = np.linspace(0, 100, 500)

# Solve ODE
y0 = [S0, I0, R0]
sol = odeint(model_SIR_RK2, y0, t, args=(beta, gamma))

# Plot results
plt.figure(figsize=[6,4])
plt.plot(t, sol[:, 0], 'b', label='S(t)')
plt.plot(t, sol[:, 1], 'r', label='I(t)')
plt.plot(t, sol[:, 2], 'g', label='R(t)')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Proportions')
plt.title('SIR model with RK2')
plt.grid(True)
plt.show()
