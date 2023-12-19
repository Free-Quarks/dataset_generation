import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that returns ds/dt, di/dt, dr/dt

def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# initial conditions
y0 = [0.99, 0.01, 0]

# time points
t = np.linspace(0, 100, 1000)

# parameters
beta = 0.2
gamma = 0.1

# solve ODE
sol = odeint(SIR_model, y0, t, args=(beta, gamma))

# plot results
plt.plot(t, sol[:, 0], 'b', label='S(t)')
plt.plot(t, sol[:, 1], 'g', label='I(t)')
plt.plot(t, sol[:, 2], 'r', label='R(t)')
plt.xlabel('Time')
plt.ylabel('Proportions')
plt.legend()
plt.show()
